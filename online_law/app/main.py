from contextlib import asynccontextmanager
import json
import os
import secrets
import uuid
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
from langchain_core.messages import HumanMessage, AIMessage

from app import db
from app.graph import build_rag_graph

load_dotenv()
PASSCODE=os.getenv("PASSCODE","")

def content_to_text(content)->str:
    if isinstance(content,str):
        return content
    if isinstance(content, list):
        parts=[]
        for part in content:
            if isinstance(part,str):
                parts.append(part)
            elif isinstance(part,dict) and part.get("type")=="text":
                parts.append(part.get("text",""))
        return "".join(parts)
    return str(content)
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    app.state.rag_graph=build_rag_graph("single")
    yield
app=FastAPI(lifespan=lifespan)
class QueryRequest(BaseModel):
    question: str
    thread_id: str | None=None
class QueryResponse(BaseModel):
    answer: str
    thread_id: str
class AuthRequest(BaseModel):
    passcode: str=""

def get_current_user(authorization: str = Header(default="")) -> int:
    """Authorization: Bearer <token> 헤더로 user_id 반환. 실패 시 401."""
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    user_id = db.get_user_by_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user_id

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/register")
def register(req: RegisterRequest):
    ok, msg = db.register_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg}

@app.post("/api/login")
def login(req: LoginRequest):
    token = db.login_user(req.username, req.password)
    if token is None:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    return {"token": token}

def check_passcode(passcode: str)->bool:
    return not PASSCODE or secrets.compare_digest(passcode, PASSCODE)
def require_passcode(x_passcode: str=Header(default="")):
    if not check_passcode(x_passcode):
        raise HTTPException(status_code=401, detail="Invalid passcode")
@app.post("/api/query/stream")
async def query_stream(req: QueryRequest, user_id: int = Depends(get_current_user)):
    thread_id = req.thread_id or str(uuid.uuid4())

    # 남의 thread 접근 방지: 기존 thread면 소유자 확인
    if req.thread_id and not db.owns_thread(user_id, req.thread_id):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")

    # 대화 기록 (첫 질문을 제목으로)
    db.upsert_conversation(thread_id, user_id, req.question)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def event_stream():
        yield sse({"thread_id": thread_id})
        # 이번 LLM 턴에서 사용자에게 이미 내보낸 토큰이 있는지.
        # 도구 호출 턴으로 판명되면 reset을 보내 프론트에서 폐기하게 한다.
        sent_in_turn = False
        try:
            async for event in app.state.rag_graph.astream_events(
                {"messages": [HumanMessage(content=req.question)]},
                config={"configurable": {"thread_id": thread_id}},
                version="v2",
            ):
                # 요약 노드의 LLM 출력은 내부용이므로 사용자에게 보내지 않는다
                if event.get("metadata", {}).get("langgraph_node") == "summarize":
                    continue

                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "tool_call_chunks", None):
                        # 이 턴은 최종 답변이 아니라 도구 호출 턴이었다
                        if sent_in_turn:
                            yield sse({"reset": True})
                            sent_in_turn = False
                        continue
                    token = content_to_text(chunk.content)
                    if token:
                        sent_in_turn = True
                        yield sse({"token": token})
                elif kind == "on_chat_model_end":
                    sent_in_turn = False
        except Exception as e:
            yield sse({"error": str(e)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive",
    })

@app.get("/api/conversations")
def get_conversations(user_id: int = Depends(get_current_user)):
    return {"conversations": db.list_conversations(user_id)}


@app.delete("/api/conversations/{thread_id}")
async def delete_conversation(thread_id: str, user_id: int = Depends(get_current_user)):
    if not db.owns_thread(user_id, thread_id):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    db.delete_conversation(user_id, thread_id)
    checkpointer = app.state.rag_graph.checkpointer
    if checkpointer is not None:
        await checkpointer.adelete_thread(thread_id)
    return {"ok": True}


@app.get("/api/conversations/{thread_id}")
async def get_conversation(thread_id: str, user_id: int = Depends(get_current_user)):
    if not db.owns_thread(user_id, thread_id):
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    graph = app.state.rag_graph
    state = await graph.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    messages = state.values.get("messages", []) if state else []
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": content_to_text(m.content)})
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            text = content_to_text(m.content)
            if text:
                history.append({"role": "assistant", "content": text})
    return {"thread_id": thread_id, "messages": history}


app.mount("/",StaticFiles(directory="static",html=True), name="static")