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

from graph import build_rag_graph

load_dotenv()
PASSCODE=os.getenv("PASSCODE","")
def check_passcode(passcode: str)->bool:
    return not PASSCODE or secrets.compare_digest(passcode, PASSCODE)
def require_passcode(x_passcode: str=Header(default="")):
    if not check_passcode(x_passcode):
        raise HTTPException(status_code=401, detail="Invalid passcode")
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
    app.state.rag_graph=build_rag_graph()
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
@app.post("/auth")
def auth(req: AuthRequest):
    if not check_passcode(req.passcode):
        raise HTTPException(status_code=401, detail="Invalid passcode")
    return {"ok": True}
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_passcode)])
async def query(req: QueryRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    result = await app.state.rag_graph.ainvoke(
        {"messages": [HumanMessage(content=req.question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = content_to_text(result["messages"][-1].content)
    return QueryResponse(answer=answer, thread_id=thread_id)
@app.post("/query/stream", dependencies=[Depends(require_passcode)])
async def query_stream(req: QueryRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def event_stream():
        yield sse({"thread_id": thread_id})
        try:
            async for chunk, metadata in app.state.rag_graph.astream(
                {"messages": [HumanMessage(content=req.question)]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node")
                if node in ("research_agent", "counsel_agent") and isinstance(chunk, AIMessage):
                    token = content_to_text(chunk.content)
                    if token:
                        yield sse({"token": token})
        except Exception as e:
            yield sse({"error": str(e)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx 버퍼링 끄기
            "Connection": "keep-alive",
        },
    )
app.mount("/",StaticFiles(directory="static",html=True), name="static")