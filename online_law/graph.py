import os
import sqlite3
from typing import Annotated, Literal
 
from dotenv import load_dotenv
from typing_extensions import TypedDict
 
from pydantic import BaseModel, Field
 
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from counsel_tools import build_counsel_tools
from vectordb import load_vectorstore


load_dotenv()

VALID_DOMAINS = {"criminal", "housing", "labor", "traffic", "civil", "all"}


RESEARCH_PROMPT = (
    "당신은 법조문과 판례를 근거로 답하는 법률 검색 어시스턴트입니다.\n"
    "search_law 도구로 관련 법조문을, search_case 도구로 관련 판례를 검색할 수 있습니다.\n"
    "두 도구 모두 domain 파라미터로 검색 범위를 좁힐 수 있습니다:\n"
    "- criminal: 형사 (살인, 폭행, 절도, 사기, 성범죄, 스토킹, 형사절차 등)\n"
    "- housing: 임대차 (전월세, 보증금, 계약갱신, 상가임대차 등)\n"
    "- labor: 노동 (임금, 해고, 퇴직금, 근로시간, 산재, 직장 내 차별 등)\n"
    "- traffic: 교통 (음주운전, 교통사고, 자동차 손해배상 등)\n"
    "- civil: 민사 (계약, 채권채무, 손해배상, 상속, 부동산, 민사소송, 파산회생 등)\n"
    "- all: 분야가 불분명하거나 여러 분야에 걸친 경우\n\n"
    "질문의 성격에 맞는 domain을 지정하면 더 정확한 결과를 얻을 수 있습니다. "
    "분야가 애매하면 all을 사용하세요.\n"
    "법조문 해석에는 search_law를, 구체적 사건이나 판단 기준이 필요하면 search_case를 사용하세요.\n"
    "답변은 간결하게, 핵심만 5~10문장 내외로 정리하세요. "
    "이모지나 과도한 제목·머리기호는 쓰지 말고, 관련 조문과 처벌 기준 위주로 자연스러운 문장으로 답하세요.\n"
    "충분한 근거를 확보한 뒤에는 반드시 근거 문서의 출처(source)를 밝히며 답하세요. "
    "근거가 부족하면 '해당하는 자료가 없습니다.'라고 답하세요."
)
 
COUNSEL_PROMPT = (
    "당신은 생활 속 법률 문제를 쉽게 안내하는 법률 상담 어시스턴트입니다.\n"
    "search_qa 도구로 법제처 생활법령에서 관련 상담 사례를 검색할 수 있습니다.\n"
    "일상적인 법률 문제에 대해 이해하기 쉬운 언어로 대처 방법과 절차를 안내하세요.\n"
    "답변은 간결하게, 핵심만 5~10문장 내외로 정리하세요. "
    "이모지나 과도한 제목·머리기호는 쓰지 말고, 자연스러운 문장으로 답하세요. "
    "긴 목록 나열보다 사용자가 실제로 해야 할 핵심 행동 위주로 안내하세요.\n"
    "검색 결과를 근거로 답하되 반드시 출처(source)를 밝히고, "
    "근거가 부족하면 '해당하는 자료가 없습니다.'라고 답하세요."
)
 
SUPERVISOR_PROMPT = (
    "당신은 사용자의 법률 질문을 적절한 전문 에이전트로 배정하는 라우터입니다.\n"
    "질문의 성격을 판단하여 destination을 결정하세요.\n"
    "- research: 구체적인 법조문·판례·처벌 기준·법적 근거가 필요한 질문 "
    "(예: '음주운전 3회 처벌은?', '부작위 살인 성립요건')\n"
    "- counsel: 생활 속 절차·대처 방법을 묻는 상담성 질문 "
    "(예: '전세금을 못 받았는데 어떻게 하나요?', '해고당했는데 뭘 해야 하죠?')\n"
    "판단이 애매하면 counsel을 선택하세요."
)


def build_llm():
    provider=os.getenv("LLM_PROVIDER", "anthropic").lower()
    print(f"LLM Provider:{provider}")
    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL","gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

def format_docs(docs: list[Document])->str:
    parts=[]
    for doc in docs:
        source=doc.metadata.get("source", "출처 미상")
        parts.append(f"[출처: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)

def _build_filter(doc_type: str, domain: str)->dict:
    """doc_type과 domain을 조합한 Chroma 필터 생성."""
    if domain=="all":
        return {"doc_type": doc_type}
    return {"$and":[
        {"doc_type":{"$eq":doc_type}},
        {"domain":{"$eq":domain}},
    ]}

def build_research_tools(vectorstore):
    @tool
    def search_law(query:str, domain: str="all")->str:
        """법조문(법률 조항)을 검색합니다.

        Args:
            query: 검색할 내용 (예: '음주운전 처벌 기준', '보증금 반환')
            domain: 검색 범위. criminal(형사) / housing(임대차) / labor(노동) /
                    traffic(교통) / civil(민사) / all(전체)
        """
        if domain not in VALID_DOMAINS:
            return (
                f"'{domain}'은 유효하지 않은 분야입니다. "
                f"다음 중 하나를 사용하세요: {', '.join(sorted(VALID_DOMAINS))}"
            )
        docs=vectorstore.similarity_search(
            query, k=3, filter=_build_filter("law", domain)
        )
        if not docs:
           return f"'{domain}' 분야에서 관련 법조문을 찾을 수 없습니다."
        return format_docs(docs)
    @tool
    def search_case(query: str, domain: str = "all") -> str:
        """판례를 검색합니다. 구체적 사건의 판단 기준, 법원의 해석, 유사 사례가 필요할 때 사용하세요.

        Args:
            query: 검색할 내용 (예: '부작위 살인 성립요건')
            domain: 검색 범위. criminal(형사) / housing(임대차) / labor(노동) /
                    traffic(교통) / civil(민사) / all(전체)
        """
        if domain not in VALID_DOMAINS:
            return (
                f"'{domain}'은 유효하지 않은 분야입니다. "
                f"다음 중 하나를 사용하세요: {', '.join(sorted(VALID_DOMAINS))}"
            )
        docs=vectorstore.similarity_search(
            query, k=3, filter=_build_filter("case", domain)
        )
        if not docs:
            return f"'{domain}' 분야에서 관련 판례를 찾을 수 없습니다."
        return format_docs(docs)
    return [search_law, search_case]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    route:str

class Route(BaseModel):
    """ 질문을 처리할 에이전트를 결정한다."""
    destination: Literal["research","counsel"]=Field(
        description=(
            "research: 구체적 법조문·판례·처벌기준·법적 근거가 필요한 질문. "
            "counsel: 생활 속 절차·대처방법을 묻는 상담성 질문."
        )
    )

def build_agent_subgraph(raw_llm, tools, system_prompt):
    """agent + tools 루프를 가진 독립 서브그래프. 체크포인터는 상위 그래프가 관리."""
    llm=raw_llm.bind_tools(tools)
    def agent(state: State):
        messages = state["messages"]
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages = [SystemMessage(content=system_prompt), *messages]
        response = llm.invoke(messages)
        return {"messages": [response]}
    sub=StateGraph(State)
    sub.add_node("agent", agent)
    sub.add_node("tools", ToolNode(tools))
    sub.add_edge(START, "agent")
    sub.add_conditional_edges("agent", tools_condition)
    sub.add_edge("tools", "agent")
    return sub.compile()


def build_rag_graph(use_api_law: bool = False):
    vectorstore = load_vectorstore()
    raw_llm = build_llm()
 
    research_agent = build_agent_subgraph(
        raw_llm, build_research_tools(vectorstore), RESEARCH_PROMPT
    )
    counsel_agent = build_agent_subgraph(
        raw_llm, build_counsel_tools(), COUNSEL_PROMPT
    )
 
    router_llm = raw_llm.with_structured_output(Route)
 
    def summarize_messages(state: State):
        messages = state["messages"]
        if len(messages) <= 20:
            return {}
        cut = len(messages) - 10
        # tool_call/tool_result 쌍이 잘리지 않게 경계 보정
        while cut < len(messages) and isinstance(messages[cut], ToolMessage):
            cut += 1
        old_messages = messages[:cut]
        if not old_messages:
            return {}
        summary = raw_llm.invoke([
            SystemMessage(content="다음 대화를 3문장으로 요약해줘"),
            *old_messages,
        ])
        return {
            "messages": [RemoveMessage(id=m.id) for m in old_messages]
            + [HumanMessage(content=f"[이전 대화 요약]\n{summary.content}")]
        }
 
    def supervisor(state: State):
        decision = router_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            state["messages"][-1],  # 최신 질문만 보고 분류
        ])
        return {"route": decision.destination}
 
    def route_edge(state: State) -> Literal["research_agent", "counsel_agent"]:
        return "research_agent" if state["route"] == "research" else "counsel_agent"
 
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize", summarize_messages)
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("research_agent", research_agent)
    graph_builder.add_node("counsel_agent", counsel_agent)
 
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "supervisor")
    graph_builder.add_conditional_edges("supervisor", route_edge)
    graph_builder.add_edge("research_agent", END)
    graph_builder.add_edge("counsel_agent", END)
 
    conn = aiosqlite.connect("checkpoints.db", check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn)
    return graph_builder.compile(checkpointer=checkpointer)