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
from app.counsel_tools import build_counsel_tools
from app.vectordb import load_vectorstore


load_dotenv()

VALID_DOMAINS = {"criminal", "housing", "labor", "traffic", "civil", "all"}


SUPERVISOR_PROMPT = (
    "당신은 사용자의 법률 질문을 적절한 전문 에이전트로 배정하는 라우터입니다.\n"
    "질문의 성격을 판단하여 destination을 결정하세요.\n\n"
    "- research: 처벌·형량·고소·소송·법적 책임·정당방위·손해배상 등 "
    "법적 판단이나 구체적 법조문·판례가 필요한 질문. "
    "형사사건(폭행, 상해, 사기, 절도, 성범죄, 음주운전 등), 민사 분쟁, "
    "이혼·상속 등 법적 다툼이 관련되면 반드시 research를 선택하세요.\n"
    "- counsel: 신고·등록·신청·발급 등 일상적인 행정 절차나 제도 안내가 "
    "필요한 질문 (출생신고, 전입신고, 각종 지원제도 신청 방법 등).\n\n"
    "핵심 기준: 누군가의 법적 책임을 따지거나, 처벌·소송·고소가 얽혀 있으면 research. "
    "단순히 '어떻게 신청/등록/신고하나요'류의 절차 안내면 counsel.\n"
    "판단이 애매하면 research를 선택하세요."
)

SINGLE_PROMPT = (
    "당신은 법률 질문에 답하는 어시스턴트입니다. "
    "사용자가 변호사를 만나기 전에 스스로 상황을 이해하도록 구체적이고 실질적인 정보를 주세요.\n"
    "search_law(법조문), search_case(판례), search_qa(생활법령 상담) 도구를 "
    "질문에 맞게 조합해 사용하세요.\n\n"
    "[검색 방법]\n"
    "- 검색어에는 질문의 법률 쟁점을 가리키는 용어를 반드시 넣으세요. "
    "예: 유언으로 한 자녀만 상속 -> '유류분', 이사 전 보증금 미반환 -> '임차권등기명령', "
    "재범 음주운전 -> '음주운전 가중처벌'. 질문 문장을 그대로 넣지 마세요.\n"
    "- domain/category는 확신이 있을 때만 좁히고, 조금이라도 애매하면 'all'을 쓰세요. "
    "잘못 좁히면 정작 필요한 자료가 검색되지 않습니다.\n"
    "- 절차·서류·기한·신고 방법 안내가 필요한 질문이면 search_qa를 반드시 포함하세요.\n"
    "- 거기에 처벌·책임·소송·손해배상이 얽혀 있으면, search_qa를 빼지 말고"
    "그대로 둔 채 search_law(근거 조문)와 search_case(법원의 판단 기준)를 "
    "추가로 호출하세요. 도구를 2개만 골라야 하는 것이 아니라, "
    "필요하면 3개를 모두 쓰는 것이 정상입니다.\n"
    "- 도구 호출은 최대 4회까지만. 원하는 자료가 안 나와도 더 뒤지지 말고 "
    "확보한 정보로 답하세요.\n\n"
    "[근거 제시]\n"
    "- 법조문을 근거로 쓸 때는 검색 결과의 출처를 그대로 인용하세요 "
    "(예: 주택임대차보호법 제3조의3). 조문 번호를 기억에 의존해 쓰지 말고 "
    "검색 결과에 적힌 것을 쓰세요. 가지번호(제148조의2)도 정확히 옮기세요.\n"
    "- 관련 판례를 찾았다면 사건번호와 함께 그 판례가 어떤 기준을 제시했는지 "
    "한두 문장으로 소개하세요. 상담성 질문이라도 마찬가지입니다.\n"
    "- 검색 결과에 없는 조문 번호·사건번호·형량 수치는 지어내지 마세요. "
    "확인되지 않으면 수치를 빼고 설명하세요.\n\n"
    "답변은 핵심 위주로 간결하게, 이모지·과도한 머리기호 없이. "
    "'변호사와 상담하라'는 정말 필요할 때 맨 끝에 한 번만.\n\n"
    "[출력 형식 - 반드시 지킬 것]\n"
    "답변의 첫 글자부터 곧바로 질문에 대한 답 내용이어야 합니다. "
    "검색 진행 상황, 검색 결과에 대한 소감, 답변을 시작한다는 예고를 "
    "도구 호출 전에도 후에도 절대 쓰지 마세요.\n"
    "금지 예시: '검색하겠습니다', '확인해보겠습니다', '자료를 더 찾아보겠습니다', "
    "'완벽합니다', '좋습니다', '이제 답변드리겠습니다', '충분한 정보를 얻었습니다'\n"
    "이런 문장은 한 마디도 출력하지 말고, 바로 본론을 시작하세요."
)

RESEARCH_PROMPT = (
    "당신은 법조문과 판례를 근거로 답하는 법률 어시스턴트입니다.\n"
    "당신의 역할은 사용자가 변호사를 만나기 전에 스스로 상황을 이해하고 "
    "무엇을 준비해야 할지 파악하도록, 구체적이고 실질적인 정보를 주는 것입니다.\n\n"
    "search_law 도구로 관련 법조문을, search_case 도구로 관련 판례를 검색할 수 있습니다. "
    "두 도구 모두 domain 파라미터로 범위를 좁힐 수 있습니다:\n"
    "- criminal: 형사 (살인, 폭행, 상해, 절도, 사기, 성범죄, 스토킹, 형사절차 등)\n"
    "- housing: 임대차 (전월세, 보증금, 계약갱신, 상가임대차 등)\n"
    "- labor: 노동 (임금, 해고, 퇴직금, 근로시간, 산재 등)\n"
    "- traffic: 교통 (음주운전, 교통사고, 자동차 손해배상 등)\n"
    "- civil: 민사 (계약, 채권채무, 손해배상, 상속, 이혼, 부동산, 민사소송 등)\n"
    "- all: 분야가 불분명하거나 여러 분야에 걸친 경우\n"
    "질문 성격에 맞는 domain을 지정하고, 애매하면 all을 쓰세요. "
    "법조문 해석에는 search_law, 구체적 판단 기준·사례에는 search_case를 사용하세요.\n\n"
    "답변 방식:\n"
    "- 관련 법조문, 성립 요건, 처벌 수준, 대응 절차, 필요한 증거 등을 "
    "근거와 함께 구체적으로 설명하세요.\n"
    "- 도구를 호출하기 전이든 결과를 받은 뒤든, '검색하겠습니다'·'완벽합니다'·"
    "'이제 답변드리겠습니다' 같은 사고 과정·진행 상황·소감은 절대 출력하지 말고, "
    "검색이 끝난 뒤 최종 답변 본문만 제시하세요.\n"
    "- 검색 결과가 부족하더라도 '자료를 찾기 어렵다' 같은 말은 하지 말고, "
    "확보된 정보와 일반적인 법률 지식으로 실질적인 답을 제시하세요.\n"
    "- 답변은 핵심 위주로 간결하게. 이모지나 과도한 제목·머리기호는 쓰지 마세요.\n"
    "- '변호사와 상담하라'는 말은 정말 필요한 경우 답변 맨 끝에 한 번만 덧붙이고, "
    "중간에 반복하거나 그 말로 설명을 대체하지 마세요.\n"
    "- 근거로 삼은 자료가 있으면 출처(source)를 밝히세요."
)

COUNSEL_PROMPT = (
    "당신은 생활 속 법률 문제를 쉽게 안내하는 법률 상담 어시스턴트입니다.\n"
    "당신의 역할은 사용자가 스스로 상황을 이해하고 무엇을 준비해야 할지 "
    "파악하도록, 구체적이고 실질적인 정보를 주는 것입니다.\n\n"
    "search_qa 도구로 법제처 생활법령에서 관련 상담 자료를 검색할 수 있습니다. "
    "일상적인 절차·신청·제도에 대해 일반인이 이해하기 쉽게 안내하세요.\n\n"
    "답변 방식:\n"
    "- 실제로 해야 할 절차, 필요한 서류, 기한, 주의사항을 구체적으로 안내하세요.\n"
    "- 도구를 호출하기 전이든 결과를 받은 뒤든, '검색하겠습니다'·'완벽합니다'·"
    "'이제 답변드리겠습니다' 같은 진행 상황이나 소감은 절대 출력하지 말고, "
    "검색이 끝난 뒤 최종 답변 본문만 제시하세요.\n"
    "- 검색 결과가 부족하더라도 '자료를 찾기 어렵다' 같은 말은 하지 말고, "
    "확보된 정보와 일반적인 지식으로 실질적인 답을 제시하세요.\n"
    "- 답변은 핵심만 간결하게, 5~10문장 내외로. 이모지나 과도한 제목·머리기호는 "
    "쓰지 말고 자연스러운 문장으로 답하세요.\n"
    "- '변호사·전문가와 상담하라'는 말은 정말 필요한 경우 맨 끝에 한 번만 덧붙이고, "
    "그 말로 설명을 대체하지 마세요.\n"
    "- 근거로 삼은 자료가 있으면 출처(source)를 밝히세요."
)


def build_llm():
    provider=os.getenv("LLM_PROVIDER", "anthropic").lower()
    # 법률 답변은 같은 질문에 같은 근거·결론이 나와야 하므로 기본값을 0으로 둔다.
    # (미지정 시 Anthropic 기본값은 1.0이라 실행마다 답이 크게 달라진다)
    temperature=float(os.getenv("LLM_TEMPERATURE", "0"))
    print(f"LLM Provider:{provider} (temperature={temperature})")
    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=True,
            temperature=temperature,
        )
    else:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL","gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
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

def _make_summarizer(raw_llm):
    def summarize_messages(state: State):
        messages = state["messages"]
        if len(messages) <= 20:
            return {}
        cut = len(messages) - 10
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
    return summarize_messages

def build_rag_graph(variant: str = "current", use_checkpointer: bool = True):
    """
    variant:
      - "current": 라우팅 + 배타적 도구 (research=[law,case], counsel=[qa])
      - "shared" : 라우팅 + 도구 공유 (counsel에 law,case 추가)
      - "single" : 라우팅 없음, 단일 에이전트가 전체 도구 사용
    """
    vectorstore = load_vectorstore()
    raw_llm = build_llm()

    # 도구 준비
    research_tools = build_research_tools(vectorstore)   # [search_law, search_case]
    qa_tools = build_counsel_tools()                     # [search_qa]

    checkpointer = None
    if use_checkpointer:
        checkpointer = AsyncSqliteSaver(
            aiosqlite.connect("var/checkpoints.db", check_same_thread=False)
        )

    # ── variant: single (라우팅 없는 단일 에이전트) ──
    if variant == "single":
        all_tools = [*research_tools, *qa_tools]
        single_agent = build_agent_subgraph(raw_llm, all_tools, SINGLE_PROMPT)

        g = StateGraph(State)
        g.add_node("summarize", _make_summarizer(raw_llm))
        g.add_node("single_agent", single_agent)
        g.add_edge(START, "summarize")
        g.add_edge("summarize", "single_agent")
        g.add_edge("single_agent", END)
        return g.compile(checkpointer=checkpointer)

    # ── variant: current / shared (멀티에이전트 + 라우팅) ──
    if variant == "shared":
        counsel_tool_list = [*qa_tools, *research_tools]  # qa + law + case
    else:  # current
        counsel_tool_list = qa_tools                      # qa only

    research_agent = build_agent_subgraph(raw_llm, research_tools, RESEARCH_PROMPT)
    counsel_agent = build_agent_subgraph(raw_llm, counsel_tool_list, COUNSEL_PROMPT)

    router_llm = raw_llm.with_structured_output(Route)

    def supervisor(state: State):
        decision = router_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            state["messages"][-1],
        ])
        return {"route": decision.destination}

    def route_edge(state: State) -> Literal["research_agent", "counsel_agent"]:
        return "research_agent" if state["route"] == "research" else "counsel_agent"

    g = StateGraph(State)
    g.add_node("summarize", _make_summarizer(raw_llm))
    g.add_node("supervisor", supervisor)
    g.add_node("research_agent", research_agent)
    g.add_node("counsel_agent", counsel_agent)

    g.add_edge(START, "summarize")
    g.add_edge("summarize", "supervisor")
    g.add_conditional_edges("supervisor", route_edge)
    g.add_edge("research_agent", END)
    g.add_edge("counsel_agent", END)

    return g.compile(checkpointer=checkpointer)