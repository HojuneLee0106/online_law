import os
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict
import sqlite3
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from vectordb import build_vectorstore
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()
SYSTEM_PROMPT = (
    "당신은 법조문과 판례를 근거로 답하는 법률 검색 어시스턴트입니다.\n"
    "search_law 도구로 관련 법조문을, search_case 도구로 관련 판례를 검색할 수 있습니다.\n"
    "질문에 따라 필요한 도구를 선택해서 호출하세요. 법조문 해석에는 search_law를, "
    "구체적 사건이나 판단 기준이 필요하면 search_case를 사용하세요.\n"
    "충분한 근거를 확보한 뒤에는 반드시 근거 문서의 출처(source)를 밝히며 답하세요. "
    "근거가 부족하면 '해당하는 자료가 없습니다.'라고 답하세요."
)


def build_llm():
    provider=os.getenv("LLM_PROVIDER", "anthropic").lower()
    print(f"LLM Provider:{provider}")
    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
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


def build_tools(vectorstore):
    @tool
    def search_law(query: str)->str:
        """법조문(법률 조항)을 검색합니다. 특정 법률의 조문 내용이나 정의가 필요할 때 사용하세요."""
        docs=vectorstore.similarity_search(query, k=3, filter={"doc_type":"law"})
        if not docs:
            return "관련 법조문을 찾을 수 없습니다."
        return format_docs(docs)
    
    @tool
    def search_case(query: str)->str:
        """판례를 검색합니다. 구체적 사건의 판단 기준, 법원의 해석, 유사 사례가 필요할 때 사용하세요."""
        docs=vectorstore.similarity_search(query, k=3, filter={"doc_type":"case"})
        if not docs:
            return "관련 판례를 찾을 수 없습니다."
        return format_docs(docs)
    return [search_law, search_case]

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_rag_graph(use_api_law: bool=False):
    vectorstore=build_vectorstore()
    tools=build_tools(vectorstore)
    raw_llm=build_llm()
    llm=raw_llm.bind_tools(tools)
    
    def summarize_messages(state: State):
        messages=state["messages"]
        if len(messages)>20:
            old_messages=messages[:-10]
            summary=raw_llm.invoke([
                SystemMessage(content="다음 대화를 3문장으로 요약해줘"),
                *old_messages
            ])
            return {"messages": [RemoveMessage(id=m.id) for m in old_messages] + [summary]}
        return {}
    def agent(state: State):
        messages=state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages=[SystemMessage(content=SYSTEM_PROMPT), *messages]
        response=llm.invoke(messages)
        return {"messages":[response]}
    graph_builder=StateGraph(State)
    graph_builder.add_node("summarize",summarize_messages)
    graph_builder.add_node("agent",agent)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")


    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph_builder.compile(checkpointer=checkpointer)