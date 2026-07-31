"""각 variant를 실행해 답변·사용도구·라우팅을 추출하는 target 함수."""
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph import build_rag_graph

# variant별 그래프를 미리 빌드 (checkpointer 없이 - 평가는 독립 실행)
_graphs = {}

def get_graph(variant):
    if variant not in _graphs:
        _graphs[variant] = build_rag_graph(variant, use_checkpointer=False)
    return _graphs[variant]


def extract_tools_used(messages):
    """messages에서 호출된 도구 이름 목록 추출."""
    tools = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tools.append(tc["name"])
    return tools


def extract_final_answer(messages):
    """마지막 AIMessage(도구 호출 없는 최종 답변) 추출."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            content = m.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if content:
                return content
    return ""


def make_target(variant):
    """LangSmith evaluate에 넘길 async target 함수 생성."""
    graph = get_graph(variant)

    async def target(inputs: dict) -> dict:
        question = inputs["question"]
        start = time.time()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        elapsed = time.time() - start

        messages = result["messages"]
        tools_used = extract_tools_used(messages)
        answer = extract_final_answer(messages)

        # 라우팅 결과: state의 route (single은 없음)
        route = result.get("route", "none")

        return {
            "answer": answer,
            "tools_used": tools_used,
            "route": route,
            "latency_sec": round(elapsed, 2),
        }

    return target