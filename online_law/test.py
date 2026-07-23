import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langsmith import Client
from langsmith.evaluation import evaluate
from pydantic import BaseModel, Field

from graph import build_rag_graph

load_dotenv()
client = Client()
DATASET_NAME = "online_law_criminal"

EXAMPLES = [
    {
        "inputs": {"question": "형법상 부작위범이 성립하려면 어떤 요건이 필요해?"},
        "outputs": {
            "expected_doc_type": "law",
            "expected_keywords": ["부작위", "위험"],
            "reference_answer": (
                "형법 제18조에 따르면, 위험 발생을 방지할 의무가 있거나 자기 행위로 "
                "위험발생의 원인을 야기한 자가 그 위험을 방지하지 않은 경우, "
                "발생한 결과에 따라 처벌한다는 부작위범 성립 요건이 규정되어 있습니다."
            ),
        },
    },
    {
        "inputs": {"question": "스토킹범죄로 처벌받으면 형량이 어떻게 돼?"},
        "outputs": {
            "expected_doc_type": "law",
            "expected_keywords": ["스토킹"],
            "reference_answer": (
                "스토킹범죄의 처벌 등에 관한 법률에 따라 스토킹범죄를 저지른 사람은 "
                "징역 또는 벌금에 처해지며, 흉기 등을 이용한 경우 가중처벌됩니다."
            ),
        },
    },
    {
        "inputs": {"question": "성폭력 범죄에서 카메라 등을 이용한 촬영은 어떻게 처벌돼?"},
        "outputs": {
            "expected_doc_type": "law",
            "expected_keywords": ["촬영", "성폭력"],
            "reference_answer": (
                "성폭력범죄의 처벌 등에 관한 특례법에 따라 카메라나 이와 유사한 기능을 갖춘 "
                "기계장치를 이용하여 성적 욕망 또는 수치심을 유발할 수 있는 신체를 "
                "촬영한 경우 처벌 대상이 됩니다."
            ),
        },
    },
    {
        "inputs": {"question": "2026년에 선고된 모욕죄 관련 판례가 있어?"},
        "outputs": {
            "expected_doc_type": "case",
            "expected_keywords": ["모욕"],
            "reference_answer": (
                "2026년에 선고된 형사 판례 중 형법상 모욕죄와 관련된 대법원 판례가 있습니다."
            ),
        },
    },
    {
        "inputs": {"question": "화성에 인간이 살고 있다는 판례가 있어?"},
        "outputs": {
            "expected_doc_type": None,
            "expected_keywords": ["확인할 수 없습니다", "찾을 수 없습니다", "없습니다"],
            "reference_answer": "주어진 자료에서는 확인할 수 없다는 취지로 답해야 합니다.",
        },
    },
]


def get_or_create_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_name=DATASET_NAME)
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        dataset_id=dataset.id,
    )
    return dataset


# ---------- 평가 대상 ----------

graph = build_rag_graph()


def run_agent(inputs: dict) -> dict:
    result = graph.invoke(
        {"messages": [HumanMessage(content=inputs["question"])]},
        config={"configurable": {"thread_id": f"eval-{hash(inputs['question'])}"}},
    )
    final_message = result["messages"][-1]
    used_tools = [
        tc["name"]
        for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
        for tc in m.tool_calls
    ]
    return {"answer": final_message.content, "used_tools": used_tools}


# ---------- 평가자 ----------

class CorrectnessGrade(BaseModel):
    score: bool = Field(description="답변이 참고 정답과 의미상 일치하고 근거가 있으면 True, 아니면 False")
    reasoning: str = Field(description="판단 이유 한 줄 설명")


judge_llm = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
).with_structured_output(CorrectnessGrade)


def correctness_evaluator(run, example) -> dict:
    question = example.inputs["question"]
    answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("reference_answer", "")
    grade = judge_llm.invoke(
        f"질문: {question}\n"
        f"참고 정답: {reference_answer}\n"
        f"실제 답변: {answer}\n\n"
        f"실제 답변이 참고 정답과 의미상 일치하는지 평가해줘. "
        f"문장 표현이 달라도 핵심 내용과 근거가 맞으면 True로 판단해."
    )
    return {"key": "correctness", "score": grade.score, "comment": grade.reasoning}


def tool_selection_evaluator(run, example) -> dict:
    expected_doc_type = example.outputs.get("expected_doc_type")
    used_tools = run.outputs.get("used_tools", [])
    if expected_doc_type is None:
        return {"key": "tool_selection", "score": True, "comment": "자료 없음 케이스는 통과"}
    expected_tool = "search_law" if expected_doc_type == "law" else "search_case"
    score = expected_tool in used_tools
    return {
        "key": "tool_selection",
        "score": score,
        "comment": f"기대 도구: {expected_tool}, 실제 호출: {used_tools}",
    }


def keyword_coverage_evaluator(run, example) -> dict:
    answer = run.outputs.get("answer", "")
    expected_keywords = example.outputs.get("expected_keywords", [])
    if not expected_keywords:
        return {"key": "keyword_coverage", "score": True, "comment": "체크할 키워드 없음"}
    hit = sum(1 for kw in expected_keywords if kw in answer)
    score = hit / len(expected_keywords)
    return {
        "key": "keyword_coverage",
        "score": score,
        "comment": f"{hit}/{len(expected_keywords)} 키워드 매칭",
    }


# ---------- 실행 ----------

if __name__ == "__main__":
    dataset = get_or_create_dataset()
    evaluate(
        run_agent,
        data=dataset.name,
        evaluators=[correctness_evaluator, tool_selection_evaluator, keyword_coverage_evaluator],
        experiment_prefix="online_law_criminal",
        max_concurrency=4,
    )
    print("평가 완료. LangSmith 대시보드에서 결과를 확인하세요.")