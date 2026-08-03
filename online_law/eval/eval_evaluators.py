"""평가자 3종: 라우팅 정확도, 도구 사용 적절성, 품질(LLM-judge)."""
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv() 
# ── (a) 라우팅 정확도 ──
def routing_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs.get("expected_route")
    actual = outputs.get("route")
    # single variant는 route가 "none" → 이 평가 스킵(1점 처리 안 함)
    if actual == "none":
        return {"key": "routing_accuracy", "score": None}
    return {
        "key": "routing_accuracy",
        "score": 1.0 if actual == expected else 0.0,
    }

# ── (b) 도구 사용 적절성 ──
def tool_usage(outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_tools", []))
    actual = set(outputs.get("tools_used", []))
    if not expected:
        return {"key": "tool_usage", "score": None}
    # 기대 도구 중 실제로 쓴 비율 (recall)
    hit = len(expected & actual) / len(expected)
    return {"key": "tool_usage", "score": round(hit, 2)}

# ── (c) 품질 LLM-judge ──
_judge = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

JUDGE_PROMPT = """당신은 법률 상담 답변을 평가하는 채점자입니다.

[사용자 질문]
{question}

[채점 기준 - 이 답변에 반드시 담겨야 할 핵심]
{rubric}

[평가할 답변]
{answer}

다음 4개 항목을 각 0~25점으로 채점하고, 총점(0~100)을 매기세요.
1. 정확성: 법적으로 맞는 내용인가, 틀린 정보는 없는가
2. 충실성: 채점 기준의 핵심 요소를 얼마나 담았는가
3. 실용성: 사용자가 실제로 뭘 해야 할지 구체적으로 알려주는가 (변호사에게만 미루면 감점)
4. 명료성: 이해하기 쉽고 간결한가 (불필요한 preamble·중복 감점)

반드시 아래 JSON 형식으로만 답하세요:
{{"accuracy": <점수>, "completeness": <점수>, "practicality": <점수>, "clarity": <점수>, "total": <합계>, "reason": "<한 줄 이유>"}}"""

def answer_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    import json as _json
    prompt = JUDGE_PROMPT.format(
        question=inputs["question"],
        rubric=reference_outputs.get("rubric_notes", ""),
        answer=outputs.get("answer", ""),
    )
    resp = _judge.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = _json.loads(text)
        return {
            "key": "answer_quality",
            "score": parsed["total"] / 100.0,  # 0~1 정규화
            "comment": parsed.get("reason", ""),
        }
    except Exception as e:
        return {"key": "answer_quality", "score": None, "comment": f"파싱실패: {e}"}