"""평가자: 라우팅 정확도, 도구 사용 적절성, 인용 근거성, 품질(LLM-judge)."""
import os
import re
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

# ── (b-2) 과잉 검색 여부 (precision) ──
def tool_precision(outputs: dict, reference_outputs: dict) -> dict:
    """실제 호출한 도구 중 기대 도구의 비율. 불필요한 도구까지 부르면 낮아진다."""
    expected = set(reference_outputs.get("expected_tools", []))
    actual = set(outputs.get("tools_used", []))
    if not expected or not actual:
        # 아예 도구를 안 쓴 경우는 tool_usage(recall)가 0으로 잡으므로 여기선 스킵
        return {"key": "tool_precision", "score": None}
    hit = len(expected & actual) / len(actual)
    return {"key": "tool_precision", "score": round(hit, 2)}

# ── (b-3) 인용 근거성 ──
# LLM-judge 는 조문 번호나 형량 수치가 검색 결과와 일치하는지를 안정적으로
# 잡아내지 못한다(실측: rubric 요구 항목이 0/3 -> 3/3 이 되었는데 judge 점수는
# 오히려 내려갔다). 그래서 규칙으로 직접 센다.
#
# 법령명이 앞에 붙은 조문만 센다. 앞의 (?<![가-힣]) 는 "이를 위반하면 동물보호법"
# 처럼 앞 단어가 법령명에 딸려 들어가는 것을 막는다.
LAW_ARTICLE_RE = re.compile(
    r'(?<![가-힣])([가-힣ㆍ·]{1,20}법(?:률)?)[」]?\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?'
)
CASE_NO_RE = re.compile(r'\b(\d{4}[가-힣]{1,3}\d+)\b')
# 조문 토큰 주변에서 법령명을 찾을 범위. 검색 결과는 "[출처: 도로교통법 제148조의2
# (벌칙)]" 처럼 법령명과 조문이 붙어 나오지만, 생활법령은 "…에 처해집니다
# (「동물보호법」 제97조제1항제1호)" 처럼 떨어져 있어 여유를 둔다.
_CITE_WINDOW = 150


def _extract_citations(text: str) -> list[tuple[str, str, str]]:
    """답변에서 (종류, 법령명, 토큰) 인용 목록을 중복 없이 추출."""
    found = []
    for m in LAW_ARTICLE_RE.finditer(text):
        law = re.sub(r'\s+', '', m.group(1))
        article = f"제{m.group(2)}조" + (f"의{m.group(3)}" if m.group(3) else "")
        found.append(("law", law, article))
    for m in CASE_NO_RE.finditer(text):
        found.append(("case", "", m.group(1)))
    return list(dict.fromkeys(found))


def _is_grounded(kind: str, law: str, token: str, retrieved: str) -> bool:
    """그 인용이 검색 결과에 실제로 있었는지."""
    if kind == "case":
        return token in retrieved
    # 공백을 지워서 "제 148 조의 2" 같은 표기 차이를 흡수한다
    flat = re.sub(r'\s+', '', retrieved)
    return any(
        law[:4] in flat[max(0, m.start() - _CITE_WINDOW): m.end() + 80]
        for m in re.finditer(re.escape(token), flat)
    )


def citation_grounding(outputs: dict) -> dict:
    """답변이 인용한 조문·사건번호 중 검색 결과에 실제로 있던 비율.

    1.0 이면 인용이 전부 검색으로 확인된 것이고, 낮을수록 모델이 기억에
    의존해 조문 번호를 지어낸 것이다. 인용이 하나도 없으면 채점하지 않는다
    (인용을 아예 안 한 답변이 1.0 을 받아서는 안 된다).
    """
    answer = outputs.get("answer", "")
    retrieved = outputs.get("retrieved", "")
    if not retrieved:
        return {"key": "citation_grounding", "score": None,
                "comment": "검색 결과 없음"}
    citations = _extract_citations(answer)
    if not citations:
        return {"key": "citation_grounding", "score": None, "comment": "인용 없음"}
    ungrounded = [f"{law} {tok}".strip()
                  for kind, law, tok in citations
                  if not _is_grounded(kind, law, tok, retrieved)]
    score = 1 - len(ungrounded) / len(citations)
    return {
        "key": "citation_grounding",
        "score": round(score, 2),
        "comment": (f"{len(citations)}건 중 미근거 {len(ungrounded)}건: "
                    f"{', '.join(ungrounded)}" if ungrounded
                    else f"{len(citations)}건 전부 검색 결과에 존재"),
    }


def citation_recall(outputs: dict, reference_outputs: dict) -> dict:
    """정답 조문·법정형(expected_citations)을 답변이 실제로 담았는지.

    judge 가 놓치는 '조문 번호와 법정형이 있는가'를 문자열로 직접 확인한다.
    데이터셋에 expected_citations 가 없는 문항은 채점하지 않는다.
    """
    expected = reference_outputs.get("expected_citations") or []
    if not expected:
        return {"key": "citation_recall", "score": None}
    answer = re.sub(r'\s+', '', outputs.get("answer", ""))
    missing = [e for e in expected if re.sub(r'\s+', '', e) not in answer]
    score = 1 - len(missing) / len(expected)
    return {
        "key": "citation_recall",
        "score": round(score, 2),
        "comment": f"누락: {', '.join(missing)}" if missing else "전부 포함",
    }


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