"""single agent를 10문항으로 평가 실행.

평가 항목:
  - tool_usage         : 기대 도구를 실제로 호출했는지 (recall)
  - tool_precision     : 불필요한 도구까지 부르지 않았는지 (precision)
  - citation_grounding : 답변이 인용한 조문·사건번호가 검색 결과에 실재했는지
  - citation_recall    : 정답 조문·법정형(expected_citations)을 담았는지
  - answer_quality     : LLM-judge 기반 답변 품질(정확성·충실성·실용성·명료성)

뒤의 두 규칙 기반 평가자는 judge 가 조문 번호·형량 오류를 잡아내지 못해서
추가했다. 실측에서 judge 는 rubric 요구 항목이 0/3 -> 3/3 으로 개선된 답변에
오히려 낮은 점수를 줬다.

사용법:
    python eval/upload_dataset_single.py   # 최초 1회 (데이터셋 업로드)
    python eval/run_eval_single.py
"""
import asyncio

from dotenv import load_dotenv
from langsmith import Client

from eval_targets import make_target
from eval_evaluators import (
    tool_usage,
    tool_precision,
    citation_grounding,
    citation_recall,
    answer_quality,
)

load_dotenv()

DATASET_NAME = "online_law_single_eval"
VARIANT = "single"
# 문항당 반복 횟수. 실측 기준 문항 내 편차는 평균 0.048(최대 0.11)로
# 작지만, 1회 실행으로는 그 편차와 실제 개선(0.05 내외)을 구분할 수 없다.
# 3회면 평균이 안정되고 문항별 편차도 함께 확인할 수 있다.
NUM_REPETITIONS = 3

client = Client()


async def main():
    print(f"\n{'=' * 50}\n[{VARIANT}] 평가 시작 (dataset={DATASET_NAME})\n{'=' * 50}")
    target = make_target(VARIANT)

    await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            tool_usage,
            tool_precision,
            citation_grounding,
            citation_recall,
            answer_quality,
        ],
        experiment_prefix=f"online_law_{VARIANT}",
        max_concurrency=2,       # 너무 높이면 rate limit
        num_repetitions=NUM_REPETITIONS,
    )
    print(f"\n[{VARIANT}] 평가 완료. LangSmith 대시보드에서 결과를 확인하세요.")


if __name__ == "__main__":
    asyncio.run(main())
