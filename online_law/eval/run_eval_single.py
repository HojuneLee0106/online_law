"""single agent를 10문항으로 평가 실행.

평가 항목:
  - tool_usage     : 기대 도구를 실제로 호출했는지 (recall)
  - tool_precision : 불필요한 도구까지 부르지 않았는지 (precision)
  - answer_quality : LLM-judge 기반 답변 품질(정확성·충실성·실용성·명료성)

사용법:
    python eval/upload_dataset_single.py   # 최초 1회 (데이터셋 업로드)
    python eval/run_eval_single.py
"""
import asyncio

from dotenv import load_dotenv
from langsmith import Client

from eval_targets import make_target
from eval_evaluators import tool_usage, tool_precision, answer_quality

load_dotenv()

DATASET_NAME = "online_law_single_eval"
VARIANT = "single"

client = Client()


async def main():
    print(f"\n{'=' * 50}\n[{VARIANT}] 평가 시작 (dataset={DATASET_NAME})\n{'=' * 50}")
    target = make_target(VARIANT)

    await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[tool_usage, tool_precision, answer_quality],
        experiment_prefix=f"online_law_{VARIANT}",
        max_concurrency=2,   # 너무 높이면 rate limit
    )
    print(f"\n[{VARIANT}] 평가 완료. LangSmith 대시보드에서 결과를 확인하세요.")


if __name__ == "__main__":
    asyncio.run(main())
