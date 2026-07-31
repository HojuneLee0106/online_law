"""세 variant(current/shared/single)를 16문항으로 평가 실행."""
import asyncio
from dotenv import load_dotenv
from langsmith import Client

from eval_targets import make_target
from eval_evaluators import routing_accuracy, tool_usage, answer_quality

load_dotenv()

DATASET_NAME = "online_law_eval"
VARIANTS = ["current", "shared", "single"]

client = Client()


async def run_variant(variant: str):
    print(f"\n{'='*50}\n[{variant}] 평가 시작\n{'='*50}")
    target = make_target(variant)

    results = await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[routing_accuracy, tool_usage, answer_quality],
        experiment_prefix=f"online_law_{variant}",
        max_concurrency=2,   # 동시 실행 (너무 높이면 rate limit)
    )
    print(f"[{variant}] 평가 완료")
    return results


async def main():
    for variant in VARIANTS:
        await run_variant(variant)
    print("\n전체 평가 완료. LangSmith 대시보드에서 비교하세요.")


if __name__ == "__main__":
    asyncio.run(main())