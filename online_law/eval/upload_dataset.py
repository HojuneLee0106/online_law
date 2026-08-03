"""eval_dataset.json의 단일턴 16문항을 LangSmith 데이터셋으로 업로드."""
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "online_law_eval"
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

# 멀티턴 제외 (single-turn만)
examples = [ex for ex in data["examples"] if ex["category"] != "multiturn"]
print(f"업로드 대상: {len(examples)}문항")

client = Client()

# 기존 데이터셋 있으면 재사용, 없으면 생성
if client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print("기존 데이터셋 재사용")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=data["description"],
    )
    print("새 데이터셋 생성")

# example 추가: inputs=질문, outputs=기대 메타데이터(채점 기준)
client.create_examples(
    dataset_id=dataset.id,
    inputs=[{"question": ex["question"]} for ex in examples],
    outputs=[
        {
            "expected_route": ex["expected_route"],
            "expected_tools": ex["expected_tools"],
            "rubric_notes": ex["rubric_notes"],
            "category": ex["category"],
            "id": ex["id"],
        }
        for ex in examples
    ],
)
print(f"업로드 완료: {DATASET_NAME}")