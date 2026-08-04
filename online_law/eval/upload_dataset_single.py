"""eval_dataset_single.json(10문항)을 LangSmith 데이터셋으로 업로드 (single agent 평가용)."""
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "online_law_single_eval"
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset_single.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

examples = data["examples"]
print(f"업로드 대상: {len(examples)}문항")

client = Client()

if client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print("기존 데이터셋 재사용")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=data["description"],
    )
    print("새 데이터셋 생성")

# 이미 올라간 example 은 건너뛴다 (재실행 시 중복 방지)
existing = {
    ex.outputs.get("id")
    for ex in client.list_examples(dataset_id=dataset.id)
    if ex.outputs
}
new_examples = [ex for ex in examples if ex["id"] not in existing]

if not new_examples:
    print("추가할 신규 문항 없음")
else:
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": ex["question"]} for ex in new_examples],
        outputs=[
            {
                "expected_tools": ex["expected_tools"],
                "rubric_notes": ex["rubric_notes"],
                "category": ex["category"],
                "domain": ex["domain"],
                "id": ex["id"],
            }
            for ex in new_examples
        ],
    )
    print(f"업로드 완료: {DATASET_NAME} (신규 {len(new_examples)}문항)")
