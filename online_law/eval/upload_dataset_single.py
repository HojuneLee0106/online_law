"""eval_dataset_single.json(12문항)을 LangSmith 데이터셋으로 업로드 (single agent 평가용)."""
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

def to_outputs(ex: dict) -> dict:
    out = {
        "expected_tools": ex["expected_tools"],
        "rubric_notes": ex["rubric_notes"],
        "category": ex["category"],
        "domain": ex["domain"],
        "id": ex["id"],
    }
    # 없는 문항은 citation_recall 평가자가 채점을 건너뛴다
    if ex.get("expected_citations"):
        out["expected_citations"] = ex["expected_citations"]
    return out


# 이미 올라간 example 은 새로 만들지 않고 outputs 만 갱신한다.
# (건너뛰기만 하면 rubric·expected_citations 을 고쳐도 반영되지 않는다)
existing = {
    ex.outputs.get("id"): ex.id
    for ex in client.list_examples(dataset_id=dataset.id)
    if ex.outputs
}
new_examples = [ex for ex in examples if ex["id"] not in existing]
stale_examples = [ex for ex in examples if ex["id"] in existing]

if new_examples:
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": ex["question"]} for ex in new_examples],
        outputs=[to_outputs(ex) for ex in new_examples],
    )
    print(f"신규 {len(new_examples)}문항 추가")

if stale_examples:
    client.update_examples(
        example_ids=[existing[ex["id"]] for ex in stale_examples],
        outputs=[to_outputs(ex) for ex in stale_examples],
    )
    print(f"기존 {len(stale_examples)}문항 갱신")

print(f"완료: {DATASET_NAME} (전체 {len(examples)}문항)")
