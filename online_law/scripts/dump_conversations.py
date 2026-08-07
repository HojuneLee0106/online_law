"""checkpoints.db 에서 실제 질문·답변을 꺼내 본다.

배포 환경은 LangSmith 로 추적을 보내지 않으므로(deploy.yml 이 LANGSMITH_*
환경변수를 넘기지 않는다) 사용자 대화가 남는 곳은 이 파일뿐이다.

임베딩 모델을 올리지 않으므로 컨테이너 안에서 돌려도 메모리 부담이 없다.

    uv run python scripts/dump_conversations.py                 # 최근 10개 대화
    uv run python scripts/dump_conversations.py --limit 30
    uv run python scripts/dump_conversations.py --thread <id>   # 특정 대화 전문
    uv run python scripts/dump_conversations.py --grep invoke   # 내용 검색

EC2 컨테이너에서:
    docker exec online-law uv run python scripts/dump_conversations.py
    (이미지에 scripts/ 가 없으면 docker cp 로 넣거나 호스트에서 직접 실행)
"""
import argparse
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

DB = Path(__file__).resolve().parent.parent / "var" / "checkpoints.db"
serde = JsonPlusSerializer()


def latest_state(conn, thread_id):
    """그 대화의 가장 최근 체크포인트에서 메시지 목록을 복원."""
    row = conn.execute(
        "SELECT type, checkpoint FROM checkpoints WHERE thread_id = ? "
        "ORDER BY checkpoint_id DESC LIMIT 1",
        (thread_id,),
    ).fetchone()
    if not row:
        return []
    ckpt = serde.loads_typed((row[0], row[1]))
    return ckpt.get("channel_values", {}).get("messages", []) or []


def render(messages):
    """(역할, 본문) 목록으로. 도구 호출 턴과 검색 결과는 건너뛴다."""
    out = []
    for m in messages:
        kind = m.__class__.__name__
        if kind == "ToolMessage" or getattr(m, "tool_calls", None):
            continue
        content = m.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not str(content).strip():
            continue
        role = "질문" if kind == "HumanMessage" else "답변"
        out.append((role, str(content)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--thread")
    ap.add_argument("--grep", help="이 문자열이 포함된 메시지만")
    ap.add_argument("--full", action="store_true", help="본문 전체 출력")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    if args.thread:
        threads = [args.thread]
    else:
        threads = [r[0] for r in conn.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(rowid) DESC LIMIT ?", (args.limit,))]

    for tid in threads:
        pairs = render(latest_state(conn, tid))
        if args.grep:
            pairs = [p for p in pairs if args.grep in p[1]]
            if not pairs:
                continue
        print("=" * 76)
        print(f"thread {tid}  ({len(pairs)}개 메시지)")
        for role, text in pairs:
            body = text if (args.full or args.grep) else text[:200]
            print(f"  [{role}] {body}" + ("" if len(text) <= len(body) else " …"))


if __name__ == "__main__":
    main()
