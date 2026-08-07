"""배포된 서버에 동시 질문을 보내 응답 지연이 어떻게 늘어나는지 측정한다.

검색은 sync 도구라 스레드풀에서 직렬화되고(로컬 실측: 동시 8이면 건당 8배),
LLM 왕복은 비동기라 겹쳐서 진행된다. 둘 중 무엇이 지배적인지는 인스턴스
메모리에 달려 있어서 실제 배포 환경에서 재봐야 안다.

    export LAW_URL=http://<host>:8000
    export LAW_USER=<id> LAW_PASS=<pw>
    uv run python scripts/bench_concurrency.py          # 동시 1,2,3 으로 각 1회전
    uv run python scripts/bench_concurrency.py 1 2      # 동시 수 직접 지정

주의: 질문 1건마다 Anthropic API 를 호출한다. 동시 3까지 돌리면 총 6건이다.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.environ.get("LAW_URL", "").rstrip("/")
USER = os.environ.get("LAW_USER", "")
PASS = os.environ.get("LAW_PASS", "")

# 동시 수를 바꿔가며 비교하려면 매 회전의 작업량이 같아야 한다.
# 질문을 섞으면 무거운 질문이 섞인 회전만 느려져서 동시성 효과와 구분되지 않는다.
# 그래서 모든 요청이 같은 질문을 쓴다. 판례 검색(EC2 최대 병목)을 타는 것으로 고른다.
QUESTION = "밤에 모르는 사람이 저를 밀치길래 저도 밀쳐서 넘어뜨렸는데 정당방위인가요?"

# 첫 요청은 Chroma 인덱스가 페이지 캐시에 없어 디스크를 읽는다(콜드 스타트).
# 측정 전에 한 번 버리는 요청을 보내 조건을 맞춘다.
WARMUP = True


def login() -> str:
    r = requests.post(f"{BASE}/api/login", json={"username": USER, "password": PASS}, timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def ask(token: str, question: str) -> tuple[float, float, str]:
    """(첫 토큰까지, 전체 완료까지, 상태) 를 돌려준다."""
    t0 = time.perf_counter()
    first = None
    err = ""
    with requests.post(
        f"{BASE}/api/query/stream",
        json={"question": question},
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            if '"token"' in payload and first is None:
                first = time.perf_counter() - t0
            if '"error"' in payload:
                err = payload[:120]
    total = time.perf_counter() - t0
    return (first or total), total, err


def main():
    if not (BASE and USER and PASS):
        sys.exit("LAW_URL, LAW_USER, LAW_PASS 환경변수를 설정하세요.")
    levels = [int(x) for x in sys.argv[1:]] or [1, 2, 3]
    token = login()
    print(f"대상: {BASE}")
    if WARMUP:
        t = time.perf_counter()
        ask(token, QUESTION)
        print(f"워밍업 1건 완료 ({time.perf_counter() - t:.1f}s, 측정에서 제외)")
    print()
    print(f'{"동시":>4}  {"첫토큰 평균":>12}  {"완료 평균":>10}  {"완료 최대":>10}  {"벽시계":>8}')
    print("-" * 56)
    for n in levels:
        qs = [QUESTION] * n
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as exe:
            res = list(exe.map(lambda q: ask(token, q), qs))
        wall = time.perf_counter() - t0
        firsts = [r[0] for r in res]
        totals = [r[1] for r in res]
        errs = [r[2] for r in res if r[2]]
        print(f"{n:>4}  {sum(firsts)/len(firsts):>11.1f}s  {sum(totals)/len(totals):>9.1f}s  "
              f"{max(totals):>9.1f}s  {wall:>7.1f}s")
        for e in errs:
            print(f"      [오류] {e}")
        time.sleep(3)   # 다음 회전 전에 서버를 잠깐 쉬게 한다


if __name__ == "__main__":
    main()
