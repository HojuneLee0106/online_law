"""
찾기쉬운 생활법령 책자형 PDF 281개 다운로드 스크립트 (로컬 실행용)
 
사용법:
    python download.py
 
- csm_catalog.json 을 읽어 각 콘텐츠의 전체 PDF를 pdfs/ 폴더에 저장
- 파일명은 {csmSeq}.pdf 로 저장 (파싱 시 카탈로그와 매핑하기 위함)
- 이미 받은 파일은 건너뜀 (재실행 시 실패분만 재시도)
- 서버 부하를 위해 요청 간 딜레이
 
주의: 이 스크립트는 본인 PC(easylaw 접근이 되는 환경)에서 실행하세요.
"""
 
import json
import os
import time
import urllib.request
import urllib.error
 
CATALOG_PATH = "csm_catalog.json"
OUT_DIR = "pdfs"
BASE_URL = "https://www.easylaw.go.kr/CSP/FileDownload.laf"
DELAY_SEC = 1.0          # 요청 간 대기 (서버 예의)
TIMEOUT = 60
 
# 브라우저처럼 보이도록 헤더 설정 (403 회피)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.easylaw.go.kr/CSP/CsmSortRetrieveLst.laf?sortType=cate",
    "Accept": "application/pdf,*/*",
}
 
 
def download_one(csmseq: int, dest: str) -> tuple[bool, str]:
    url = f"{BASE_URL}?flType=pdf&onhunqnaYn=N&csmSeq={csmseq}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        if data[:4] != b"%PDF":
            return False, f"PDF 아님 (앞부분: {data[:30]!r})"
        with open(dest, "wb") as f:
            f.write(data)
        return True, f"{len(data):,} bytes"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
 
 
def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
 
    os.makedirs(OUT_DIR, exist_ok=True)
 
    total = len(catalog)
    ok, skip, fail = 0, 0, []
    for i, rec in enumerate(catalog, 1):
        csmseq = rec["csmSeq"]
        title = rec["title"]
        dest = os.path.join(OUT_DIR, f"{csmseq}.pdf")
 
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            skip += 1
            print(f"[{i}/{total}] skip  {csmseq} {title}")
            continue
 
        success, msg = download_one(csmseq, dest)
        if success:
            ok += 1
            print(f"[{i}/{total}] OK    {csmseq} {title} ({msg})")
        else:
            fail.append((csmseq, title, msg))
            print(f"[{i}/{total}] FAIL  {csmseq} {title} -> {msg}")
        time.sleep(DELAY_SEC)
 
    print("\n" + "=" * 50)
    print(f"완료: 성공 {ok} / 스킵 {skip} / 실패 {len(fail)} (전체 {total})")
    if fail:
        print("\n실패 목록 (재실행하면 재시도됩니다):")
        for csmseq, title, msg in fail:
            print(f"  {csmseq} {title} -> {msg}")
 
 
if __name__ == "__main__":
    main()