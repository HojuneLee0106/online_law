"""
책자형 생활법령 PDF -> 섹션 단위 JSON 파싱 (로컬 실행용)
 
사용법:
    pip install pdfplumber
    python parse_pdfs.py
 
- pdfs/{csmSeq}.pdf 들을 읽어 섹션 단위로 쪼갠다
- csm_catalog.json 으로 카테고리(도메인) 메타데이터를 붙인다
- 결과를 qa_documents.json 으로 저장 (counsel_agent 적재용)
- 파싱 실패/이상 파일은 parse_report.txt 에 기록
 
출력 JSON 각 항목 스키마:
{
  "doc_type": "qa",
  "category": "가족관계 등록",        # 생활법령 대분류
  "category_seq": 1,
  "csm_seq": 707,
  "content_title": "가족관계 등록",   # 콘텐츠(PDF) 제목
  "section": "1.1.1 가족관계등록제도 이해하기",  # 섹션 경로
  "content": "...(섹션 본문)...",
  "source": "생활법령 - 가족관계 등록 > 1.1.1 가족관계등록제도 이해하기"
}
"""
 
import json
import os
import re
 
import pdfplumber
 
PDF_DIR = "pdfs"
CATALOG_PATH = "csm_catalog.json"
OUT_PATH = "qa_documents.json"
REPORT_PATH = "parse_report.txt"
 
DISCLAIMER_START = "이 정보는"
DISCLAIMER_END = "국민 신문고에 문의하시기 바랍니다."
# 저작권/이용 안내 블록(가족관계등록 PDF에서 본 추가 안내문)도 잘라낸다
COPYRIGHT_MARKERS = [
    "공공데이터 정책에 따라",
    "위조ㆍ변조하거나",
    "저작권을 갖는 저작물의 경우",
]
 
# 섹션 헤더: 1 / 1.1 / 1.1.1 형태. 단, 4자리 이상 숫자(연도 등)는 제외
SEC_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(\S.*)$")
 
 
def clean_lines(text: str) -> list[str]:
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s == "찾기쉬운 생활법령":
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", s):   # 페이지번호 N / N
            continue
        out.append(ln)
    return out
 
 
def is_toc_line(ln: str) -> bool:
    # 목차 라인: 점선(. . .)이 있고 끝이 숫자
    return bool(re.search(r"\.\s*\.\s*\.", ln)) and bool(re.search(r"\d+\s*$", ln))
 
 
def strip_boilerplate(text: str) -> str:
    # 면책 안내문 제거
    di = text.find(DISCLAIMER_START)
    de = text.find(DISCLAIMER_END)
    if di != -1 and de != -1 and de > di:
        text = (text[:di] + text[de + len(DISCLAIMER_END):]).strip()
    # 저작권/이용안내 블록 제거 (마커가 나오는 줄부터 몇 줄 스킵은 과하므로
    # 여기서는 해당 문장이 포함된 라인만 후처리에서 걸러낸다)
    return text
 
 
def is_boilerplate_line(ln: str) -> bool:
    return any(m in ln for m in COPYRIGHT_MARKERS) or ln.strip().startswith("※ 다만")
 
 
def valid_section_num(num: str) -> bool:
    # 1~2자리 숫자로 시작하는 섹션만 유효 (연도 2005 등 배제)
    first = num.split(".")[0]
    return len(first) <= 2 and int(first) <= 30
 
 
def parse_pdf(path: str) -> tuple[str, list[dict]]:
    """PDF -> (콘텐츠 제목, 섹션 리스트[{section, content}])."""
    with pdfplumber.open(path) as pdf:
        cover = " ".join((pdf.pages[0].extract_text() or "").split())
        body_pages = [p.extract_text() or "" for p in pdf.pages[1:]]
    full = strip_boilerplate("\n".join(body_pages))
 
    lines = clean_lines(full)
 
    # 목차 구간 제거: 마지막 목차 라인 이후를 본문으로
    toc_end = -1
    for i, ln in enumerate(lines):
        if is_toc_line(ln):
            toc_end = i
    body_lines = [
        ln for i, ln in enumerate(lines)
        if i > toc_end and not is_toc_line(ln) and not is_boilerplate_line(ln)
    ]
 
    # 섹션 분할
    sections = []
    cur_title = "서두"
    cur_buf = []
    for ln in body_lines:
        m = SEC_RE.match(ln.strip())
        if m and valid_section_num(m.group(1)):
            # 이전 섹션 저장
            if cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
                cur_buf = []
            cur_title = f"{m.group(1)} {m.group(2).strip()}"
        else:
            cur_buf.append(ln)
    if cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
 
    # 너무 짧은 섹션(30자 미만)은 노이즈로 제거
    sections = [(t, c) for t, c in sections if len(c) >= 30]
    return cover, sections
 
 
def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {rec["csmSeq"]: rec for rec in json.load(f)}
 
    docs = []
    report = []
    pdf_files = sorted(
        f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")
    )
    for fn in pdf_files:
        csmseq = int(fn[:-4])
        meta = catalog.get(csmseq)
        if not meta:
            report.append(f"[경고] 카탈로그에 없는 파일: {fn}")
            continue
        path = os.path.join(PDF_DIR, fn)
        try:
            cover, sections = parse_pdf(path)
        except Exception as e:
            report.append(f"[실패] {fn} ({meta['title']}) -> {type(e).__name__}: {e}")
            continue
 
        if not sections:
            report.append(f"[빈결과] {fn} ({meta['title']}) -> 섹션 0개")
            continue
 
        for sec_title, content in sections:
            docs.append({
                "doc_type": "qa",
                "category": meta["category"],
                "category_seq": meta["categorySeq"],
                "csm_seq": csmseq,
                "content_title": meta["title"],
                "section": sec_title,
                "content": content,
                "source": f"생활법령 - {meta['title']} > {sec_title}",
            })
        report.append(f"[OK] {fn} ({meta['title']}) -> 섹션 {len(sections)}개")
 
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
 
    print(f"파싱 완료: PDF {len(pdf_files)}개 -> 청크(섹션) {len(docs)}개")
    print(f"  결과: {OUT_PATH}")
    print(f"  리포트: {REPORT_PATH}")
 
 
if __name__ == "__main__":
    main()