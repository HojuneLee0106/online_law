import argparse
import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


from app.law_api import (
    LAW_DOMAINS,
    PRECEDENT_START_DATE,
    PRECEDENT_END_DATE,
    find_current_law,
    get_law_full_text,
    search_precedents_by_law,
    get_precedent_full_text,
)
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./var/chroma_persist")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "./var/chroma_persist_manifest.json")

# 법령 Document의 메타데이터 스키마 버전.
# 스키마가 바뀌면 이 값을 올려서 기존 법령을 강제로 재임베딩한다
# (법령 자체가 개정되지 않으면 mst/efYd가 그대로라 스킵되기 때문).
#   v2: 조문가지번호를 반영해 제148조의2 형태로 출처 표기
LAW_SCHEMA_VERSION = "v2"

DOMAIN_CASE_TYPES = {
    "criminal": {"형사"},
    "housing": {"민사"},
    "labor": {"민사", "일반행정"},
    "traffic": {"형사", "민사"},
    "civil": {"민사", "가사"},
}

def _clean_html(text: str)->str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _content_to_text(content) -> str:
    """조문내용은 list[list[str]] 형태로 온다. 모두 평탄화해서 하나의 텍스트로."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        flat=[]
        for item in content:
            if isinstance(item, list):
                flat.extend(str(x) for x in item if x)
            elif item:
                flat.append(str(item))
        return "\n".join(flat)
    return str(content)

def load_manifest()->dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def build_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

def precedent_to_documents(
    law_name: str, domain: str, skip_prec_ids: set | None = None
) -> list[Document]:
    """참조법령명(JO) 기준 대법원 판례를 본문조회하여 Document 리스트로 변환.

    skip_prec_ids: 이미 적재된 판례일련번호. 본문조회 전에 걸러내 불필요한
    API 호출을 막는다 (본문조회는 건당 요청 + 대기가 있어 재실행 비용이 크다).
    """
    items = search_precedents_by_law(law_name, PRECEDENT_START_DATE, PRECEDENT_END_DATE, supreme_only=True)
    allowed_types=DOMAIN_CASE_TYPES.get(domain, set())
    skip_prec_ids=skip_prec_ids or set()
    docs = []
    for item in items:
        if allowed_types and item.get("사건종류명") not in allowed_types:
            continue

        prec_id = item.get("판례일련번호")
        if str(prec_id) in skip_prec_ids:
            continue
        case_name = item.get("사건명", "")
        case_no = item.get("사건번호", "")
        court = item.get("법원명", "")
        judged_date = item.get("선고일자", "")
        case_type=item.get("사건종류명", "")

        detail = get_precedent_full_text(prec_id)
        service_body = detail.get("PrecService", {})

        full_text = _clean_html(service_body.get("판례내용", ""))
        judgment_summary = _clean_html(service_body.get("판결요지", ""))
        holding = _clean_html(service_body.get("판시사항", ""))

        if not full_text:
            print(f"    [case] {case_name}({case_no}) 본문을 가져오지 못했습니다.")
            time.sleep(0.2)
            continue

        combined_text = (
            f"[사건명: {case_name}]\n[판시사항]\n{holding}\n\n"
            f"[판결요지]\n{judgment_summary}\n\n[전문]\n{full_text}"
        )

        docs.append(Document(
            page_content=combined_text,
            metadata={
                "doc_type": "case",
                "domain": domain,
                "law_name": law_name,
                "case_name": case_name,
                "case_no": case_no,
                "case_type": case_type,
                "court": court,
                "judged_date": judged_date,
                "prec_id": prec_id,
                "source": f"{case_name} ({case_no}, {court} {judged_date})",
            },
        ))
        time.sleep(0.2)
    return docs


def law_to_documents(law_name: str, domain: str) -> tuple[list[Document], str | None, str | None]:
    """법령명으로 현행 법령을 찾아 조문 단위 Document 리스트로 변환."""
    law_meta=find_current_law(law_name)
    if not law_meta:
        print(f"  [law] '{law_name}' 현행 법령을 찾을 수 없습니다.")
        return [], None, None
    mst=law_meta.get("법령일련번호")
    efYd=law_meta.get("시행일자","")
    detail=get_law_full_text(mst)

    articles=detail.get("법령", {}).get("조문", {}).get("조문단위",[])
    if isinstance(articles, dict):
        articles=[articles]
    docs=[]
    for art in articles:
        if art.get("조문여부") !="조문":
            continue
        parts=[_content_to_text(art.get("조문내용"))]

        hangs=art.get("항",[])
        if isinstance(hangs, dict):
            hangs=[hangs]
        for hang in hangs:
            hang_text=_content_to_text(hang.get("항내용"))
            if hang_text:
                parts.append(hang_text)
            hos=hang.get("호",[])
            if isinstance(hos,dict):
                hos=[hos]
            for ho in hos:
                ho_text=_content_to_text(ho.get("호내용"))
                if ho_text:
                    parts.append(ho_text)
        content="\n".join(p for p in parts if p).strip()
        if not content:
            continue
        
        # 개정으로 추가된 조문은 '조문번호'와 '조문가지번호'로 나뉘어 온다.
        # (예: 제148조의2 -> 조문번호=148, 조문가지번호=2)
        # 가지번호를 빼먹으면 제148조/제148조의2/제148조의3이 모두
        # '제148조'가 되어 출처가 틀리고 서로 구분되지 않는다.
        jo_no=art.get("조문번호","")
        jo_branch=art.get("조문가지번호") or ""
        jo_label=f"제{jo_no}조의{jo_branch}" if jo_branch else f"제{jo_no}조"
        jo_title=art.get("조문제목","")
        docs.append(Document(
            page_content=content,
            metadata={
                "doc_type": "law",
                "domain": domain,
                "law_name": law_name,
                "article_no": jo_no,
                "article_branch_no": jo_branch,
                "article_label": jo_label,
                "article_title": jo_title,
                "mst": mst,
                "efYd": efYd,
                "source": f"{law_name} {jo_label}" + (f"({jo_title})" if jo_title else ""),
            },
        ))
    return docs, mst, efYd

def build_domain(domain: str, vectorstore, manifest: dict, splitter) -> None:
    """한 도메인의 법령 + 판례를 처리."""
    law_names=LAW_DOMAINS.get(domain)
    if not law_names:
        print(f"알 수 없는 도메인: {domain}")
        return 
    print(f"\n{'#' * 60}")
    print(f"# 도메인: {domain} ({len(law_names)}개 법령)")
    print(f"{'#' * 60}")
    for law_name in law_names:
        print(f"\n=== {law_name} ===")
        law_docs, mst, efYd=law_to_documents(law_name, domain)
        if law_docs:
            manifest_key=f"law:{domain}:{law_name}"
            version_key=f"{LAW_SCHEMA_VERSION}:{mst}:{efYd}"
            if manifest.get(manifest_key)!=version_key:
                vectorstore.delete(where={"$and": [
                    {"law_name": {"$eq": law_name}},
                    {"domain": {"$eq": domain}},
                    {"doc_type":{"$eq":"law"}},
                ]})
                split_docs=splitter.split_documents(law_docs)
                vectorstore.add_documents(split_docs)
                manifest[manifest_key]=version_key
                print(f"  법령 청크 {len(split_docs)}개 저장 (MST={mst}, 시행일={efYd})")
            else:
                print("  법령 변경 없음, 스킵")
        # 이미 적재된 판례는 본문조회 자체를 건너뛴다
        done_prec_ids={
            key.split(":", 2)[2]
            for key, val in manifest.items()
            if key.startswith(f"case:{domain}:") and val == "done"
        }
        new_case_docs=precedent_to_documents(law_name, domain, skip_prec_ids=done_prec_ids)
        for doc in new_case_docs:
            manifest[f"case:{domain}:{doc.metadata['prec_id']}"]="done"
        if new_case_docs:
            split_docs=splitter.split_documents(new_case_docs)
            vectorstore.add_documents(split_docs)
            print(f"  판례 청크 {len(split_docs)}개 저장 (신규 {len(new_case_docs)}건)")
        else:
            print("  신규 판례 없음")
        # 법령 단위로 manifest를 저장한다. 도메인 단위로 저장하면 중간에 실패했을 때
        # 이미 적재된 청크가 manifest에 남지 않아 재실행 시 중복 적재된다.
        save_manifest(manifest)

def build_vectorstore(domains: list[str] | None = None,
                      persist_directory: str = CHROMA_PERSIST_DIR):
    embeddings = build_embeddings()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    manifest = load_manifest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    targets=domains if domains else list(LAW_DOMAINS.keys())

    for domain in targets:
        build_domain(domain, vectorstore, manifest, splitter)
        save_manifest(manifest)
    print("\nChroma DB 구축 완료")
    return vectorstore

def load_vectorstore(persist_directory: str=CHROMA_PERSIST_DIR):
    """이미 구축된 벡터DB를 열기만 함 (API 호출 없음). 서버에서 사용."""
    return Chroma(
        embedding_function=build_embeddings(),
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="법령/판례 벡터DB 구축")
    parser.add_argument(
        "--domain",
        nargs="*",
        choices=list(LAW_DOMAINS.keys()),
        help="처리할 도메인 (생략 시 전체). 예: --domain housing labor",
    )
    args=parser.parse_args()
    build_vectorstore(domains=args.domain)