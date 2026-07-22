import json
import os
import time
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from law_api import (
    CRIMINAL_LAWS,
    PRECEDENT_START_DATE,
    PRECEDENT_END_DATE,
    find_current_law,
    get_law_full_text,
    search_precedents_by_law,
    get_precedent_full_text,
)
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_persist")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "./chroma_persist_manifest.json")

def _clean_html(text: str)->str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def _content_to_text(content)-> str:
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

def precedent_to_documents(law_name: str, start_date: str, end_date: str) -> list[Document]:
    items = search_precedents_by_law(law_name, start_date, end_date)
    docs = []
    for item in items:
        prec_id = item.get("판례일련번호")
        case_name = item.get("사건명", "")
        case_no = item.get("사건번호", "")
        court = item.get("법원명", "")
        judged_date = item.get("선고일자", "")

        detail = get_precedent_full_text(prec_id)
        service_body = detail.get("PrecService", {})

        full_text = _clean_html(service_body.get("판례내용", ""))
        judgment_summary = _clean_html(service_body.get("판결요지", ""))
        holding = _clean_html(service_body.get("판시사항", ""))

        if not full_text:
            print(f"    [case] {case_name}({case_no}) 본문을 가져오지 못했습니다.")
            time.sleep(0.2)
            continue

        combined_text = f"[사건명: {case_name}]\n[판시사항]\n{holding}\n\n[판결요지]\n{judgment_summary}\n\n[전문]\n{full_text}"

        docs.append(Document(
            page_content=combined_text,
            metadata={
                "doc_type": "case",
                "law_field": law_name,
                "case_name": case_name,
                "case_no": case_no,
                "court": court,
                "judged_date": judged_date,
                "prec_id": prec_id,
                "source": f"{case_name} ({case_no}, {court} {judged_date})",
            },
        ))
        time.sleep(0.2)
    return docs

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def build_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

def law_to_documents(law_name: str)-> tuple[list[Document], str | None, str | None]:
    """법령명으로 현행 법령을 찾아 조문 단위 Document 리스트로 변환.
    반환: (문서 리스트, MST, 시행일자)"""
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
        
        jo_no=art.get("조문번호","")
        jo_title=art.get("조문제목","")
        docs.append(Document(
            page_content=content,
            metadata={
                "doc_type": "law",
                "law_field": law_name,
                "law_name": law_name,
                "article_no": jo_no,
                "article_title": jo_title,
                "mst": mst,
                "efYd": efYd,
                "source": f"{law_name} 제{jo_no}조" + (f"({jo_title})" if jo_title else ""),
            },
        ))
    return docs, mst, efYd


def build_vectorstore(persist_directory: str = CHROMA_PERSIST_DIR):
    embeddings = build_embeddings()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    manifest = load_manifest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for law_name in CRIMINAL_LAWS:
        print(f"=== {law_name} 처리 중 ===")

        # --- 법령 처리 ---
        law_docs, mst, efYd = law_to_documents(law_name)
        if law_docs:
            manifest_key = f"law:{law_name}"
            version_key = f"{mst}:{efYd}"
            if manifest.get(manifest_key) != version_key:
                vectorstore.delete(where={"law_name": law_name})
                split_docs = splitter.split_documents(law_docs)
                vectorstore.add_documents(split_docs)
                manifest[manifest_key] = version_key
                print(f"  법령 청크 {len(split_docs)}개 저장 완료 (MST={mst}, 시행일={efYd})")
            else:
                print("  법령 변경 없음, 스킵")

        # --- 판례 처리 ---
        case_docs = precedent_to_documents(law_name, PRECEDENT_START_DATE, PRECEDENT_END_DATE)
        new_case_docs = []
        for doc in case_docs:
            prec_id = doc.metadata["prec_id"]
            manifest_key = f"case:{prec_id}"
            if manifest.get(manifest_key) != "done":
                new_case_docs.append(doc)
                manifest[manifest_key] = "done"

        if new_case_docs:
            split_docs = splitter.split_documents(new_case_docs)
            vectorstore.add_documents(split_docs)
            print(f"  판례 청크 {len(split_docs)}개 저장 완료 (신규 판례 {len(new_case_docs)}건)")
        else:
            print("  신규 판례 없음")

    save_manifest(manifest)
    print("Chroma DB 구축 완료")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()