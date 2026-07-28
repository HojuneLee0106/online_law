"""
qa_documents.json -> Chroma 벡터DB 적재 (counsel_agent 전용)
 
사용법:
    python build_qa_vectordb.py
 
- 생활법령 섹션 청크를 임베딩해 별도 컬렉션(collection_name="life_law")에 저장
- research_agent의 law/case 벡터DB와 분리 → 상담 검색이 섞이지 않음
- 섹션이 길면 추가 청킹(1000자) 후 적재
- csm_seq 기반 안정적 ID로 중복 삽입 방지(재실행 시 upsert처럼 동작)
 
기존 vectordb.py와 동일한 임베딩 모델(jhgan/ko-sroberta-multitask) 사용.
"""
 
import json
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
 
QA_JSON = "qa_documents.json"
PERSIST_DIR = "./chroma_qa"          # counsel 전용 (research와 폴더 분리)
COLLECTION = "life_law"
 
 
def main():
    with open(QA_JSON, encoding="utf-8") as f:
        records = json.load(f)
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
 
    docs, ids = [], []
    for rec in records:
        base = Document(
            page_content=rec["content"],
            metadata={
                "doc_type": "qa",
                "category": rec["category"],
                "category_seq": rec["category_seq"],
                "csm_seq": rec["csm_seq"],
                "content_title": rec["content_title"],
                "section": rec["section"],
                "source": rec["source"],
            },
        )
        # 섹션이 길면 재청킹
        for i, chunk in enumerate(splitter.split_documents([base])):
            docs.append(chunk)
            ids.append(f"qa-{rec['csm_seq']}-{len(docs)}")
 
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    BATCH = 5000
    for i in range(0, len(docs), BATCH):
        vectorstore.add_documents(docs[i:i+BATCH], ids=ids[i:i+BATCH])
        print(f"  적재 중... {min(i+BATCH, len(docs))}/{len(docs)}")

    print(f"적재 완료: 원본 섹션 {len(records)}개 -> 청크 {len(docs)}개")
 
 
if __name__ == "__main__":
    main()