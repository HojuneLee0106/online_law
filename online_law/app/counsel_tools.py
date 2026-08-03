"""
counsel_agent용 search_qa 도구 실제 구현.
 
graph.py에서 기존의 스텁 build_counsel_tools()를 이걸로 교체하세요.
(build_qa_vectordb.py로 chroma_qa 를 먼저 구축해야 합니다.)
 
생활법령 18개 대분류를 domain으로 필터링할 수 있게 했습니다.
"""
 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
 
QA_PERSIST_DIR = "./var/chroma_qa"
QA_COLLECTION = "life_law"
 
# 생활법령 대분류(카테고리) — search_qa의 category 필터 값
LIFE_LAW_CATEGORIES = {
    "가정법률", "아동·청소년/교육", "부동산/임대차", "금융/금전", "사업", "창업",
    "무역/출입국", "소비자", "문화/여가생활", "민형사/소송", "교통/운전", "근로/노동",
    "복지", "국방/보훈", "정보통신/기술", "환경/에너지", "사회안전/범죄", "국가 및 지자체",
    "all",
}
 
 
def load_qa_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return Chroma(
        collection_name=QA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=QA_PERSIST_DIR,
    )
 
 
def _format_qa(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "출처 미상")
        parts.append(f"[출처: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
 
 
def build_counsel_tools():
    vectorstore = load_qa_vectorstore()
 
    @tool
    def search_qa(query: str, category: str = "all") -> str:
        """생활법령 상담 자료에서 관련 내용을 검색합니다.
        일상적인 법률 문제의 절차·대처 방법 안내가 필요할 때 사용하세요.
 
        Args:
            query: 검색할 내용 (예: '재혼 후 자녀 성 변경', '전세 계약 갱신')
            category: 검색 범위(생활법령 대분류). 예: 가정법률 / 부동산/임대차 /
                      근로/노동 / 복지 / 소비자 / 민형사/소송 / all(전체).
                      분야가 애매하면 all 을 사용하세요.
        """
        if category not in LIFE_LAW_CATEGORIES:
            cats = ", ".join(sorted(c for c in LIFE_LAW_CATEGORIES if c != "all"))
            return f"'{category}'는 유효하지 않은 분야입니다. 사용 가능: {cats}, all"
 
        flt = None if category == "all" else {"category": {"$eq": category}}
        docs = vectorstore.similarity_search(query, k=3, filter=flt)
        if not docs:
            return f"'{category}' 분야에서 관련 생활법령 자료를 찾을 수 없습니다."
        return _format_qa(docs)
 
    return [search_qa]