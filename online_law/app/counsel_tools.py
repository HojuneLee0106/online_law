"""생활법령(법제처 '찾기쉬운 생활법령') 검색 도구.

chroma_qa 를 scripts/build_qa_vectordb.py 로 먼저 구축해야 합니다.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

QA_PERSIST_DIR = "./var/chroma_qa"
QA_COLLECTION = "life_law"

# 검색 결과 청크 수. 카테고리 필터를 없앤 뒤로는 무관한 분야의 청크가 상위에
# 섞일 수 있어, 정답이 밀려나면 이 값을 올린다. 3 -> 6 은 호출당 입력 토큰이
# 약 2,500개 늘지만 검색 시간 자체는 변하지 않는다(k와 무관, 12ms 내외).
QA_TOP_K = 3

# 생활법령 대분류(category) 필터를 두지 않는 이유:
# 법제처의 분류 체계가 사용자의 어휘와 어긋나서, 모델이 합리적으로 고른
# 카테고리가 정답을 통째로 걸러내는 일이 반복됐다.
#   "고양이를 죽였는데 처벌받나요" -> 사회안전/범죄 로 좁힘
#     정답(동물보호법 제97조)은 복지·문화/여가생활 에 있어 5개 검색어 전부 실패
#   "중고거래 사기 고소"          -> 민형사/소송 으로 좁힘
#     정답(중고거래 피해자)은 소비자 에 있어 거리 65 -> 145 로 악화
# 필터 없이 검색하면 위 두 경우 모두 상위 3건 안에 정답이 들어온다.
# 부수 효과로 Chroma 메타데이터 사전필터링이 사라져 검색이 41ms -> 12ms.


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
    def search_qa(query: str) -> str:
        """생활법령 상담 자료에서 관련 내용을 검색합니다.
        법제처가 2,500개 이상의 법률을 조문번호·법정형과 함께 알기 쉽게 정리한
        자료입니다. 절차·대처 방법 안내는 물론, search_law 로 조문이 나오지 않는
        분야(동물보호, 개인정보, 정보통신망, 소비자 등)의 처벌·책임 질문에도
        여기에 근거가 있는 경우가 많으니 함께 검색하세요.

        Args:
            query: 검색할 내용 (예: '재혼 후 자녀 성 변경', '전세 계약 갱신')
        """
        docs = vectorstore.similarity_search(query, k=QA_TOP_K)
        if not docs:
            return "관련 생활법령 자료를 찾을 수 없습니다."
        return _format_qa(docs)

    return [search_qa]
