import os

import requests
from dotenv import load_dotenv

load_dotenv()
LAW_API_OC = os.getenv("LAW_API_OC", "")
BASE_URL = "https://www.law.go.kr/DRF"

CRIMINAL_LAWS = [
    "형법",
    "형사소송법",
    "형의 집행 및 수용자의 처우에 관한 법률",
    "특정범죄 가중처벌 등에 관한 법률",
    "특정경제범죄 가중처벌 등에 관한 법률",
    "성폭력범죄의 처벌 등에 관한 특례법",
    "성폭력방지 및 피해자보호 등에 관한 법률",
    "폭력행위 등 처벌에 관한 법률",
    "아동ㆍ청소년의 성보호에 관한 법률",
    "가정폭력범죄의 처벌 등에 관한 특례법",
    "스토킹범죄의 처벌 등에 관한 법률",
    "소년법",
    "즉결심판에 관한 절차법",
    "보안관찰법",
]

PRECEDENT_START_DATE = "20260101"
PRECEDENT_END_DATE = "20260721"
def _get(path: str, params: dict) -> dict:
    """공통 GET 요청. requests가 한글 파라미터를 자동으로 URL 인코딩해준다."""
    full_params={"OC":LAW_API_OC,"type":"JSON",**params}
    resp=requests.get(f"{BASE_URL}/{path}",params=full_params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def find_current_law(law_name: str)->dict | None:
    """법령명과 정확히 일치하는 현행 법령 정보를 찾아 반환. 없으면 None."""
    data=_get("lawSearch.do", {"target": "eflaw", "query":law_name,"display":100})
    laws=data.get("LawSearch",{}).get("law",[])
    if isinstance(laws, dict):
        laws=[laws]
    for law in laws:
        if law.get("법령명한글")==law_name and law.get("현행연혁코드")=="현행":
            return law
    return None

def get_law_full_text(mst: str) -> dict:
    """법령일련번호(MST)로 본문조회. (법령상세링크의 target=eflaw와 동일하게 맞춤)"""
    return _get("lawService.do",{"target":"eflaw","MST":mst})

def search_precedents_by_law(law_name: str, start_date: str, end_date: str, display: int =100)->list[dict]:
    """참조법령명(JO) 기준으로 판례 목록 조회 후, 사건종류명이 '형사'인 것만 반환."""
    data=_get("lawSearch.do",{
        "target":"prec",
        "JO": law_name,
        "prncYd":f"{start_date}~{end_date}",
        "display":display,
    })
    precs=data.get("PrecSearch",{}).get("prec",[])
    if isinstance(precs, dict):
        precs=[precs]
    return [p for p in precs if p.get("사건종류명")=="형사"]

def get_precedent_full_text(prec_id: str) -> dict:
    "판례일련번호로 본문조회"
    return _get("lawService.do",{"target":"prec","ID":prec_id})