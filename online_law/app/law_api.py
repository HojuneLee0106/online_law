import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
LAW_API_OC = os.getenv("LAW_API_OC", "")
BASE_URL = "https://www.law.go.kr/DRF"

LAW_DOMAINS = {
    "criminal": [
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
    ],
    "housing": [
        "주택임대차보호법",
        "상가건물 임대차보호법",
    ],
    "labor": [
        "근로기준법",
        "근로자퇴직급여 보장법",
        "최저임금법",
        "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
        "산업재해보상보험법",
    ],
    "traffic": [
        "도로교통법",
        "교통사고처리 특례법",
        "자동차손해배상 보장법",
        "특정범죄 가중처벌 등에 관한 법률",   # criminal과 중복 (위험운전치사상)
    ],
    "civil": [
        "민법",
        "민사소송법",
        "민사집행법",
        "채무자 회생 및 파산에 관한 법률",
    ],
}

PRECEDENT_START_DATE = "20200101"
PRECEDENT_END_DATE = "20260721"
SUPREME_COURT_CODE = "400201"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
# law.go.kr은 부하가 걸리면 정상 요청에도 간헐적으로 404를 반환한다
# (같은 요청을 다시 보내면 200이 온다). 따라서 404도 재시도 대상에 포함한다.
RETRYABLE_STATUS = {404, 429}


def _get(path: str, params: dict) -> dict:
    """공통 GET 요청. requests가 한글 파라미터를 자동으로 URL 인코딩해준다.
    일시적인 네트워크 오류로 대량 수집이 통째로 중단되지 않도록 지수 백오프로 재시도한다."""
    full_params={"OC":LAW_API_OC,"type":"JSON",**params}
    last_err=None
    for attempt in range(MAX_RETRIES):
        try:
            resp=requests.get(
                f"{BASE_URL}/{path}", params=full_params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err=e
        except requests.HTTPError as e:
            # 5xx와 일시적 4xx(404/429)는 재시도, 그 외 4xx는 즉시 실패
            status=e.response.status_code if e.response is not None else None
            if status is None or (status < 500 and status not in RETRYABLE_STATUS):
                raise
            last_err=e
        wait=2 ** attempt
        print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {type(last_err).__name__} -> {wait}초 후 재시도")
        time.sleep(wait)
    raise last_err

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
    """법령일련번호(MST)로 본문조회."""
    return _get("lawService.do",{"target":"eflaw","MST":mst})

def search_precedents_by_law(
    law_name: str,
    start_date: str,
    end_date: str,
    display: int = 100,
    supreme_only: bool = True,
) -> list[dict]:
    """참조법령명(JO) + 기간으로 판례 목록 조회.
    주의: org 파라미터를 prncYd와 함께 쓰면 결과가 0건이 되므로,
    대법원 필터링은 응답에서 직접 수행한다.
    display는 한 페이지 상한(100)이므로, totalCnt를 보고 끝까지 페이징한다."""
    collected=[]
    page=1
    while True:
        params={
            "target": "prec",
            "JO": law_name,
            "prncYd": f"{start_date}~{end_date}",
            "display": display,
            "page": page,
        }
        data = _get("lawSearch.do", params)
        search = data.get("PrecSearch", {})
        precs = search.get("prec", [])
        if isinstance(precs, dict):
            precs=[precs]
        if not precs:
            break
        collected.extend(precs)
        total = int(search.get("totalCnt", 0))
        if len(collected) >= total:
            break
        page += 1
        time.sleep(0.2)

    if supreme_only:
        collected = [p for p in collected if p.get("법원명") == "대법원"]

    return collected

def get_precedent_full_text(prec_id: str) -> dict:
    "판례일련번호로 본문조회"
    return _get("lawService.do",{"target":"prec","ID":prec_id})