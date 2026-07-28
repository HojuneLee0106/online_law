from law_api import _get

CANDIDATES = [
    # housing
    "주택임대차보호법",
    "상가건물 임대차보호법",
    # labor
    "근로기준법",
    "근로자퇴직급여 보장법",
    "최저임금법",
    "남녀고용평등과 일·가정 양립 지원에 관한 법률",
    "산업재해보상보험법",
    # traffic
    "도로교통법",
    "교통사고처리 특례법",
    "자동차손해배상 보장법",
    # debt / civil
    "민법",
    "민사집행법",
    "채무자 회생 및 파산에 관한 법률",
    "민사소송법",
]


def check(name: str):
    data = _get("lawSearch.do", {"target": "eflaw", "query": name, "display": 100})
    laws = data.get("LawSearch", {}).get("law", [])
    if isinstance(laws, dict):
        laws = [laws]

    # 현행 법령 중 이름이 비슷한 것들 추출
    matches = [
        l for l in laws
        if l.get("현행연혁코드") == "현행" and l.get("법령구분명") == "법률"
    ]

    print(f"\n{'=' * 70}")
    print(f"검색어: {name!r}")
    if not matches:
        print("  ⚠️  현행 법률 없음")
        return

    for l in matches[:5]:
        actual = l.get("법령명한글")
        exact = "✅ 정확히 일치" if actual == name else "⚠️  다름"
        print(f"  {exact}")
        print(f"    실제명: {actual!r}")
        print(f"    MST: {l.get('법령일련번호')}, 시행일: {l.get('시행일자')}")


if __name__ == "__main__":
    for name in CANDIDATES:
        check(name)