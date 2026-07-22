FROM python:3.12-slim

# 시스템 패키지 (일부 라이브러리 빌드에 필요할 수 있음)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv 설치
RUN pip install --no-cache-dir uv

# 의존성 먼저 복사 & 설치 (레이어 캐싱 활용)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 임베딩 모델을 빌드 시 미리 다운로드 (첫 실행 지연 방지)
RUN uv run python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')"

# 애플리케이션 코드 + 미리 구축한 벡터DB 복사
COPY . .

# static 폴더가 없으면 빈 폴더 생성 (main.py의 mount 에러 방지)
RUN mkdir -p static

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]