FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    FASTEMBED_CACHE_PATH=/opt/models \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN useradd --uid 1000 --create-home app \
    && mkdir -p /app/data \
    && chown -R app /app/data /opt/models
USER app

EXPOSE 7860
CMD ["uvicorn", "investigation_engine.app.main:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
