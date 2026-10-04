FROM python:3.12.8-slim-bookworm AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --prefix=/install -r requirements.txt

FROM python:3.12.8-slim-bookworm

ENV HOME=/tmp \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIH_HOSTED=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd --system app && useradd --system --gid app --home-dir /tmp app

COPY --from=dependencies /install /usr/local
WORKDIR /app
COPY --chown=app:app app.py .
COPY --chown=app:app sih26147/ ./sih26147/
COPY --chown=app:app .streamlit/config.toml .streamlit/config.toml

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=3)"
CMD ["sh", "-c", "exec streamlit run app.py --server.address 0.0.0.0 --server.port \"${PORT:-8501}\""]
