FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --system claimai \
    && useradd --system --gid claimai --home-dir /app --create-home claimai

WORKDIR /app

COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/src /app/backend/src
# Editable install keeps repository-relative synthetic fixture/config paths at /app.
RUN python -m pip install --no-cache-dir -e /app/backend

COPY --chown=claimai:claimai configs /app/configs
COPY --chown=claimai:claimai samples /app/samples

USER claimai
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)" || exit 1

CMD ["uvicorn", "claim_ai.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
