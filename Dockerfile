FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model-cache \
    TRANSFORMERS_CACHE=/app/model-cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt

COPY aether ./aether
COPY worker.py ./worker.py
COPY configs ./configs
COPY pyproject.toml ./

RUN useradd --create-home --uid 10001 aether \
    && mkdir -p /app/data /app/model-cache \
    && chown -R aether:aether /app

USER aether

EXPOSE 8000

CMD ["uvicorn", "aether.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
