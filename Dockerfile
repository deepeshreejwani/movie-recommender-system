FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RECSYS_PROJECT_ROOT=/app \
    PYTHONPATH="/app/src:/app"

WORKDIR /app


# ---------------------------------------------------------
# Runtime dependencies
# ---------------------------------------------------------

COPY requirements-serving.txt ./

# Install CPU-only PyTorch explicitly so CUDA/NVIDIA
# dependencies are not pulled into the serving image.
RUN python -m pip install \
        --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0" \
    && python -m pip install \
        --no-cache-dir \
        -r requirements-serving.txt


# ---------------------------------------------------------
# Application code
# ---------------------------------------------------------

COPY src ./src
COPY app ./app


# ---------------------------------------------------------
# Runtime artifact mount points
# ---------------------------------------------------------

# Model and serving artifacts are intentionally NOT copied
# into the Docker image.
#
# Mount the local artifacts directory at runtime:
#
#     /app/artifacts

RUN mkdir -p \
    /app/artifacts/models \
    /app/artifacts/embeddings \
    /app/artifacts/serving


# ---------------------------------------------------------
# Non-root runtime user
# ---------------------------------------------------------

RUN useradd \
        --create-home \
        --uid 10001 \
        appuser \
    && chown -R appuser:appuser /app

USER appuser


# ---------------------------------------------------------
# API
# ---------------------------------------------------------

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]