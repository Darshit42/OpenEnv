# ── OpenEnv SRE — Dockerfile ─────────────────────────────────────────────────
# Multi-stage: builder (trains models) → runtime (API server)
# Base image pinned for full reproducibility.

FROM python:3.11.9-slim AS builder

WORKDIR /app

# Threading safety — prevent OpenMP SHM crashes in containers
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# System deps for causal-learn and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY backend/ /app/backend/

# Pre-train all ML models (IsolationForest + XGBoost + LSTM) at build time
RUN cd /app && python backend/models/train_models.py

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11.9-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source + pre-trained models
COPY --from=builder /app/backend /app/backend

# Copy frontend (served as static files via FastAPI StaticFiles)
COPY frontend/ /app/frontend/

# Copy root-level project files required by validator
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY pyproject.toml /app/pyproject.toml
COPY server/ /app/server/

# Environment
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1
ENV RANDOM_SEED=42
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Expose API port
EXPOSE 7860

# Health check — must match the port we serve on
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()"

# Start FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
