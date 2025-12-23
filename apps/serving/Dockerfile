# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU version for building (lighter)
RUN uv pip install --no-cache -e "." \
    && uv pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src ./src

# ============================================
# Stage 2: Production (GPU)
# ============================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runner-gpu

WORKDIR /app

# Install Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Reinstall PyTorch with CUDA support
RUN uv pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy source code
COPY --from=builder /app/src ./src

# Add non-root user
RUN groupadd --system --gid 1001 python && \
    useradd --system --uid 1001 --gid python fastapi && \
    chown -R fastapi:python /app

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV USE_GPU=true
ENV PORT=8000

USER fastapi

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 3: Production (CPU only)
# ============================================
FROM python:3.13-slim AS runner-cpu

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY --from=builder /app/src ./src

# Add non-root user
RUN groupadd --system --gid 1001 python && \
    useradd --system --uid 1001 --gid python fastapi && \
    chown -R fastapi:python /app

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV USE_GPU=false
ENV PORT=8000

USER fastapi

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
