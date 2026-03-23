# ─── SUM Quantum Knowledge OS ────────────────────────────────────────
# Multi-stage build: Zig core → Python API
# Usage:
#   docker build -t sum .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sum

# ── Stage 1: Build Zig Core ──────────────────────────────────────────
FROM debian:bookworm-slim AS zig-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl xz-utils ca-certificates && rm -rf /var/lib/apt/lists/*

ARG ZIG_VERSION=0.13.0
RUN curl -L https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz \
    | tar -xJ -C /opt && mv /opt/zig-linux-x86_64-${ZIG_VERSION} /opt/zig

ENV PATH="/opt/zig:$PATH"

WORKDIR /build
COPY core-zig/ ./core-zig/
RUN cd core-zig && zig build -Doptimize=ReleaseFast

# ── Stage 2: Python Runtime ──────────────────────────────────────────
FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt && \
    python -m spacy download en_core_web_sm

# Copy Zig shared library
COPY --from=zig-builder /build/core-zig/zig-out/lib/ /app/core-zig/zig-out/lib/

# Copy application
COPY . .

# Environment
ENV SUM_HOST=0.0.0.0
ENV SUM_PORT=8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/quantum/state').raise_for_status()" || exit 1

# Run
CMD ["python", "quantum_main.py"]
