# ==============================================================================
# DeepSupport: Unified Multi-Framework Causal & Deep Learning Container
# Basis: Debian 12 (Bookworm) Slim mit Python 3.12
# ==============================================================================

FROM python:3.12-slim

LABEL maintainer="Wilfried Keller"
LABEL description="DeepSupport Research Environment for Causal Inference, Survival Analysis, and Deep Learning"

# Verhindere interaktive Eingabeaufforderungen bei apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV TF_ENABLE_ONEDNN_OPTS=0

# System-Abhängigkeiten installieren:
# - build-essential: C++ Compiler für native Extensions und HiGHS
# - libgomp1: OpenMP Multithreading für PyTorch und DuckDB
# - git, procps: Versionskontrolle und Prozessüberwachung (psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pip auf den neuesten Stand bringen
RUN pip install --no-cache-dir --upgrade pip

# Python-Abhängigkeiten installieren
COPY requirements.txt .

# Installation mit PyTorch CPU Extra-Index für minimale Image-Größe
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Standard-Befehl
CMD ["/bin/bash"]
