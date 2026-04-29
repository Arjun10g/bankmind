# BankMind: multi-domain RAG dashboard.
# Single-stage image. Built with HF Spaces and self-hosting in mind.

FROM python:3.11-slim

# System deps for pdfplumber/unstructured/lxml/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Match HF Spaces layout: a non-root user with a known home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Install deps as the non-root user (HF Spaces requirement).
COPY --chown=user:user requirements.txt /app/requirements.txt
USER user
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r /app/requirements.txt

# Copy the rest of the project.
COPY --chown=user:user . /app

EXPOSE 7860

# `app.py` at the repo root is the entry point that HF Spaces auto-discovers.
CMD ["python", "app.py"]
