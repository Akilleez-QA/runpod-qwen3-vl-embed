FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch + transformers + flash-attn
RUN pip3 install --no-cache-dir \
    torch>=2.6.0 \
    transformers>=4.51.0 \
    accelerate>=0.30.0 \
    qwen-vl-utils>=0.0.14 \
    runpod>=1.7.0 \
    Pillow \
    sentencepiece

# Flash Attention 2 (for speed)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# Clone official Qwen3-VL-Embedding repo
RUN git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git /app
WORKDIR /app
RUN pip3 install --no-cache-dir -e . 2>/dev/null || true

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
