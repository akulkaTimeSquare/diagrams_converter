FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SKIP_PRELOAD=1
ENV MAX_IMAGE_SIDE=1024
ENV USE_GPU=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt \
    && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision

COPY . /app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
