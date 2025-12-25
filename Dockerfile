FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .

ARG TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --extra-index-url ${TORCH_EXTRA_INDEX_URL} -r requirements.txt

COPY . .

RUN mkdir -p logs/sessions logs/analysis temp/cache models/assets

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 9100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9100"]
