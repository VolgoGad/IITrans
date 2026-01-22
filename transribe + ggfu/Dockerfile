FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Optional: cache dir for HF models inside container
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

ENTRYPOINT [ "python", "main.py" ]