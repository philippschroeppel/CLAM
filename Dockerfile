FROM ghcr.io/astral-sh/uv:python3.12-noble

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y gcc ffmpeg libsm6 libxext6 libvips-dev

RUN uv pip install --system --no-cache .
