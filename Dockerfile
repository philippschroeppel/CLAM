FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ffmpeg libsm6 libxext6 libglib2.0-0 \
    && add-apt-repository ppa:openslide/openslide \
    && apt-get update && apt-get install -y --no-install-recommends \
        openslide-tools \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.12

COPY . /app

RUN uv pip install --system --no-cache .
