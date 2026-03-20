FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Runtime dependencies for opencv and openslide-python
# OpenSlide 4.0 (with CZI support) is provided by the openslide-bin pip package
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN uv pip install --system --no-cache .
