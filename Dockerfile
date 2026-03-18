FROM fedora:42

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN dnf upgrade -y && \
    dnf install -y \
        gcc ffmpeg libSM libXext \
        openslide-devel \
        vips-devel && \
    dnf clean all

RUN uv python install 3.12

COPY . /app

RUN uv pip install --system --no-cache .
