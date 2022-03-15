FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/home/python/.local/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev python-is-python3 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pip==22.0.4 pip-tools==6.5.1 wheel==0.37.1

RUN groupadd --gid 1000 python && \
    useradd  --uid 1000 --gid python --shell /bin/bash --create-home python

USER 1000
RUN mkdir /home/python/app
WORKDIR /home/python/app

COPY --chown=python:python requirements.txt .

RUN python3 -m piptools sync

COPY --chown=python:python . .
