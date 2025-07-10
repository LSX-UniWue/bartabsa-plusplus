FROM docker.io/pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

SHELL ["/bin/bash", "-c"]
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /bartabsa-lightning
RUN chmod -R 777 .

COPY requirements.txt .
ARG PIP_NO_CACHE_DIR=1
RUN pip install -r requirements.txt --no-cache-dir
