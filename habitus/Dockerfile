ARG BUILD_FROM=ghcr.io/hassio-addons/debian-base:9.1.0
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY habitus/ ./habitus/
COPY run.sh .
RUN chmod +x run.sh

CMD ["/app/run.sh"]
