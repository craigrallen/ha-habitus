ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12
FROM $BUILD_FROM

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY habitus/ ./habitus/
COPY run.sh .
RUN chmod +x run.sh

CMD ["/app/run.sh"]
