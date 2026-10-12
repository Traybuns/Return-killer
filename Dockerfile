# syntax=docker/dockerfile:1
# AWS Lambda Web Adapter: lets the unchanged FastAPI app run on Lambda.
ARG LWA_VERSION=0.9.1
FROM public.ecr.aws/awsguru/aws-lambda-adapter:${LWA_VERSION} AS lwa

# Host architecture (Intel). Do not pip-install under QEMU.
FROM --platform=$BUILDPLATFORM public.ecr.aws/docker/library/python:3.12-slim AS wheels
WORKDIR /w
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip download \
      --dest /wheels \
      --only-binary=:all: \
      --platform manylinux_2_34_aarch64 \
      --implementation cp \
      --python-version 312 \
      --abi cp312 \
      --timeout 180 \
      --retries 10 \
      -r requirements.txt

FROM public.ecr.aws/docker/library/python:3.12-slim
COPY --from=lwa /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

ARG BUILD_SHA=unknown
ENV BUILD_SHA=${BUILD_SHA} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    RETURNKILLER_USE_BEDROCK=1

COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

COPY analyzer.py app.py catalog.py research.py mcp_server.py ./
COPY static ./static
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
