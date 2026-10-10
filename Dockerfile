# AWS Lambda Web Adapter: lets the unchanged FastAPI app run on Lambda.
# Pin a released tag (https://github.com/awslabs/aws-lambda-web-adapter/releases).
# COPY --from cannot expand variables, so the adapter is declared as its own stage.
ARG LWA_VERSION=0.9.1
FROM public.ecr.aws/awsguru/aws-lambda-adapter:${LWA_VERSION} AS lwa

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyzer.py app.py catalog.py mcp_server.py sample_products.json ./
COPY static ./static
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
