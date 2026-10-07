FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    RETURNKILLER_USE_BEDROCK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyzer.py app.py ./
# sample data: try local file, data/, or parent data/
COPY sample_products.json* ./
RUN mkdir -p data && \
    (cp sample_products.json data/sample_products.json 2>/dev/null || true)

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
