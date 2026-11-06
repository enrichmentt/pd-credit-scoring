FROM python:3.12-slim

WORKDIR /app

# Зависимости API (отдельный лёгкий requirements без mlflow/dvc)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Код и обученная модель
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
