FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src


COPY models /app/models


ENV MODEL_PATH=/app/models/rf_classifier.joblib \
    KM_PATH=/app/models/kmeans.joblib \
    ISO_PATH=/app/models/isoforest.joblib \
    PROTOCOL_MODEL_PATH=/app/models/rf_protocol_classifier.joblib

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
