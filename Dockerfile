# Inference service image. Build after exporting the model:
#   python -m training.export_model --version v1 --out models
#   docker build -t mlops-agent-api:1.0.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/
COPY models/ models/

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
