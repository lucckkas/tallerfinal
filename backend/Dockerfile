FROM python:3.11-slim

WORKDIR /app

COPY ml/requirements.txt ./ml-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r ml-requirements.txt -r backend-requirements.txt

COPY config ./config
COPY ml ./ml
COPY backend ./backend

ENV PYTHONPATH="/app/ml/src"

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
