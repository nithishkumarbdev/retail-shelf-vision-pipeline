FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY services/ services/
COPY orchestrator/ orchestrator/
COPY templates/ templates/
RUN mkdir -p static/uploads static/outputs logs

ENV ORCHESTRATOR_PORT=5000
ENV PYTHONPATH=/app
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "60", "orchestrator.app:app"]
