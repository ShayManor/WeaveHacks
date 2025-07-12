# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src .
ENV PYTHONUNBUFFERED=true
CMD ["sh", "-c", "exec gunicorn -b 0.0.0.0:${PORT} app:app"]
