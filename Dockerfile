# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src .
ENV PYTHONUNBUFFERED=true
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
