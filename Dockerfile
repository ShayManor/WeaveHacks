FROM python:3.11-slim
WORKDIR .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY src/ ./src/
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["sh","-c","exec gunicorn app:app --bind 0.0.0.0:${PORT}"]