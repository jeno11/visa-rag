FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY static/ static/
COPY data/raw/ data/raw/
COPY data/chunks.jsonl data/chunks.jsonl

# Download the model and build the index once, at build time
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -m src.build_index

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]