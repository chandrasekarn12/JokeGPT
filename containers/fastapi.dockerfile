# Base image
FROM python:3.11-slim

# Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy code
WORKDIR /app
COPY requirements.txt .

# Install python dependencies/ used cached 
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --retries 10 --timeout 120

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
        --retries 10 --timeout 120

COPY containers/app.py ./
COPY config.py model.py ./
COPY data/checkpoint.pt data/meta.pkl ./data/

# Startup
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]