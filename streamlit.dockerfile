# Base image
FROM python:3.11-slim

WORKDIR /app

# Install python dependencies/ used cached 
RUN pip install --no-cache-dir streamlit requests

COPY demo.py .

# Backend URL
ENV BACKEND_URL=https://jokegpt-api-gcp-892107215241.us-central1.run.app/

# Startup
EXPOSE 8080
CMD ["streamlit", "run", "demo.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.fileWatcherType", "none"]