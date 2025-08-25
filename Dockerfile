FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# System deps (often not needed if using wheels, but harmless and small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY hoops-main/ /app/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["bash", "-lc", "gunicorn api_server:api --bind 0.0.0.0:${PORT} --timeout 120"]


