# Multi-stage build for Agentic AI Fraud Detection System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install pytest pytest-asyncio pytest-cov black isort mypy
COPY . .
CMD ["python", "-m", "pytest", "-v"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY src/ ./src/
COPY demo.py .
COPY simple_demo.py .
COPY env.example .env

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/system-status || exit 1

# Expose port
EXPOSE 8000

# Default command - run web interface
CMD ["python", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000"]