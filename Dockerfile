# Multi-stage build for RAG Chatbot application with optimized dependency handling
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies in stages to handle large downloads
# First install core dependencies without GPU support
RUN pip install --no-cache-dir \
    langchain==0.1.20 \
    langchain-openai==0.1.7 \
    langchain-core==0.1.53 \
    openai==1.30.1 \
    qdrant-client==1.9.1 \
    streamlit==1.47.1 \
    python-dotenv==1.0.1 \
    pytest==8.4.1 \
    pandas==2.0.3 \
    numpy==1.24.4 \
    psutil==6.1.1

# Install sentence-transformers and transformers (smaller packages first)
RUN pip install --no-cache-dir \
    transformers==4.54.0

# Install PyTorch CPU version to avoid CUDA dependencies in container
# This reduces image size significantly and works for most inference workloads
RUN pip install --no-cache-dir \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers after torch
RUN pip install --no-cache-dir \
    sentence-transformers==5.0.0

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["python", "-m", "streamlit", "run", "src/ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
