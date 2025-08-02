# Docker Build Troubleshooting Guide

## Common Docker Build Issues and Solutions

### Issue: CUDA Dependencies Download Failure
```
failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF
```

**Root Cause**: Large CUDA dependencies (nvidia-cudnn-cu12, torch with CUDA) causing network timeouts and memory issues.

### Solutions

#### 1. Use CPU-Only Build (Recommended)
```powershell
# PowerShell
.\build-docker.ps1 -CPU

# Or with docker-compose
$env:DOCKERFILE = "Dockerfile.cpu"
docker-compose build
```

```bash
# Bash
./build-docker.sh --cpu

# Or with docker-compose
export DOCKERFILE=Dockerfile.cpu
docker-compose build
```

#### 2. Increase Docker Resources
- **Memory**: Increase to 8GB+ in Docker Desktop settings
- **Disk Space**: Ensure 20GB+ free space
- **CPU**: Allocate 4+ cores for build

#### 3. Use Build Script with Retry Logic
```powershell
# With retries and cleanup
.\build-docker.ps1 -MaxRetries 5 -NoBuildCache
```

#### 4. Manual Step-by-Step Build
If automated builds fail, build manually:

```bash
# Build base image first
docker build --target base -t rag-base .

# Then build final image
docker build -t rag-chatbot:latest .
```

### Network-Related Issues

#### Slow Downloads
```bash
# Increase pip timeout
export PIP_DEFAULT_TIMEOUT=1000

# Use different index
pip install --index-url https://pypi.org/simple/ package-name
```

#### Corporate Firewalls
```bash
# Set proxy if needed
docker build --build-arg HTTP_PROXY=http://proxy:port \
             --build-arg HTTPS_PROXY=http://proxy:port \
             -t rag-chatbot .
```

### Memory Issues

#### Symptoms
- Build process killed
- "No space left on device" errors
- Slow build performance

#### Solutions
```bash
# Clean Docker cache
docker system prune -af
docker builder prune -af

# Increase Docker memory in Docker Desktop
# Or use memory-optimized build
docker build --memory=4g --memory-swap=8g -t rag-chatbot .
```

### Alternative Approaches

#### 1. Multi-Stage Build with Pre-built Base
Create a base image with dependencies:

```dockerfile
# Dockerfile.base
FROM python:3.10-slim
RUN pip install torch transformers sentence-transformers
```

#### 2. Use Official PyTorch Images
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# Add your app code
```

#### 3. Install Dependencies at Runtime
```dockerfile
# Install heavy dependencies when container starts
CMD pip install torch && python app.py
```

### Performance Optimization

#### BuildKit Features
```bash
# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1
docker build .
```

#### Cache Mounting
```dockerfile
# In Dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### Quick Tests

#### Test CPU-Only Build
```bash
# Quick test of CPU build
docker run --rm rag-chatbot:cpu python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### Test GPU Build (if needed)
```bash
# Test GPU availability
docker run --gpus all --rm rag-chatbot:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment-Specific Solutions

#### Windows with WSL2
```powershell
# Increase WSL2 memory
# Create .wslconfig in %UserProfile%
[wsl2]
memory=8GB
swap=4GB
```

#### macOS
```bash
# Increase Docker Desktop memory in preferences
# Use Rosetta 2 if on Apple Silicon
```

#### Linux
```bash
# Increase system memory for Docker
sudo systemctl edit docker
# Add:
[Service]
LimitMEMLOCK=infinity
```

### Production Considerations

For production deployments:

1. **Use multi-stage builds** to reduce final image size
2. **Pin exact versions** of all dependencies  
3. **Use official base images** when possible
4. **Implement health checks** for reliability
5. **Use CPU builds** unless GPU is absolutely necessary

### Getting Help

If issues persist:

1. Check Docker Desktop logs
2. Use `docker build --progress=plain` for detailed output
3. Try building with `--no-cache` flag
4. Consider using cloud build services (Docker Hub, GitHub Actions)

## Quick Commands Reference

```bash
# Clean everything
docker system prune -af

# Build with CPU only
docker build -f Dockerfile.cpu -t rag-chatbot:cpu .

# Build with retries
for i in {1..3}; do docker build . && break || sleep 30; done

# Check image size
docker images rag-chatbot

# Test container
docker run --rm -p 8501:8501 rag-chatbot:cpu
```
