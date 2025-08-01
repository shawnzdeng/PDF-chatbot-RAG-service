#!/bin/bash

# Docker build script with retry logic and optimizations

set -e

# Default values
CPU_ONLY=false
MAX_RETRIES=3
NO_BUILD_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --no-cache)
            NO_BUILD_CACHE=true
            shift
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--cpu] [--no-cache] [--max-retries N]"
            exit 1
            ;;
    esac
done

IMAGE_NAME="rag-chatbot"
TAG="latest"

if [ "$CPU_ONLY" = true ]; then
    echo "🔧 Building CPU-only image (lightweight)..."
    DOCKERFILE="Dockerfile.cpu"
    TAG="cpu"
else
    echo "🔧 Building full image with GPU support..."
    DOCKERFILE="Dockerfile"
fi

# Build arguments
BUILD_ARGS=(
    "build"
    "-t" "${IMAGE_NAME}:${TAG}"
    "-f" "$DOCKERFILE"
)

if [ "$NO_BUILD_CACHE" = true ]; then
    BUILD_ARGS+=("--no-cache")
fi

# Add build optimizations
BUILD_ARGS+=(
    "--memory=4g"
    "--memory-swap=8g"
    "."
)

attempt=1
success=false

while [ $attempt -le $MAX_RETRIES ] && [ "$success" = false ]; do
    echo "🚀 Build attempt $attempt of $MAX_RETRIES..."
    
    # Set Docker buildkit for better performance
    export DOCKER_BUILDKIT=1
    
    if docker "${BUILD_ARGS[@]}"; then
        success=true
        echo "✅ Build successful!"
    else
        echo "❌ Build attempt $attempt failed"
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "⏳ Waiting 30 seconds before retry..."
            sleep 30
            
            # Clean up failed build cache
            echo "🧹 Cleaning up build cache..."
            docker builder prune -f || true
        fi
        
        attempt=$((attempt + 1))
    fi
done

if [ "$success" = false ]; then
    echo "💥 All build attempts failed!"
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "1. Try the CPU-only build: ./build-docker.sh --cpu"
    echo "2. Check Docker memory settings (increase to 8GB+)"
    echo "3. Use build without cache: ./build-docker.sh --no-cache"
    echo "4. Check internet connection for large downloads"
    exit 1
fi

# Show image info
echo ""
echo "📦 Image details:"
docker images "${IMAGE_NAME}:${TAG}"

echo ""
echo "🎯 Next steps:"
echo "   docker run -p 8501:8501 --env-file .env ${IMAGE_NAME}:${TAG}"
echo "   Or use docker-compose up -d"
