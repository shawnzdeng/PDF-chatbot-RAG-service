# Docker build script with retry logic and optimizations
param(
    [switch]$CPU = $false,
    [int]$MaxRetries = 3,
    [switch]$NoBuildCache = $false
)

$imageName = "rag-chatbot"
$tag = "latest"

if ($CPU) {
    Write-Host "🔧 Building CPU-only image (lightweight)..." -ForegroundColor Green
    $dockerfile = "Dockerfile.cpu"
    $tag = "cpu"
} else {
    Write-Host "🔧 Building full image with GPU support..." -ForegroundColor Green
    $dockerfile = "Dockerfile"
}

$buildArgs = @(
    "build"
    "-t", "$imageName`:$tag"
    "-f", $dockerfile
)

if ($NoBuildCache) {
    $buildArgs += "--no-cache"
}

# Add build optimizations
$buildArgs += @(
    "--memory=4g"
    "--memory-swap=8g"
    "."
)

$attempt = 1
$success = $false

while ($attempt -le $MaxRetries -and -not $success) {
    Write-Host "🚀 Build attempt $attempt of $MaxRetries..." -ForegroundColor Blue
    
    try {
        # Set Docker buildkit for better performance
        $env:DOCKER_BUILDKIT = "1"
        
        # Run docker build
        & docker @buildArgs
        
        if ($LASTEXITCODE -eq 0) {
            $success = $true
            Write-Host "✅ Build successful!" -ForegroundColor Green
        } else {
            throw "Docker build failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "❌ Build attempt $attempt failed: $_" -ForegroundColor Red
        
        if ($attempt -lt $MaxRetries) {
            Write-Host "⏳ Waiting 30 seconds before retry..." -ForegroundColor Yellow
            Start-Sleep -Seconds 30
            
            # Clean up failed build cache
            Write-Host "🧹 Cleaning up build cache..." -ForegroundColor Blue
            docker builder prune -f
        }
        
        $attempt++
    }
}

if (-not $success) {
    Write-Host "💥 All build attempts failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Try the CPU-only build: .\build-docker.ps1 -CPU" -ForegroundColor White
    Write-Host "2. Check Docker memory settings (increase to 8GB+)" -ForegroundColor White
    Write-Host "3. Use build without cache: .\build-docker.ps1 -NoBuildCache" -ForegroundColor White
    Write-Host "4. Check internet connection for large downloads" -ForegroundColor White
    exit 1
}

# Show image info
Write-Host ""
Write-Host "📦 Image details:" -ForegroundColor Cyan
docker images $imageName`:$tag

Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Green
Write-Host "   docker run -p 8501:8501 --env-file .env $imageName`:$tag" -ForegroundColor White
Write-Host "   Or use docker-compose up -d" -ForegroundColor White
