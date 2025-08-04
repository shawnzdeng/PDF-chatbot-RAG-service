# PowerShell deployment script for RAG Chatbot with Minikube
# This script sets up the entire stack including monitoring

param(
    [switch]$SkipBuild = $false,
    [switch]$CPU = $true  # Use CPU build by default to avoid CUDA issues
)

Write-Host "🚀 Starting RAG Chatbot deployment on Minikube..." -ForegroundColor Green

# Check if minikube is running
try {
    $minikubeStatus = minikube status 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Minikube not running"
    }
} catch {
    Write-Host "❌ Minikube is not running. Please start minikube first:" -ForegroundColor Red
    Write-Host "   minikube start --driver=docker --memory=8192 --cpus=4" -ForegroundColor Yellow
    exit 1
}

# Enable required addons
Write-Host "📦 Enabling Minikube addons..." -ForegroundColor Blue
minikube addons enable ingress
minikube addons enable metrics-server

if (-not $SkipBuild) {
    # Set docker environment to use minikube's docker daemon
    Write-Host "🐳 Setting up Docker environment for Minikube..." -ForegroundColor Blue
    $env = minikube docker-env --shell powershell | Out-String
    Invoke-Expression $env

    # Build the RAG chatbot image
    if ($CPU) {
        Write-Host "🔨 Building RAG Chatbot Docker image (CPU-optimized)..." -ForegroundColor Blue
        .\build-docker.ps1 -CPU
        # Tag the CPU image as latest for Kubernetes
        docker tag rag-chatbot:cpu rag-chatbot:latest
    } else {
        Write-Host "🔨 Building RAG Chatbot Docker image (with GPU support)..." -ForegroundColor Blue
        .\build-docker.ps1
    }
}

# Apply Kubernetes manifests
Write-Host "⚙️  Applying Kubernetes manifests..." -ForegroundColor Blue
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/rag-chatbot.yaml
kubectl apply -f k8s/rag-secrets.yaml

# Wait for deployments to be ready
Write-Host "⏳ Waiting for deployments to be ready..." -ForegroundColor Blue
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n rag-system
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n rag-system
kubectl wait --for=condition=available --timeout=300s deployment/rag-chatbot -n rag-system

# Get Minikube IP
$minikubeIP = minikube ip

# Display access information
Write-Host ""
Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Access your applications:" -ForegroundColor Cyan
Write-Host "   RAG Chatbot:  http://rag-chatbot.local" -ForegroundColor White
Write-Host "   Grafana:      http://grafana.local (admin/admin)" -ForegroundColor White
Write-Host "   Minikube IP:  $minikubeIP" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Add these entries to your hosts file (C:\Windows\System32\drivers\etc\hosts):" -ForegroundColor Yellow
Write-Host "$minikubeIP rag-chatbot.local" -ForegroundColor White
Write-Host "$minikubeIP grafana.local" -ForegroundColor White
Write-Host ""
Write-Host "🔍 Useful commands:" -ForegroundColor Cyan
Write-Host "   kubectl get pods -n rag-system" -ForegroundColor White
Write-Host "   kubectl logs -f deployment/rag-chatbot -n rag-system" -ForegroundColor White
Write-Host "   minikube dashboard" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Don't forget to update the secrets with your actual API keys!" -ForegroundColor Yellow
Write-Host "   # Update API Keys:" -ForegroundColor White
$secretCommand = 'kubectl create secret generic rag-secrets --from-literal=OPENAI_API_KEY=YOUR_OPENAI_API_KEY --from-literal=QDRANT_API_KEY=YOUR_QDRANT_API_KEY -n rag-system --dry-run=client -o yaml | kubectl apply -f -'
Write-Host "   $secretCommand" -ForegroundColor White
Write-Host ""
Write-Host "   # Or update the configmap for Qdrant URL:" -ForegroundColor White
$configCommand = 'kubectl patch configmap rag-config -n rag-system --patch "{""data"":{""QDRANT_URL"":""https://your-cluster-url.qdrant.io:6333""}}"'
Write-Host "   $configCommand" -ForegroundColor White
