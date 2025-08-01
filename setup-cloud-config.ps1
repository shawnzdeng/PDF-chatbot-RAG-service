# Quick setup script for cloud Qdrant configuration
# Run this after deploying to update your credentials

param(
    [Parameter(Mandatory=$true)]
    [string]$OpenAIApiKey,
    
    [Parameter(Mandatory=$true)]
    [string]$QdrantApiKey,
    
    [Parameter(Mandatory=$true)]
    [string]$QdrantUrl
)

Write-Host "🔧 Configuring RAG system with cloud Qdrant..." -ForegroundColor Green

# Update secrets
Write-Host "📝 Updating Kubernetes secrets..." -ForegroundColor Blue
$secretsCommand = @"
kubectl create secret generic rag-secrets \
  --from-literal=openai-api-key=$OpenAIApiKey \
  --from-literal=qdrant-api-key=$QdrantApiKey \
  -n rag-system --dry-run=client -o yaml | kubectl apply -f -
"@

Invoke-Expression $secretsCommand

# Update ConfigMap with Qdrant URL
Write-Host "🌐 Updating Qdrant URL..." -ForegroundColor Blue
$configMapPatch = @{
    data = @{
        QDRANT_URL = $QdrantUrl
    }
} | ConvertTo-Json -Compress

kubectl patch configmap rag-config -n rag-system --patch $configMapPatch

# Restart deployment to pick up new config
Write-Host "🔄 Restarting RAG chatbot deployment..." -ForegroundColor Blue
kubectl rollout restart deployment/rag-chatbot -n rag-system

# Wait for rollout to complete
Write-Host "⏳ Waiting for deployment to be ready..." -ForegroundColor Blue
kubectl rollout status deployment/rag-chatbot -n rag-system

Write-Host "✅ Configuration updated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Check status:" -ForegroundColor Cyan
Write-Host "   kubectl get pods -n rag-system" -ForegroundColor White
Write-Host "   kubectl logs -f deployment/rag-chatbot -n rag-system" -ForegroundColor White
