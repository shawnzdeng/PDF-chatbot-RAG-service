#!/bin/bash

# Quick setup script for cloud Qdrant configuration
# Usage: ./setup-cloud-config.sh <openai-api-key> <qdrant-api-key> <qdrant-url>

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <openai-api-key> <qdrant-api-key> <qdrant-url>"
    echo "Example: $0 'sk-...' 'your-qdrant-key' 'https://your-cluster.qdrant.io:6333'"
    exit 1
fi

OPENAI_API_KEY="$1"
QDRANT_API_KEY="$2"
QDRANT_URL="$3"

echo "🔧 Configuring RAG system with cloud Qdrant..."

# Update secrets
echo "📝 Updating Kubernetes secrets..."
kubectl create secret generic rag-secrets \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  --from-literal=qdrant-api-key="$QDRANT_API_KEY" \
  -n rag-system --dry-run=client -o yaml | kubectl apply -f -

# Update ConfigMap with Qdrant URL
echo "🌐 Updating Qdrant URL..."
kubectl patch configmap rag-config -n rag-system \
  --patch "{\"data\":{\"QDRANT_URL\":\"$QDRANT_URL\"}}"

# Restart deployment to pick up new config
echo "🔄 Restarting RAG chatbot deployment..."
kubectl rollout restart deployment/rag-chatbot -n rag-system

# Wait for rollout to complete
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/rag-chatbot -n rag-system

echo "✅ Configuration updated successfully!"
echo ""
echo "🔍 Check status:"
echo "   kubectl get pods -n rag-system"
echo "   kubectl logs -f deployment/rag-chatbot -n rag-system"
