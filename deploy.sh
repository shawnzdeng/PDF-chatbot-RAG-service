#!/bin/bash

# Deployment script for RAG Chatbot with Minikube
# This script sets up the entire stack including monitoring

set -e

echo "🚀 Starting RAG Chatbot deployment on Minikube..."

# Check if minikube is running
if ! minikube status &> /dev/null; then
    echo "❌ Minikube is not running. Please start minikube first:"
    echo "   minikube start --driver=docker --memory=8192 --cpus=4"
    exit 1
fi

# Enable required addons
echo "📦 Enabling Minikube addons..."
minikube addons enable ingress
minikube addons enable metrics-server

# Set docker environment to use minikube's docker daemon
echo "🐳 Setting up Docker environment for Minikube..."
eval $(minikube docker-env)

# Build the RAG chatbot image
echo "🔨 Building RAG Chatbot Docker image..."
docker build -t rag-chatbot:latest .

# Apply Kubernetes manifests
echo "⚙️  Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/rag-chatbot.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n rag-system
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n rag-system
kubectl wait --for=condition=available --timeout=300s deployment/rag-chatbot -n rag-system

# Get Minikube IP
MINIKUBE_IP=$(minikube ip)

# Add entries to hosts file (requires admin privileges)
echo "🌐 Setting up local DNS entries..."
echo "To access the applications, add these entries to your hosts file:"
echo "$MINIKUBE_IP rag-chatbot.local"
echo "$MINIKUBE_IP grafana.local"

# Display access information
echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Access your applications:"
echo "   RAG Chatbot:  http://rag-chatbot.local (or http://$MINIKUBE_IP:$(kubectl get svc rag-chatbot-service -n rag-system -o jsonpath='{.spec.ports[0].nodePort}') if using NodePort)"
echo "   Grafana:      http://grafana.local (admin/admin) (or http://$MINIKUBE_IP:$(kubectl get svc grafana-service -n rag-system -o jsonpath='{.spec.ports[0].nodePort}') if using NodePort)"
echo "   Prometheus:   http://$MINIKUBE_IP:$(kubectl get svc prometheus-service -n rag-system -o jsonpath='{.spec.ports[0].nodePort}') (if using NodePort)"
echo ""
echo "🔍 Useful commands:"
echo "   kubectl get pods -n rag-system"
echo "   kubectl logs -f deployment/rag-chatbot -n rag-system"
echo "   minikube dashboard"
echo ""
echo "⚠️  Don't forget to:"
echo "   1. Update the API keys and Qdrant URL with your actual values"
echo "   2. Add the DNS entries to your hosts file for domain access"
echo ""
echo "🔑 Update secrets with your actual API keys:"
echo "   kubectl create secret generic rag-secrets \\"
echo "     --from-literal=openai-api-key=YOUR_OPENAI_API_KEY \\"
echo "     --from-literal=qdrant-api-key=YOUR_QDRANT_API_KEY \\"
echo "     -n rag-system --dry-run=client -o yaml | kubectl apply -f -"
echo ""
echo "🌐 Update Qdrant URL:"
echo "   kubectl patch configmap rag-config -n rag-system \\"
echo "     --patch '{\"data\":{\"QDRANT_URL\":\"https://your-cluster-url.qdrant.io:6333\"}}'"
