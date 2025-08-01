# RAG System Docker & Kubernetes Deployment

This guide helps you deploy the RAG Chatbot system using Docker and Kubernetes (Minikube) with comprehensive monitoring via Prometheus and Grafana.

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (with Kubernetes enabled)
- **Minikube** (v1.32.0 or later)
- **kubectl** (v1.28.0 or later)
- **Git**

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: 4 cores recommended
- **Disk**: 20GB free space

## 🚀 Quick Start

### 1. Environment Setup

1. **Clone and setup environment**:
   ```bash
   git clone <your-repo-url>
   cd PDF-chatbot-RAG-service
   cp .env.example .env
   ```

2. **Configure your credentials**:
   Edit `.env` file and add your actual keys and URLs:
   ```
   OPENAI_API_KEY=your-actual-openai-api-key-here
   QDRANT_URL=https://your-cluster-url.qdrant.io:6333
   QDRANT_API_KEY=your-actual-qdrant-api-key-here
   ```

### 2. Local Development with Docker Compose

For quick local testing:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-chatbot

# Stop services
docker-compose down
```

**Access points**:
- RAG Chatbot: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Qdrant: Your cloud Qdrant URL (managed service)

### 3. Production Deployment with Minikube

#### Windows (PowerShell)

```powershell
# Start Minikube
minikube start --driver=docker --memory=8192 --cpus=4

# Deploy the entire stack
.\deploy.ps1

# Check deployment status
kubectl get pods -n rag-system
```

#### Linux/macOS (Bash)

```bash
# Start Minikube
minikube start --driver=docker --memory=8192 --cpus=4

# Make script executable and deploy
chmod +x deploy.sh
./deploy.sh

# Check deployment status
kubectl get pods -n rag-system
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Chatbot   │    │  Qdrant Cloud   │    │   Prometheus    │
│   (Streamlit)   │◄──►│  Vector Store   │    │   (Metrics)     │
│                 │    │   (Managed)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         └──────────────►│    Grafana      │◄────────────┘
                        │ (Visualization) │
                        └─────────────────┘
```

### Components

1. **RAG Chatbot**: Streamlit-based UI with RAG capabilities
2. **Qdrant Cloud**: Managed vector database service
3. **Prometheus**: Metrics collection and monitoring
4. **Grafana**: Visualization and dashboards
5. **Node Exporter**: System metrics collection

## 📊 Monitoring & Metrics

### Available Metrics

- **Request metrics**: Total requests, request rate, response times
- **Error metrics**: Error counts by type, error rates
- **System metrics**: CPU, memory, disk usage
- **RAG operations**: Embedding operations, retrievals, response generations
- **Session metrics**: Active sessions, user activity

### Accessing Monitoring

After deployment, access monitoring through:

1. **Grafana Dashboard**: 
   - URL: http://grafana.local (or http://MINIKUBE_IP:grafana-port)
   - Credentials: admin/admin
   - Pre-configured RAG System dashboard

2. **Prometheus**: 
   - URL: http://MINIKUBE_IP:prometheus-port
   - Query interface for custom metrics

### Setting up DNS (Required for Ingress)

Add these entries to your hosts file:

**Windows**: `C:\Windows\System32\drivers\etc\hosts`
**Linux/macOS**: `/etc/hosts`

```
MINIKUBE_IP rag-chatbot.local
MINIKUBE_IP grafana.local
```

Get your Minikube IP with: `minikube ip`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `QDRANT_URL` | Cloud Qdrant cluster URL | Required |
| `QDRANT_API_KEY` | Your Qdrant API key | Required |
| `PYTHONPATH` | Python path for imports | /app |

### Kubernetes Secrets

Update the secrets with your actual credentials:

```bash
# Update both API keys
kubectl create secret generic rag-secrets \
  --from-literal=openai-api-key=YOUR_OPENAI_API_KEY \
  --from-literal=qdrant-api-key=YOUR_QDRANT_API_KEY \
  -n rag-system --dry-run=client -o yaml | kubectl apply -f -

# Update Qdrant URL in ConfigMap
kubectl patch configmap rag-config -n rag-system \
  --patch '{"data":{"QDRANT_URL":"https://your-cluster-url.qdrant.io:6333"}}'
```

### Quick Setup Script

Use the provided setup script for easy configuration:

```powershell
# PowerShell
.\setup-cloud-config.ps1 -OpenAIApiKey "your-key" -QdrantApiKey "your-key" -QdrantUrl "https://your-cluster.qdrant.io:6333"
```

## 🔍 Troubleshooting

### Common Issues

1. **Pods stuck in Pending state**:
   ```bash
   kubectl describe pod POD_NAME -n rag-system
   # Check resource availability
   kubectl top nodes
   ```

2. **ImagePullBackOff errors**:
   ```bash
   # Ensure Docker environment is set for Minikube
   eval $(minikube docker-env)
   docker build -t rag-chatbot:latest .
   ```

3. **Service not accessible**:
   ```bash
   # Check service status
   kubectl get svc -n rag-system
   # Port forward if needed
   kubectl port-forward svc/rag-chatbot-service 8501:8501 -n rag-system
   ```

4. **Ingress not working**:
   ```bash
   # Enable ingress addon
   minikube addons enable ingress
   # Check ingress controller
   kubectl get pods -n ingress-nginx
   ```

### Useful Commands

```bash
# View all resources
kubectl get all -n rag-system

# Check logs
kubectl logs -f deployment/rag-chatbot -n rag-system
kubectl logs -f deployment/qdrant -n rag-system

# Describe resources for troubleshooting
kubectl describe deployment rag-chatbot -n rag-system

# Access Minikube dashboard
minikube dashboard

# Clean up deployment
kubectl delete namespace rag-system
```

## 🔄 Updates and Maintenance

### Updating the Application

1. **Make code changes**
2. **Rebuild image**:
   ```bash
   eval $(minikube docker-env)
   docker build -t rag-chatbot:latest .
   ```
3. **Restart deployment**:
   ```bash
   kubectl rollout restart deployment/rag-chatbot -n rag-system
   ```

### Scaling

```bash
# Scale RAG chatbot replicas
kubectl scale deployment rag-chatbot --replicas=3 -n rag-system

# Check resource usage
kubectl top pods -n rag-system
```

## 📈 Performance Optimization

### Resource Allocation

Adjust resource limits in the Kubernetes manifests based on your needs:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Monitoring Best Practices

1. Set up alerts in Grafana for high resource usage
2. Monitor response times and error rates
3. Use resource quotas to prevent resource exhaustion
4. Regular backup of Qdrant data (persistent volumes)

## 🛡️ Security Considerations

1. **Secrets Management**: Store API keys securely in Kubernetes secrets
2. **Network Policies**: Implement network policies for pod-to-pod communication
3. **RBAC**: Use role-based access control for kubectl access
4. **Image Security**: Scan Docker images for vulnerabilities
5. **TLS**: Enable TLS for production deployments

## 📚 Additional Resources

- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

## 📄 License

[Your License Here]
