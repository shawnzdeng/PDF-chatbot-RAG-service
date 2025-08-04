# 🔄 RAG Chatbot System Restart Guide

This guide provides step-by-step instructions for restarting the RAG Chatbot system after a computer restart, Docker restart, or Kubernetes cluster restart.

## 📋 Prerequisites

- Docker Desktop installed and running
- Minikube installed
- kubectl configured
- PowerShell with administrator privileges

## 🚀 Complete System Restart Procedure

### Step 1: Start Docker Desktop
```powershell
# Ensure Docker Desktop is running
# Check Docker status
docker version
```

### Step 2: Start Minikube
```powershell
# Start minikube with sufficient resources
minikube start --driver=docker --memory=2200 --cpus=4

# Verify minikube is running
minikube status
```

### Step 3: Configure Docker Environment for Minikube
```powershell
# Set up minikube docker environment
minikube docker-env --shell powershell | Invoke-Expression

# Verify connection
docker images
```

### Step 4: Build and Tag Docker Images
```powershell
# Navigate to project directory
cd "c:\Users\wanxi\Desktop\BMO_Project\GitHub\PDF-chatbot-RAG-service"

# Build the Docker image
docker build -t rag-chatbot:cpu -f Dockerfile.cpu .

# Tag the image for Kubernetes
docker tag rag-chatbot:cpu rag-chatbot:latest

# Verify images exist
docker images | findstr "rag-chatbot"
```

### Step 5: Apply Kubernetes Configurations
```powershell
# Apply all Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rag-secrets.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml
kubectl apply -f k8s/rag-chatbot.yaml

# Verify all pods are running
kubectl get pods -n rag-system
```

### Step 6: Wait for Pods to be Ready
```powershell
# Check pod status (wait until all show 1/1 Running)
kubectl get pods -n rag-system -w

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# grafana-77f895bfb7-xxxxx       1/1     Running   0          XXm
# prometheus-649ccb7875-xxxxx    1/1     Running   0          XXm
# rag-chatbot-6787685f74-xxxxx   1/1     Running   0          XXm
```

### Step 7: Set Up Port Forwarding

#### Option A: Individual Terminal Windows (Recommended)
Open **three separate PowerShell windows as Administrator** and run:

**Terminal 1 - RAG Chatbot:**
```powershell
kubectl port-forward -n rag-system svc/rag-chatbot-service 8501:8501
```

**Terminal 2 - Grafana:**
```powershell
kubectl port-forward -n rag-system svc/grafana-service 3000:3000
```

**Terminal 3 - Prometheus:**
```powershell
kubectl port-forward -n rag-system svc/prometheus-service 9090:9090
```

#### Option B: Automated Port Forwarding Script
```powershell
# Start all port forwards in background
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Write-Host "RAG Chatbot Port Forward Active" -ForegroundColor Green; kubectl port-forward -n rag-system svc/rag-chatbot-service 8501:8501'

Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Write-Host "Grafana Port Forward Active" -ForegroundColor Green; kubectl port-forward -n rag-system svc/grafana-service 3000:3000'

Start-Process powershell -ArgumentList '-NoExit', '-Command', 'Write-Host "Prometheus Port Forward Active" -ForegroundColor Green; kubectl port-forward -n rag-system svc/prometheus-service 9090:9090'
```

### Step 8: Update Hosts File (if using ingress)
If you want to use the original ingress setup, update your hosts file:

```powershell
# Get minikube IP
minikube ip

# Add to C:\Windows\System32\drivers\etc\hosts (as Administrator)
# Replace X.X.X.X with your minikube IP
X.X.X.X rag-chatbot.local
X.X.X.X grafana.local
```

### Step 9: Start Minikube Tunnel (for ingress access)
```powershell
# Open PowerShell as Administrator
minikube tunnel
# Keep this terminal open
```

## 🌐 Access URLs

After completing all steps, your services will be available at:

### Port Forwarding Access (Recommended)
- **RAG Chatbot:** http://localhost:8501
- **Grafana Dashboard:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090

### Ingress Access (Alternative)
- **RAG Chatbot:** http://rag-chatbot.local
- **Grafana Dashboard:** http://grafana.local

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Pod in `ErrImageNeverPull` status
```powershell
# Rebuild and retag the image
docker build -t rag-chatbot:cpu -f Dockerfile.cpu .
docker tag rag-chatbot:cpu rag-chatbot:latest
kubectl delete pod -n rag-system -l app=rag-chatbot
```

#### 2. Port forwarding not working
```powershell
# Check if ports are in use
netstat -an | findstr ":8501"

# Kill existing port forwards and restart
taskkill /f /im kubectl.exe
# Then restart port forwarding
```

#### 3. Minikube won't start
```powershell
# Delete and recreate minikube
minikube delete
minikube start --driver=docker --memory=2200 --cpus=4
```

#### 4. Services not accessible
```powershell
# Check service and pod status
kubectl get svc -n rag-system
kubectl get pods -n rag-system
kubectl describe pod -n rag-system <pod-name>
```

## ⚡ Quick Start Script

Create a PowerShell script `start-rag-system.ps1` for automated startup:

```powershell
# start-rag-system.ps1
Write-Host "🚀 Starting RAG Chatbot System..." -ForegroundColor Green

# Start minikube
Write-Host "Starting Minikube..." -ForegroundColor Yellow
minikube start --driver=docker --memory=2200 --cpus=4

# Configure docker environment
Write-Host "Configuring Docker environment..." -ForegroundColor Yellow
minikube docker-env --shell powershell | Invoke-Expression

# Build images
Write-Host "Building Docker images..." -ForegroundColor Yellow
docker build -t rag-chatbot:cpu -f Dockerfile.cpu .
docker tag rag-chatbot:cpu rag-chatbot:latest

# Apply Kubernetes configs
Write-Host "Applying Kubernetes configurations..." -ForegroundColor Yellow
kubectl apply -f k8s/

# Wait for pods
Write-Host "Waiting for pods to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=ready pod --all -n rag-system --timeout=300s

# Start port forwarding
Write-Host "Starting port forwarding..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'kubectl port-forward -n rag-system svc/rag-chatbot-service 8501:8501'
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'kubectl port-forward -n rag-system svc/grafana-service 3000:3000'
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'kubectl port-forward -n rag-system svc/prometheus-service 9090:9090'

Write-Host "✅ RAG Chatbot System is ready!" -ForegroundColor Green
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "• RAG Chatbot: http://localhost:8501" -ForegroundColor White
Write-Host "• Grafana: http://localhost:3000" -ForegroundColor White
Write-Host "• Prometheus: http://localhost:9090" -ForegroundColor White
```

## 📊 Monitoring and Verification

### Verify System Health
```powershell
# Check all resources
kubectl get all -n rag-system

# Check logs
kubectl logs -n rag-system deployment/rag-chatbot
kubectl logs -n rag-system deployment/grafana
kubectl logs -n rag-system deployment/prometheus
```

### Test Metrics
1. Open RAG Chatbot: http://localhost:8501
2. Ask a question to trigger the RAG pipeline
3. Open Prometheus: http://localhost:9090
4. Search for metrics:
   - `rag_requests_total`
   - `rag_retrievals_total`
   - `rag_query_embeddings_total`
   - `rag_response_generations_total`

### Test Grafana Dashboards
1. Open Grafana: http://localhost:3000
2. Login: admin/admin
3. Navigate to dashboards to view RAG system metrics

## 📝 Notes

- Keep port forwarding terminals open while using the system
- The system takes approximately 2-3 minutes to fully start up
- Monitor resource usage with `kubectl top pods -n rag-system`
- For production use, consider using ingress controllers instead of port forwarding

## 🆘 Support

If you encounter issues:
1. Check this troubleshooting section
2. Verify all prerequisites are met
3. Check pod logs for specific error messages
4. Ensure sufficient system resources (memory/CPU)

---
*Last updated: August 3, 2025*
