# Cloud Qdrant Configuration Changes

## Summary of Changes

This document outlines the changes made to support cloud-hosted Qdrant instead of local deployment.

### Files Modified

#### 1. Environment Configuration
- **`.env.example`**: Updated to include `QDRANT_URL` and `QDRANT_API_KEY`
- **`requirements.txt`**: Added `psutil` for system metrics

#### 2. Docker Configuration
- **`docker-compose.yml`**: 
  - Removed local Qdrant service
  - Updated RAG chatbot to use cloud Qdrant environment variables
  - Removed Qdrant data volume

#### 3. Kubernetes Configuration
- **`k8s/configmap.yaml`**: 
  - Added `qdrant-api-key` to secrets
  - Updated `QDRANT_URL` to use cloud URL format
- **`k8s/rag-chatbot.yaml`**: Added `QDRANT_API_KEY` environment variable
- **`k8s/qdrant.yaml`**: Removed (no longer needed)

#### 4. Monitoring Configuration
- **`monitoring/prometheus.yml`**: Removed local Qdrant monitoring
- **`k8s/prometheus.yaml`**: Updated scrape configs to exclude Qdrant

#### 5. Deployment Scripts
- **`deploy.ps1`**: 
  - Removed Qdrant deployment steps
  - Updated secret configuration instructions
- **`deploy.sh`**: Same updates as PowerShell version

#### 6. New Files
- **`setup-cloud-config.ps1`**: PowerShell script for quick configuration
- **`setup-cloud-config.sh`**: Bash script for quick configuration
- **`CLOUD_QDRANT_CHANGES.md`**: This documentation file

#### 7. Documentation
- **`DEPLOYMENT.md`**: Updated to reflect cloud Qdrant usage

### Configuration Required

Before deploying, you need to:

1. **Set up your `.env` file**:
   ```
   OPENAI_API_KEY=your-openai-api-key
   QDRANT_URL=https://your-cluster-url.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-api-key
   ```

2. **Update Kubernetes secrets after deployment**:
   ```bash
   # Use the setup script
   ./setup-cloud-config.sh "openai-key" "qdrant-key" "https://your-cluster.qdrant.io:6333"
   
   # Or manually
   kubectl create secret generic rag-secrets \
     --from-literal=openai-api-key=YOUR_KEY \
     --from-literal=qdrant-api-key=YOUR_QDRANT_KEY \
     -n rag-system --dry-run=client -o yaml | kubectl apply -f -
   ```

### Architecture Changes

The new architecture removes the local Qdrant deployment:

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

### Benefits

1. **Reduced complexity**: No need to manage Qdrant infrastructure
2. **Better reliability**: Cloud-managed Qdrant with built-in redundancy
3. **Simplified deployment**: Fewer components to deploy and monitor
4. **Cost optimization**: Pay only for what you use
5. **Automatic backups**: Cloud provider handles data persistence

### Testing

After deployment, verify the connection:

```bash
# Check pod logs
kubectl logs -f deployment/rag-chatbot -n rag-system

# Test the application
curl http://rag-chatbot.local (or your configured URL)
```
