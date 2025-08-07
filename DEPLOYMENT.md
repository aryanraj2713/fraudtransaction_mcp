# 🚀 Deployment Guide

This guide covers deploying the Agentic AI Fraud Detection System in various environments.

## 📋 Prerequisites

### Required
- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key

### Optional
- Kubernetes cluster
- Logfire account for monitoring
- Load balancer (for production)

## 🏠 Local Development

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd fraud-detection-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env with your credentials
OPENAI_API_KEY=your_openai_api_key_here
LOGFIRE_TOKEN=your_logfire_token_here  # Optional
FRAUD_THRESHOLD=0.7
```

### 3. Run Application
```bash
# Start web interface
python -m uvicorn src.web_interface:app --host 0.0.0.0 --port 8000

# Access at: http://localhost:8000
```

### 4. Verify Installation
```bash
# Run tests
python test_system.py

# Run demo
python simple_demo.py

# Check API
curl http://localhost:8000/system-status
```

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -d \
  --name fraud-detection \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your_key" \
  -e LOGFIRE_TOKEN="your_token" \
  fraud-detection
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables for Docker
```yaml
# docker-compose.yml environment section
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - LOGFIRE_TOKEN=${LOGFIRE_TOKEN}
  - FRAUD_THRESHOLD=0.7
  - MODEL_UPDATE_INTERVAL=3600
  - VECTOR_DB_SIZE=10000
```

## ☸️ Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Configure Secrets
```bash
# Create secret for API keys
kubectl create secret generic fraud-detection-secrets \
  --from-literal=openai-api-key="your_openai_key" \
  --from-literal=logfire-token="your_logfire_token" \
  -n fraud-detection
```

### 3. Deploy Application
```bash
# Deploy all resources
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n fraud-detection
kubectl get services -n fraud-detection
```

### 4. Access Application
```bash
# Port forward for testing
kubectl port-forward service/fraud-detection-service 8000:8000 -n fraud-detection

# Or use LoadBalancer/Ingress for production access
```

### 5. Monitor Deployment
```bash
# View logs
kubectl logs -f deployment/fraud-detection -n fraud-detection

# Check resource usage
kubectl top pods -n fraud-detection
```

## 🌐 Production Deployment

### Infrastructure Requirements
- **CPU**: 2+ cores per instance
- **Memory**: 4GB+ RAM per instance
- **Storage**: 10GB+ for logs and patterns
- **Network**: Load balancer with SSL termination

### High Availability Setup
```yaml
# k8s/deployment.yaml - production settings
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection
spec:
  replicas: 3  # Multiple instances
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: fraud-detection
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        readinessProbe:
          httpGet:
            path: /system-status
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /system-status
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

### Load Balancer Configuration
```nginx
# nginx.conf example
upstream fraud_detection {
    server fraud-detection-1:8000;
    server fraud-detection-2:8000;
    server fraud-detection-3:8000;
}

server {
    listen 80;
    server_name fraud-detection.yourdomain.com;
    
    location / {
        proxy_pass http://fraud_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        proxy_pass http://fraud_detection/system-status;
    }
}
```

## 📊 Monitoring & Observability

### Logfire Setup
1. **Create Account**: https://logfire.pydantic.dev/
2. **Get Token**: Copy your project token
3. **Configure**: Add `LOGFIRE_TOKEN` to environment
4. **Verify**: Check logs appear in dashboard

### Health Checks
```bash
# Application health
curl http://localhost:8000/system-status

# Docker health
docker ps --filter "name=fraud-detection"

# Kubernetes health
kubectl get pods -n fraud-detection
```

### Metrics to Monitor
- **Response Time**: < 100ms target
- **Throughput**: 10K+ TPS capacity
- **Error Rate**: < 1% target
- **Memory Usage**: Monitor for leaks
- **CPU Usage**: Scale based on load

## 🔒 Security Considerations

### API Keys
```bash
# Use secrets management
kubectl create secret generic api-secrets \
  --from-literal=openai-key="$(cat openai-key.txt)" \
  --from-literal=logfire-token="$(cat logfire-token.txt)"
```

### Network Security
- Use HTTPS in production
- Implement rate limiting
- Add API authentication if needed
- Restrict network access to necessary ports

### Data Privacy
- No PII stored in logs
- Transaction data processed in memory only
- Secure API key storage
- Audit trail logging

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Check virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

**2. API Key Issues**
```bash
# Solution: Verify environment variables
echo $OPENAI_API_KEY
cat .env | grep OPENAI
```

**3. Port Already in Use**
```bash
# Solution: Kill existing process
lsof -ti:8000 | xargs kill -9
```

**4. Docker Build Fails**
```bash
# Solution: Clear Docker cache
docker system prune -a
docker build --no-cache -t fraud-detection .
```

**5. Kubernetes Pod Crashes**
```bash
# Solution: Check logs and resources
kubectl logs -f pod-name -n fraud-detection
kubectl describe pod pod-name -n fraud-detection
```

### Performance Issues

**High Memory Usage**
- Reduce `VECTOR_DB_SIZE` in environment
- Implement memory limits in Kubernetes
- Monitor for memory leaks

**Slow Response Times**
- Check OpenAI API latency
- Scale horizontally with more replicas
- Optimize vector database queries

**High CPU Usage**
- Increase CPU limits
- Optimize pattern matching algorithms
- Add more worker processes

## 📈 Scaling

### Horizontal Scaling
```bash
# Kubernetes scaling
kubectl scale deployment fraud-detection --replicas=5 -n fraud-detection

# Docker Compose scaling
docker-compose up --scale fraud-detection=3 -d
```

### Auto-scaling
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🚀 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy Fraud Detection
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker Image
      run: |
        docker build -t fraud-detection:${{ github.sha }} .
        
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/fraud-detection \
          fraud-detection=fraud-detection:${{ github.sha }} \
          -n fraud-detection
```

## 📞 Support

For deployment issues:
1. Check logs first: `kubectl logs -f deployment/fraud-detection`
2. Verify configuration: `kubectl describe deployment fraud-detection`
3. Test connectivity: `curl http://service-url/system-status`
4. Monitor resources: `kubectl top pods`

---

**🎯 Production Checklist:**
- [ ] Environment variables configured
- [ ] API keys securely stored
- [ ] Health checks enabled
- [ ] Monitoring configured
- [ ] Load balancer setup
- [ ] SSL certificates installed
- [ ] Backup strategy defined
- [ ] Scaling policies configured