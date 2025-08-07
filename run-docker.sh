#!/bin/bash

# Build and run the Fraud Detection System in Docker
echo "🚀 Building Fraud Detection System Docker Image..."

# Build the image
docker build --target production -t fraud-detection-web . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✅ Build complete!"
echo "🌐 Starting web interface on http://localhost:8000"

# Run the container
docker run -it --rm \
    --name fraud-detection-test \
    -p 8000:8000 \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -e LOGFIRE_TOKEN="${LOGFIRE_TOKEN}" \
    -e FRAUD_THRESHOLD=0.7 \
    -e MODEL_UPDATE_INTERVAL=3600 \
    -e VECTOR_DB_SIZE=10000 \
    fraud-detection-web

echo "🛑 Container stopped"