#!/bin/bash

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/yolo-backend

# Get the URLs to access the services
echo "API service URL:"
minikube service yolo-backend --url

echo "Deployment complete!"