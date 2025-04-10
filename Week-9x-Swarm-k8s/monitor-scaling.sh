#!/bin/bash

# Enable metrics server if not already enabled
if ! kubectl get deployment metrics-server -n kube-system &>/dev/null; then
  echo "Enabling metrics-server addon..."
  minikube addons enable metrics-server
  
  echo "Waiting for metrics-server to be ready..."
  kubectl wait --for=condition=available --timeout=300s deployment/metrics-server -n kube-system
fi

# Function to display HPA and pod status
monitor() {
  clear
  echo "==== HPA Status ===="
  kubectl get hpa yolo-backend-hpa -o wide
  
  echo -e "\n==== Pods Status ===="
  kubectl get pods -l app=yolo-backend
  
  echo -e "\n==== CPU Usage ===="
  kubectl top pods -l app=yolo-backend
}

echo "Monitoring scaling activity (refreshes every 5 seconds). Press Ctrl+C to stop."
while true; do
  monitor
  sleep 5
done
