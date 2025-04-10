# Kubernetes (k8s) Overview

## Getting Started

1. **Install Kubernetes**: Ensure you have a Kubernetes cluster set up (e.g., Minikube, Kind, or a cloud provider).

    Start a local cluster using Minikube:

    ```bash
    minikube start --nodes 4 --driver=docker
    ```

    Start dashboard:

    ```bash
    minikube dashboard
    ```

    Add metrics server:

    ```bash
    minikube addons enable metrics-server
    ```

2. **Apply Manifests**:

    ```bash
    kubectl apply -f k8s/
    ```

3. **Verify Deployment**:

    ```bash
    kubectl get nodes
    kubectl get deployments
    kubectl get pods
    kubectl get services
    ```

    Execute into the backend pod:

    ```bash
    kubectl exec -it yolo-backend-<pod-id> -- /bin/bash
    ```

    Check the logs of the backend pod:

    ```bash
    kubectl logs yolo-backend-<pod-id>
    ```

4. **Optional**: Create tunnel if using Minikube with Docker driver in MacOS:

    ```bash
    minikube service yolo-backend --url
    ```

5. **Clean Up**: When done, delete the Minikube cluster:

    ```bash
    kubectl delete -f k8s/
    minikube stop
    minikube delete
    ```
