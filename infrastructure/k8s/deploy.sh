#!/bin/bash

# Project Argus Kubernetes Deployment Script
set -e

NAMESPACE="project-argus"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting Project Argus Kubernetes deployment..."

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl is not installed or not in PATH"
        exit 1
    fi
    echo "✅ kubectl is available"
}

# Function to check if cluster is accessible
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    echo "✅ Kubernetes cluster is accessible"
}

# Function to apply Kubernetes manifests
apply_manifests() {
    echo "📦 Applying Kubernetes manifests..."
    
    # Apply in order of dependencies
    kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
    kubectl apply -f "${SCRIPT_DIR}/configmap.yaml"
    kubectl apply -f "${SCRIPT_DIR}/secrets.yaml"
    kubectl apply -f "${SCRIPT_DIR}/storage.yaml"
    
    # Wait for storage to be ready
    echo "⏳ Waiting for storage to be ready..."
    kubectl wait --for=condition=Bound pvc/postgres-pvc -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Bound pvc/redis-pvc -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Bound pvc/minio-pvc -n ${NAMESPACE} --timeout=300s
    
    # Apply database and storage services
    kubectl apply -f "${SCRIPT_DIR}/database.yaml"
    kubectl apply -f "${SCRIPT_DIR}/redis.yaml"
    kubectl apply -f "${SCRIPT_DIR}/minio.yaml"
    
    # Wait for database services to be ready
    echo "⏳ Waiting for database services to be ready..."
    kubectl wait --for=condition=available deployment/postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/redis -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/minio -n ${NAMESPACE} --timeout=300s
    
    # Apply backend services
    kubectl apply -f "${SCRIPT_DIR}/services.yaml"
    kubectl apply -f "${SCRIPT_DIR}/microservices.yaml"
    
    # Wait for backend services to be ready
    echo "⏳ Waiting for backend services to be ready..."
    kubectl wait --for=condition=available deployment/auth-service -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/security-service -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/alert-service -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/tracking-service -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/evidence-service -n ${NAMESPACE} --timeout=300s
    
    # Apply API Gateway
    kubectl apply -f "${SCRIPT_DIR}/api-gateway.yaml"
    kubectl wait --for=condition=available deployment/api-gateway -n ${NAMESPACE} --timeout=300s
    
    # Apply frontend
    kubectl apply -f "${SCRIPT_DIR}/dashboard.yaml"
    kubectl wait --for=condition=available deployment/dashboard -n ${NAMESPACE} --timeout=300s
    
    # Apply ingress
    kubectl apply -f "${SCRIPT_DIR}/ingress.yaml"
    
    echo "✅ All manifests applied successfully"
}

# Function to check deployment status
check_deployment() {
    echo "🔍 Checking deployment status..."
    
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE}
    
    echo -e "\nServices:"
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\nIngress:"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\nPersistent Volume Claims:"
    kubectl get pvc -n ${NAMESPACE}
}

# Function to get service URLs
get_urls() {
    echo "🌐 Service URLs:"
    
    INGRESS_IP=$(kubectl get ingress argus-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$INGRESS_IP" != "pending" ] && [ -n "$INGRESS_IP" ]; then
        echo "Dashboard: https://dashboard.argus.local"
        echo "API: https://api.argus.local/api"
        echo "Auth: https://api.argus.local/auth"
        echo ""
        echo "Add the following to your /etc/hosts file:"
        echo "$INGRESS_IP dashboard.argus.local"
        echo "$INGRESS_IP api.argus.local"
    else
        echo "Ingress IP is still pending. Run 'kubectl get ingress -n ${NAMESPACE}' to check status."
    fi
}

# Function to clean up deployment
cleanup() {
    echo "🧹 Cleaning up Project Argus deployment..."
    
    kubectl delete -f "${SCRIPT_DIR}/ingress.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/dashboard.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/api-gateway.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/microservices.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/services.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/minio.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/redis.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/database.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/storage.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/secrets.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/configmap.yaml" --ignore-not-found=true
    kubectl delete -f "${SCRIPT_DIR}/namespace.yaml" --ignore-not-found=true
    
    echo "✅ Cleanup completed"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_kubectl
        check_cluster
        apply_manifests
        check_deployment
        get_urls
        echo "🎉 Project Argus deployment completed successfully!"
        ;;
    "status")
        check_kubectl
        check_cluster
        check_deployment
        get_urls
        ;;
    "cleanup")
        check_kubectl
        check_cluster
        cleanup
        ;;
    "help")
        echo "Usage: $0 [deploy|status|cleanup|help]"
        echo "  deploy  - Deploy Project Argus to Kubernetes (default)"
        echo "  status  - Check deployment status"
        echo "  cleanup - Remove Project Argus from Kubernetes"
        echo "  help    - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac