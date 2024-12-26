#!/bin/bash

# Start all Docker Compose services
start_services() {
    echo "Starting services..."

    # Start mlflow
    cd ./platform/mlflow
    docker compose up -d --build

    # Start airflow
    cd ../airflow
    docker compose up -d --build

    # Start jenkins
    cd ../jenkins
    docker compose up -d --build

    # Start monitoring
    cd ../monitor
    docker compose up -d --build

    # Start backend
    cd ../../backend
    docker compose up -d --build

    # Start frontend
    cd ../frontend
    docker compose up -d --build

    echo "All services have been started."
}

# Stop all Docker Compose services
stop_services() {
    echo "Shutting down services..."

    # Stop frontend
    cd ./frontend
    docker compose down

    # Stop backend
    cd ../backend
    docker compose down

    # Stop airflow
    cd ../platform/airflow
    docker compose down

    # Stop mlflow
    cd ../mlflow
    docker compose down

    # Stop jenkins
    cd ../jenkins
    docker compose down

    # Stop monitoring
    cd ../monitor
    docker compose down

    echo "All services have been shut down."
}

# Check the argument to decide whether to start or stop services
if [ "$1" == "start" ]; then
    start_services
elif [ "$1" == "stop" ]; then
    stop_services
else
    echo "Usage: $0 {start|stop}"
    exit 1
fi
