#!/bin/bash

cd /mnt/nvme/Repository/aivn-aio2024-mlops/Week-07-Monitoring/platform/mlflow
docker compose up -d --build

cd /mnt/nvme/Repository/aivn-aio2024-mlops/Week-07-Monitoring/platform/airflow
docker compose up -d --build

cd /mnt/nvme/Repository/aivn-aio2024-mlops/Week-07-Monitoring/backend 
docker compose up -d --build

cd /mnt/nvme/Repository/aivn-aio2024-mlops/Week-07-Monitoring/frontend
docker compose up -d --build
