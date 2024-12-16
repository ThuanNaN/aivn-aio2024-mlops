#!/bin/bash

cd ./platform/mlflow
docker compose up -d --build

cd ./platform/airflow
docker compose up -d --build

cd ./backend 
docker compose up -d --build

cd ./frontend
docker compose up -d --build
