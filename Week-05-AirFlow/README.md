# Apache Airflow for Scheduler

## 1. Introduction

## 2. Setup

- Docker
- Docker Compose
- Python >= 3.9.11

### 2.1 Platform

See the [Platform README](./platform/README.md) for more information.

### 2.2 Backend

```bash
cd ./backend

fastapi dev --host 0.0.0.0 --port 8000 --reload
```

### 2.3 Frontend

```bash
cd ./frontend

python app/main.py
```
