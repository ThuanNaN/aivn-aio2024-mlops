# Week 2: Client Server Deployment

In this week, we will re-deploy previous applications using but with a different approach. We will deploy the applications using the client-server architecture.

- **The backend** server will be built using the FastAPI package.

- **The frontend** application will still use the Gradio package but the prediction process will be done on the backend server.

## 1. Backend Server

- Go to the backend directory

```bash
cd backend
```

- Install the required packages using conda virtual environment

```bash
conda create -n aio-mlops-w2-be python=3.9.11 --y
conda activate aio-mlops-w2-be
pip install -r requirements.txt
```

- Run the backend server using FastAPI CLI

```bash
fastapi run --host 0.0.0.0 --port 8000
```

## 2. Frontend UI

- Open new terminal and go to frontend directory

```bash
cd frontend
```

- Install the required packages using conda virtual environment

```bash
conda create -n aio-mlops-w2-fe python=3.9.11 --y
conda activate aio-mlops-w2-fe
pip install -r requirements.txt
```

- Run the frontend application

```bash
python app.py
```
