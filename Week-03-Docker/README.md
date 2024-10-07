# Dockerize your application

## Description

In this exercise, you will containerize your application using Docker. First, you will build a Docker image for your backend server and then you will build a Docker image for your frontend application. Finally, you will run both containers and test your application.

## Installation

- [Docker Engine](https://docs.docker.com/engine/install)

Optional:

- [Docker Desktop](https://docs.docker.com/desktop/)

## 1. Create the Docker Network for your application

- Create a Docker network with the name `aio_mlops_net`

```bash
docker network create aio_mlops_net
```

## 1. Dockerize your backend server

- Go to backend directory

```bash
cd backend
```

- Build the Docker image based on the Dockerfile with the name of image is `fastapi_be`

```bash
docker build -t fastapi_be .
```

- Run the Docker container with the name of container is `backend_cont` and expose the port 8000

```bash
docker run --rm -it --name backend_cont --network aio_mlops_net -p 8000:8000 fastapi_be
```

## 2. Dockerize your frontend application

- Open new terminal and go to frontend directory

```bash
cd frontend
```

- Build the Docker image based on the Dockerfile with the name of image is `gradio_fe`

```bash
docker build -t gradio_fe .
```

- Run the Docker container with the name of container is `frontend_cont` and expose the port 3000

- Set the environment variable `BACKEND_URL` by `-e BACKEND_URL="http://backend_cont:8000"` to be able to communicate with the backend server

```bash
docker run --rm -it --name frontend_cont --network aio_mlops_net -e BACKEND_URL="http://backend_cont:8000" -p 3000:3000 gradio_fe
```
