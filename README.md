# AIO2024 MLOps Course

## Introduction

## 1. Basic Deployment

In this first week, we build a simple application (web application) using 2 python packages: Streamlit and Gradio for serving 2 our models.

See the [Week 1: Basic Deployment](Week-01-Basic-Deployment/README.md) for more details.

## 2. Client Server Deployment

In this second week, we will re-deploy previous applications using but separate the frontend and backend applications. We use FastAPI for the backend server and Gradio for the frontend application.

See the [Week 2: Client Server Deployment](Week-02-Client-Server-Deployment/README.md) for more details.

## 3. Docker

In this exercise, you will containerize your application using Docker.

- First, cretae a Docker network to connect the backend and frontend containers.
- Build and run the Docker container for the backend server.
- Build and run the Docker container for the frontend application.

See the [Week 3: Docker](Week-03-Docker/README.md) for more details.

## 4. AWS Services

In this week, we will explore AWS services to deploy our application.

See the [Week 4: AWS Services](Week-04-AWS/README.md) for more details.

## 5. Airflow

In this week, we build a simple pipeline using Apache Airflow. We will create a DAG to schedule the training and testing of our models.

### Dataset

- `BTC Price` collected from 2016 to 2024 and split into 3 versions:
  - `v0.1`: 2016 - 2022
  - `v0.2`: 2016 - 2023
  - `v0.3`: 2016 - 2024

- `Gold Price` (Optional) collected from 2016 to 2024.

### Lifecycle

- First we will train the model using the `v0.1` dataset, taging the model as `V1`.
- Then we will test the `V1` model using the data in 2/2024.
- Next, we will retrain the model using the `v0.2` dataset, taging the model as `V2`.
- Finally, we will test the `V2` model using the data in 2/2024.

See the [Week 5: Airflow](Week-05-AirFlow/README.md) for more details.

## 6. MLflow
