# Platform

## DAGS

Deploy the dags to the airflow server by running the following commands:

```bash
cd dags
chmod +x ./deploy.sh
./deploy.sh
```

## Airflow

To start the airflow server, worker,... run the following commands:

```bash
cd airflow

# Initialize the folders
mkdir -p ./dags ./logs ./plugins ./config

# Set the UID of the user to the .env file
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Start the services
docker compose up -d --build
```

Go to `http://localhost:8080` to access the airflow UI.

- Username: `airflow`

- Password: `airflow`

## MLFlow

To start the mlflow server, run the following commands:

```bash
cd mlflow

# Start the services
docker compose up -d --build
```
