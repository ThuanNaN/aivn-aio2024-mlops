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
docker-compose up -d --build
```

Go to `http://localhost:8080` to access the airflow UI.

- Username: `airflow`

- Password: `airflow`
