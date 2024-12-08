FROM apache/airflow:2.10.3

USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

USER airflow
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt