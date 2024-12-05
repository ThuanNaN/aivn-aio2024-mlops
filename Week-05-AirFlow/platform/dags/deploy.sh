#!/bin/bash

SOURCE_DIR="."
AIRFLOW_DIR="../airflow/"
DAGS_DIR="../airflow/dags/"
CONFIG_DIR="../airflow/config/"

# Sync .py files to dags folder
rsync -av --progress --include='*.py' --exclude='*' "$SOURCE_DIR" "$DAGS_DIR"

# Sync .yaml files to config folder
rsync -av --progress --include='*.yaml' --exclude='*' "$SOURCE_DIR" "$CONFIG_DIR"

# Sync requirements.txt to config folder
rsync -av --progress --include='requirements.txt' --exclude='*' "$SOURCE_DIR" "$AIRFLOW_DIR"