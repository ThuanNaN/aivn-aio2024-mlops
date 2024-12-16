from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


DATA_SOURCE = Path("./DATA")
DATA_TRAINING = DATA_SOURCE / "training" / "btc_data"

features = ['Open', 'High', 'Low', 'Volume']
target = 'Price'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_name: str | None=None):
    config_path = "./config/btc_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config filse not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config_name:
            config = config.get(config_name, None)
            if not config:
                raise ValueError(f"Config not found: {config_name}")
    return config


def connect_mlflow(mlflow_config):
    MLFLOW_TRACKING_URI = mlflow_config['tracking_uri']
    MLFLOW_EXPERIMENT_NAME = mlflow_config['experiment_name']
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
        print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
        print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        print(f"Error: {e}")
        raise e

def create_temp_dir():
    data_config = load_config("data_config")
    path_save = Path("./btc_tmp_dir") / data_config['data_version']
    path_save.mkdir(parents=True, exist_ok=True)
    if not path_save.exists():
        raise FileNotFoundError(f"Failed to create directory: {path_save}")
    

class BTCData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)
        x = self.fc1(out[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.fc1(out[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        return x


def pull_data(data_version: str, part: str, path_save: str):
    print(f"Pulling data version: {data_version}, part: {part}")

    source_path = DATA_TRAINING / data_version / f"{part}.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Data source not found: {source_path}")

    cmd = f"cp {DATA_TRAINING}/{data_version}/{part}.csv {path_save}/{part}.csv"
    os.system(cmd)
    if not os.path.exists(f"{path_save}/{part}.csv"):
        raise FileNotFoundError(f"Data pull {part} failed")
    
    print(f"Data pull {part} successfully")


def load_data(data_version: str, part: str, path_save: str):
    pull_data(data_version, part, path_save)
    df = pd.read_csv(f"{path_save}/{part}.csv")
    return df

def save_scaler(scaler: MinMaxScaler, path_save: str, name: str):
    with open(f"{path_save}/{name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)


def clean_data(df: pd.DataFrame):
    df = df.drop(columns=['Change %'])
    df.rename(columns={'Vol.': 'Volume'}, inplace=True)

    df.sort_values('Date', inplace=True)
    df['Volume'] = (
        df['Volume']
        .str.replace('K', 'e3')
        .str.replace('M', 'e6')
        .str.replace('B', 'e9')
        .astype(float)
    )
    columns_to_clean = ['Price', 'Open', 'High', 'Low']
    for col in columns_to_clean:
        df[col] = df[col].str.replace(',', '').astype(float)

    df = df.drop(columns=['Date'])
    return df


def create_sequences(data: pd.DataFrame, 
                     lookback: int=14
                     )-> tuple:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data.iloc[i-lookback:i].values)
        y.append(data[target][i])
    return np.array(X), np.array(y)


def data_processing():
    data_config = load_config("data_config")
    path_save = Path("./btc_tmp_dir")  / data_config["data_version"]

    train_df = load_data(data_config["data_version"], "train", path_save)
    val_df = load_data(data_config["data_version"], "val", path_save)
    test_df = load_data(data_config["data_version"], "test", path_save)

    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)

    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform the training data
    train_df[features] = features_scaler.fit_transform(train_df[features])
    train_df[target] = target_scaler.fit_transform(train_df[[target]])
    save_scaler(features_scaler, path_save, "features")
    save_scaler(target_scaler, path_save, "target")
    
    # Transform the validation and test data
    val_df[features] = features_scaler.transform(val_df[features])
    val_df[target] = target_scaler.transform(val_df[[target]])

    test_df[features] = features_scaler.transform(test_df[features])
    test_df[target] = target_scaler.transform(test_df[[target]])

    X_train, y_train = create_sequences(train_df, lookback=data_config['lookback'])
    X_val, y_val = create_sequences(val_df, lookback=data_config['lookback'])
    X_test, y_test = create_sequences(test_df, lookback=data_config['lookback'])

    # Save the processed data
    np.save(path_save/"X_train.npy", X_train)
    np.save(path_save/"y_train.npy", y_train)
    np.save(path_save/"X_val.npy", X_val)
    np.save(path_save/"y_val.npy", y_val)
    np.save(path_save/"X_test.npy", X_test)
    np.save(path_save/"y_test.npy", y_test)

    print("Data processing complete")


def evaluate_model(loader, model, criterion, device):
    model.to(device)
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred.view(-1), y_batch)
            running_loss += loss.item()
    return running_loss / len(loader)


def train_model(**kwargs):
    config = load_config()
    data_config = config["data_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    mlflow_config = config["mlflow_config"]
    connect_mlflow(mlflow_config)

    seed_everything(train_config['seed'])

    path_save = Path("./btc_tmp_dir")  / data_config['data_version']

    X_train = np.load(path_save/"X_train.npy")
    y_train = np.load(path_save/"y_train.npy")
    X_val = np.load(path_save/"X_val.npy")
    y_val = np.load(path_save/"y_val.npy")
    X_test = np.load(path_save/"X_test.npy")
    y_test = np.load(path_save/"y_test.npy")

    train_data = BTCData(X_train, y_train)
    val_data = BTCData(X_val, y_val)
    test_data = BTCData(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=train_config['batch_size'])
    
    device = train_config['device']
    input_size = model_config['input_size']
    output_size = model_config['output_size']

    if model_config['model_name'] == 'rnn':
        model = RNN_Model(input_size=input_size, 
                        hidden_size=model_config['hidden_size'], 
                        output_size=output_size,
                        num_layers=model_config['num_layers'])
    elif model_config['model_name'] == 'lstm':
        model = LSTM_Model(input_size=input_size, 
                        hidden_size=model_config['hidden_size'], 
                        output_size=output_size, 
                        num_layers=model_config['num_layers'])
    else:
        raise ValueError(f"Model not supported: {model_config['model_name']}")

    model.to(device)
    criterion = nn.MSELoss()

    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=train_config['lr'], 
                                     weight_decay=train_config['weight_decay'])
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=train_config['lr'], 
                                    weight_decay=train_config['weight_decay'])
    else:
        raise ValueError(f"Optimizer not supported: {train_config['optimizer']}")

    with mlflow.start_run(run_name=train_config["run_name"]) as run:
        print(f"MLFLOW run_id: {run.info.run_id}")
        print(f"MLFLOW experiment_id: {run.info.experiment_id}")
        print(f"MLFLOW run_name: {train_config["run_name"]}")

        mlflow.set_tags({
            "data_version": data_config['data_version'],
            "model_name": model_config['model_name'],
        })
        mlflow.log_params({
            # data_config
            "lookback": data_config['lookback'],

            # model_config
            "input_size": model_config['input_size'],
            "output_size": model_config['output_size'],
            "hidden_size": model_config['hidden_size'],
            "num_layers": model_config['num_layers'],

            # train_config
            "seed": train_config['seed'],
            "epochs": train_config['epochs'],
            "batch_size": train_config['batch_size'],
            "lr": train_config['lr'],
            "weight_decay": train_config['weight_decay'],
            "optimizer": train_config['optimizer'],
            "best_model_metric": train_config['best_model_metric'],
            "best_deploy_metric": train_config['best_deploy_metric'],
            "device": train_config['device'],
        })

        best_model_state_dict = None
        best_loss = np.inf
        for epoch in range(train_config['epochs']):
            model.train()
            running_loss = 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.view(-1), y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(train_loader)
            mlflow.log_metric("training_mse_loss", f"{epoch_loss:6f}", step=epoch)

            val_loss = evaluate_model(val_loader, model, criterion, device)
            mlflow.log_metric("val_mse_loss", f"{val_loss:.6f}", step=epoch)

            if train_config["best_model_metric"] == "val_mse_loss":
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state_dict = model.state_dict()
            else: 
                raise ValueError(f"Best model metric not supported: {train_config['best_model_metric']}")

            print(f"Epoch {epoch + 1}, train loss: {epoch_loss:.6f}, val loss: {val_loss:.6f}")
        
        model.load_state_dict(best_model_state_dict)

        # Save to cloud/local
        torch.save(best_model_state_dict, path_save / "model.pth")
        local_artifacts = DATA_SOURCE / "artifacts" / data_config['data_version'] / run.info.run_id
        local_artifacts.mkdir(parents=True, exist_ok=True)
        cp_cmd = f"cp {path_save}/model.pth {local_artifacts}/model.pth" 
        os.system(cp_cmd)

        # Test the model
        test_loss = evaluate_model(test_loader, model, criterion, device)
        mlflow.log_metric("test_mse_loss", f"{test_loss:.6f}")
        print(f"Test Loss: {test_loss}")

        # Save the model and scalers to mlflow
        input_example = torch.rand(1, data_config['lookback'], input_size)
        signature = infer_signature(input_example.numpy(), model(input_example).detach().numpy())
        mlflow.pytorch.log_model(model, 
                                 artifact_path="pytorch-model", 
                                 pip_requirements="./requirements.txt", 
                                 signature=signature)

        mlflow.log_artifact(path_save / "features_scaler.pkl", artifact_path="scalers")
        mlflow.log_artifact(path_save / "target_scaler.pkl", artifact_path="scalers")
        mlflow.log_artifact("./config/btc_config.yaml", artifact_path="config")

        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")

        # Pass uri
        kwargs['ti'].xcom_push(key="run_id", value=run.info.run_id)
        kwargs['ti'].xcom_push(key="test_mse_loss", value=test_loss)
        print("Training complete")


def registered_model(client, registered_name: str, model_alias: str, run_id: str):
    try:
        print(f"Registering model: {registered_name}")
        client.create_registered_model(registered_name)
        client.get_registered_model(model_alias)
    except:
        print(f"Model: {registered_name} already exists")

    print(f"Creating model version: {model_alias}")
    model_uri = f"runs:/{run_id}/pytorch-model"
    mv = client.create_model_version(registered_name, model_uri, run_id)

    print(f"Creating model alias: {model_alias}")
    client.set_registered_model_alias(name=registered_name,
                                        alias=model_alias,
                                        version=mv.version)
    print("--Model Version--")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))
    print("Aliases: {}".format(mv.aliases))


def validate_model(**kwargs):
    run_id = kwargs['ti'].xcom_pull(task_ids='train_model', key='run_id')
    test_mse_loss = kwargs['ti'].xcom_pull(task_ids='train_model', key='test_mse_loss')

    mlflow_config = load_config("mlflow_config")
    connect_mlflow(mlflow_config)

    client = MlflowClient()
    registered_name = mlflow_config['registered_name']
    model_alias = mlflow_config['model_alias']
    try:
        alias_mv = client.get_model_version_by_alias(registered_name, model_alias)
        print(f"Alias: {model_alias} found")
    except:
        print(f"Alias: {model_alias} not found")
        registered_model(client, registered_name, model_alias, run_id)

    else:
        print(f"Retrieving run: {alias_mv.run_id}")
        prod_metric = mlflow.get_run(alias_mv.run_id).data.metrics
        prod_test_mse_loss = prod_metric['test_mse_loss']

        # Check best loss
        if prod_test_mse_loss < test_mse_loss:
            print(f"Current model is better: {prod_test_mse_loss}")
        else:
            registered_model(client, registered_name, model_alias, run_id)


with DAG(
    'BTC_Price_Prediction',
    default_args=default_args,
    description='A simple ML pipeline demonstration',
    schedule_interval=timedelta(days=1),
    ) as dag:

    create_temp_dir_task = PythonOperator(
        task_id='create_temp_dir',
        python_callable=create_temp_dir,
        dag=dag,
    )

    data_processing_task = PythonOperator(
        task_id='data_processing',
        python_callable=data_processing,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
    )

    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        dag=dag,
    )

    create_temp_dir_task >> \
    data_processing_task >> \
    train_model_task >> \
    validate_model_task

