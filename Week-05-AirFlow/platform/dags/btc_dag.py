from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import yaml
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle

DATA_SOURCE = Path("./DATA")
DATA_TRAINING = DATA_SOURCE / "training" / "btc_data"
ARITIFACTS = DATA_SOURCE / "artifacts"

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

def load_config():
    config_path = "./config/btc_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_temp_dir():
    config = load_config()
    path_save = Path("./btc_tmp_dir") / config['version']
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


def pull_data(version: str, part: str, path_save: str):
    print(f"Pulling data: version: {version}, part: {part}")

    source_path = DATA_TRAINING / version / f"{part}.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Data source not found: {source_path}")

    cmd = f"cp {DATA_TRAINING}/{version}/{part}.csv {path_save}/{part}.csv"
    os.system(cmd)
    if not os.path.exists(f"{path_save}/{part}.csv"):
        raise FileNotFoundError(f"Data pull {part} failed")
    
    print(f"Data pull {part} successfully")


def load_data(version: str, part: str, path_save: str):
    pull_data(version, part, path_save)
    df = pd.read_csv(f"{path_save}/{part}.csv")
    return df

def save_scaler(scaler: MinMaxScaler, path_save: str, name: str):
    with open(f"{path_save}/{name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

def save_model(model: nn.Module, path_save: str):
    torch.save(model.state_dict(), path_save / "model.pth")


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
    config = load_config()
    path_save = Path("./btc_tmp_dir")  / config["version"]

    train_df = load_data(config["version"], "train", path_save)
    val_df = load_data(config["version"], "val", path_save)

    train_df = clean_data(train_df)
    val_df = clean_data(val_df)

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

    X_train, y_train = create_sequences(train_df, lookback=config['lookback'])
    X_val, y_val = create_sequences(val_df, lookback=config['lookback'])

    # Save the processed data
    np.save(path_save/"X_train.npy", X_train)
    np.save(path_save/"y_train.npy", y_train)
    np.save(path_save/"X_val.npy", X_val)
    np.save(path_save/"y_val.npy", y_val)

    print("Data processing complete")


def evaluate_model(loader, model, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred.view(-1), y_batch)
            running_loss += loss.item()
    return running_loss / len(loader)


def train_model():
    config = load_config()
    path_save = Path("./btc_tmp_dir")  / config['version']

    X_train = np.load(path_save/"X_train.npy")
    y_train = np.load(path_save/"y_train.npy")
    X_val = np.load(path_save/"X_val.npy")
    y_val = np.load(path_save/"y_val.npy")

    train_data = BTCData(X_train, y_train)
    val_data = BTCData(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    
    device = config['device']
    input_size = len(features) + 1
    model = RNN_Model(input_size=input_size, 
                      hidden_size=config['hidden_size'], 
                      output_size=1, 
                      num_layers=config['num_layers'])
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
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
        val_loss = evaluate_model(val_loader, model, criterion, device)
        print(f"Epoch {epoch + 1}, train loss: {epoch_loss:.6f}, val loss: {val_loss:.6f}")
    
    print("Training complete")
    save_model(model, path_save)
    

def validate_model():
    config = load_config()
    path_save = Path("./btc_tmp_dir")  / config['version']

    test_df = load_data(config['version'], "test", path_save)
    test_df = clean_data(test_df)

    with open(path_save / "features_scaler.pkl", 'rb') as f:
        features_scaler = pickle.load(f)
    
    with open(path_save / "target_scaler.pkl", 'rb') as f:
        target_scaler = pickle.load(f)
    
    test_df[features] = features_scaler.transform(test_df[features])
    test_df[target] = target_scaler.transform(test_df[[target]])
    X_test, y_test = create_sequences(test_df)

    test_data = BTCData(X_test, y_test)    
    test_loader = DataLoader(test_data, batch_size=config['batch_size'])
    input_size = len(features) + 1
    model = RNN_Model(input_size=input_size, 
                      hidden_size=config['hidden_size'], 
                      output_size=1, 
                      num_layers=config['num_layers'])
    model_state_dict = torch.load(path_save / "model.pth", weights_only=True)
    model.load_state_dict(model_state_dict)

    criterion = nn.MSELoss()
    device = config['device']
    test_loss = evaluate_model(test_loader, model, criterion, device)
    print(f"Test Loss: {test_loss}")


def logging_artifacts():
    config = load_config()
    path_save = Path("./btc_tmp_dir")  / config["version"]
    artifacts = ARITIFACTS / "btc_model" / config["version"]
    artifacts.mkdir(parents=True, exist_ok=True)

    # Log the scalers
    cmd_0 = f"cp {path_save}/features_scaler.pkl {artifacts}/features_scaler.pkl"
    cmd_1 = f"cp {path_save}/target_scaler.pkl {artifacts}/target_scaler.pkl"

    # Log the model
    cmd_2 = f"cp {path_save}/model.pth {artifacts}/model.pth" 

    # Log config
    cmd_3 = f"cp ./config/btc_config.yaml {artifacts}/btc_config.yaml"

    for cmd in [cmd_0, cmd_1, cmd_2, cmd_3]:
        os.system(cmd)
        if not os.path.exists(f"{artifacts}/features_scaler.pkl"):
            raise FileNotFoundError(f"Failed to log artifact: {cmd}")


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

    logging_artifacts_task = PythonOperator(
        task_id='logging_artifacts',
        python_callable=logging_artifacts,
        dag=dag,
    )
    create_temp_dir_task >> data_processing_task >> train_model_task >> validate_model_task >> logging_artifacts_task
