import pandas as pd
import torch
from torch import nn
import numpy as np

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


async def predict_futures(btc_models, input_data: pd.DataFrame, next_days=1) -> list:
    try:
        model = btc_models["model"].eval()
        features = btc_models["deploy_config"].features
        target = btc_models["deploy_config"].target

        input_data = clean_data(input_data)
        input_data.loc[:, features] = btc_models["features_scaler"].transform(input_data[features])
        input_data.loc[:, target] = btc_models["target_scaler"].transform(input_data[[target]])
        X_input = torch.tensor(input_data.values, dtype=torch.float32).unsqueeze(0)

        predictions = []
        for _ in range(next_days):
            with torch.no_grad():
                y_pred = model(X_input)
            y_prediction = y_pred.item()
            predictions.append(y_prediction)
            
            last_row = input_data.iloc[-1].values.tolist()
            next_input = [y_prediction] + last_row[:-1]
            next_input = torch.tensor([next_input], dtype=torch.float32).unsqueeze(1)
            X_input = torch.cat([X_input[:, 1:, :], next_input], dim=1)
        
        unscaled_predictions = btc_models["target_scaler"].inverse_transform(np.array(predictions).reshape(-1, 1))
        return unscaled_predictions.reshape(-1).tolist()
        
    except Exception as e:
        print(str(e))
        return Exception("An error occurred while predicting the future price of Bitcoin.")
