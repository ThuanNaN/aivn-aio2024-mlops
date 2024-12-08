import pandas as pd

def clean_data(df: pd.DataFrame):
    df.rename(columns={'Change %': 'Change'}, inplace=True)
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

    df['Change'] = df['Change'].str.replace('%', '').astype(float) / 100
    return df