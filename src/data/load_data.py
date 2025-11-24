import yfinance as yf
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils import get_project_root

def create_csv(file: Path) -> pd.DataFrame:
    data = yf.download("SPY", start="2000-01-01", end='2025-11-01',  auto_adjust=False)
    if (data is not None):

        data.columns = data.columns.droplevel('Ticker')
        data = data.reset_index() 
        df =  pd.DataFrame({
            "Date": data["Date"],
            "Close/Last": data["Close"],
            "Volume": data["Volume"],
            "Open": data["Open"],
            "High": data["High"],
            "Low": data["Low"],
        })

        df = df.sort_values("Date")
        print(file)
        df.to_csv(file, index=False)
        return df
    
    raise(RuntimeError("Failed to download from yahoo finance"))
    return pd.DataFrame()



def load_data(raw_file : Path =  get_project_root() / "data" / "raw" / "SPY_history.csv")-> pd.DataFrame:
    if raw_file.exists():
        df = pd.read_csv(raw_file)
    else: 
        df = create_csv(raw_file)
    df['Close'] = df['Close/Last'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df


def compute_log_returns(df: pd.DataFrame):
    df['log_return'] = np.log(df["Close"] / df['Close'].shift(1))
    df = df.dropna(subset=['log_return'])
    return df

    
if __name__ == '__main__':
    '''df = load_data()
    compute_log_returns(df)
    print(df.head())'''