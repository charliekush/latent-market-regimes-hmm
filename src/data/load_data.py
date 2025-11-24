import yfinance as yf
from pathlib import Path
import pandas as pd
import numpy as np
import re
import datetime



def file_name(ticker: str, start: str, end: str) -> str:
    tick = re.sub(r'\W+', '', ticker)

    return f"{tick}_{start.split('-')[0]}-{end.split('-')[0]}.csv"
    

def create_csv(ticker: str, start: str, end: str) -> pd.DataFrame:
    
    try:
        data = yf.download(ticker, 
                           start=start, 
                           end=end,  
                           auto_adjust=False)
    except Exception as e:
        raise(RuntimeError(f"Failed to download from yahoo finance: {e}"))

    if (data is not None):

        try: 
            data.columns = data.columns.droplevel('Ticker')
            data = data.reset_index() 
            df =  pd.DataFrame({
                "Date": data["Date"],
                "Close/Last": data["Close"],
                "Volume": data["Volume"],
                "Open": data["Open"],
                "High": data["High"],
                "Low": data["Low"]
            })
            
            
            return df
        except Exception as e:
            raise(RuntimeError(f"Failed to download from yahoo finance: {e}"))

    
    raise(RuntimeError("Dataframe is empty"))
    return pd.DataFrame()



def load_data(dir: Path, ticker: str, start: str, end: str)-> pd.DataFrame:
    file: Path =  dir / file_name(ticker, start, end)
    
    if file.exists():
        df = pd.read_csv(file)
    else: 
        df = create_csv(ticker, start, end)
        df.to_csv(dir / file_name(ticker, start, end), index=False)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df

    
if __name__ == '__main__':
    '''df = load_data()
    compute_log_returns(df)
    print(df.head())'''
