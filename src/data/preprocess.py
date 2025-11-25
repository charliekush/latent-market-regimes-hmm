from typing import Tuple

import numpy as np
import pandas as pd
def clean_ohlc_dataframe(df: pd.DataFrame, price_col : str ="Close"):
    """Clean OHLCV data types, sorting, and obvious NaNs"""
    df = df.copy()
    df.dropna()
    df['Close'] = df['Close/Last'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date')

    return df


def add_log_returns(df, price_col="Close", return_col="log_return"):
    """Add log returns column based on the given price column"""

    df['log_return'] = np.log(df["Close"] / df['Close'].shift(1))
    df = df.dropna(subset=['log_return'])

    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    date_col: str = "Date",
    cutoff_date=None,
    split_ratio: float = -1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test partitions using a cutoff date"""
    
    ratio_passed = (split_ratio <= 1.0 and split_ratio >= 0.0)
    cutoff_passed = (cutoff_date is not None)
    if not (ratio_passed ^ cutoff_passed):
        raise ValueError("requires exactly one parameter ('cutoff_date' or 'split_ratio') to be passed")
    

    if ratio_passed:
        cutoff_row = df.iloc[int((split_ratio * len(df)))]
        cutoff_date = cutoff_row[date_col]
    

    
    
    train = df[df[date_col] < cutoff_date].copy()
    test = df[df[date_col] >= cutoff_date].copy()

    return train, test

    
