from .data.load_data import load_data, compute_log_returns

import pandas as pd

def get_data() -> pd.DataFrame:
    df = load_data()
    compute_log_returns(df)
    return df


if __name__ == '__main__':
    raw = get_data()
    