from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.preprocess import add_log_returns, clean_ohlc_dataframe, train_test_split_by_date
from src.models.gauss_hmm import GaussianHMM
from src.models.hmm_model import fit_and_evaluate_hmms
from .utils import get_project_root
from .data.load_data import compute_log_returns, load_data











if __name__ == '__main__':
    data_dir : Path =  get_project_root() / "data" / "raw"
    start: str = '1958-01-01'
    end: str = '2025-11-21'
    
    ticker: str = "^GSPC"

    split_date = "1998-01-01"
    df = load_data(data_dir, ticker, start, end)
    df = clean_ohlc_dataframe(df, "Close")
    df = add_log_returns(df)
    X_test, X_train, = train_test_split_by_date(df, "Date", split_date)
    fit_and_evaluate_hmms(X_train=X_train, X_test=X_test, state_list=[2,3,4])
    

    
    

    
