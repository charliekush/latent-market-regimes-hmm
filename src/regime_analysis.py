from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.preprocess import add_log_returns, clean_ohlc_dataframe, train_test_split_by_date
from src.models.gauss_hmm import GaussianHMM
from src.models.hmm_model import attach_regimes_to_dataframe, compute_regime_statistics, fit_and_evaluate_hmms
from src.plot import plot_price_with_regimes, plot_regime_sequence, plot_return_histograms_by_regime
from .utils import get_project_root
from .data.load_data import compute_log_returns, load_data


if __name__ == '__main__':
    data_dir: Path = get_project_root() / "data" / "raw"
    start: str = '1958-01-01'
    end: str = '2025-11-21'

    ticker: str = "^GSPC"

    split_date = "1998-01-01"
    df = load_data(data_dir, ticker, start, end)
    df = clean_ohlc_dataframe(df, "Close")
    df = add_log_returns(df)
    train_df, test_df = train_test_split_by_date(
        df, date_col="Date", cutoff_date=split_date)

    X_train = train_df["log_return"].to_numpy().reshape(-1, 1)
    X_test = test_df["log_return"].to_numpy().reshape(-1, 1)

    results = fit_and_evaluate_hmms(
        X_train=X_train, X_test=X_test, state_list=[2, 3, 4])
    for K, info in results.items():
        print(
            f"K={K}: train_ll={info['train_ll']:.1f}, "
            f"test_ll={info['test_ll']:.1f}, "
            f"AIC={info['aic']:.1f}, BIC={info['bic']:.1f}"
        )

    # choose best K by lowest BIC 
    best_K = min(results.keys(), key=lambda k: results[k]["bic"])
    best_model = results[best_K]["model"]
    print(f"Chosen K={best_K} by BIC")
    
    X_full = df["log_return"].to_numpy().reshape(-1, 1)
    logprob, states = best_model.decode(X_full)
    df = attach_regimes_to_dataframe(df, states, col_name="regime")
    
    
    summary = best_model.summarize_regimes()
    print("Means:", summary["means"])
    print("Stds:", summary["stds"])
    print("Expected durations:", summary["expected_durations"])

    stats_df = compute_regime_statistics(df)
    print(stats_df)

    plot_price_with_regimes(df)
    plot_return_histograms_by_regime(df)
    plot_regime_sequence(df)

    
