from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.models.gauss_hmm import GaussianHMM

import pandas as pd



def fit_and_evaluate_hmms(
    X_train: np.ndarray,
    X_test: np.ndarray,
    state_list: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    """Fit HMMs for each state count and collect train/test scores + AIC/BIC."""
    results: Dict[int, Dict[str, Any]] = {}
    T = X_train.shape[0]

    for K in state_list:
        model = GaussianHMM(n_states=K, n_iter=100, random_state=0)
        model.fit(X_train)

        train_ll = model.score(X_train)
        test_ll = model.score(X_test)

        # number of free parameters for 1D Gaussian HMM
        p = K**2 + 2 * K - 1

        aic = 2 * p - 2 * train_ll
        bic = p * np.log(T) - 2 * train_ll

        results[K] = {
            "model": model,
            "train_ll": train_ll,
            "test_ll": test_ll,
            "aic": aic,
            "bic": bic,
        }

    return results

def attach_regimes_to_dataframe(
    df: pd.DataFrame, states: Sequence[int] | np.ndarray, col_name: str = "regime"
) -> pd.DataFrame:
    """Attach decoded regime labels to the DataFrame."""
    df = df.copy()
    if len(df) != len(states):
        raise ValueError("Length of states dos not match DataFrame length")
    df[col_name] = states
    return df


def compute_regime_statistics(
    df: pd.DataFrame, return_col: str = "log_return", regime_col: str = "regime"
) -> pd.DataFrame:
    """Compute per-regime return statistics."""
    grouped = df.groupby(regime_col)[return_col]
    stats = grouped.agg(["mean", "std", "count"])
    stats["fraction"] = stats["count"] / len(df)
    return stats