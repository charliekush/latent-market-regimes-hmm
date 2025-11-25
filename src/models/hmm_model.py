from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.models.gauss_hmm import GaussianHMM

import pandas as pd



def fit_and_evaluate_hmms(X_train: np.ndarray, X_test: np.ndarray, state_list: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    """Fit HMMs for each state count and collect train/test scores."""
    results = {}

    for K in state_list:
        model = GaussianHMM(n_states=K, n_iter=100, random_state=0)
        model.fit(X_train)

        train_ll = model.score(X_train)
        test_ll = model.score(X_test)

        results[K] = {
            "model": model,
            "train_ll": train_ll,
            "test_ll": test_ll
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