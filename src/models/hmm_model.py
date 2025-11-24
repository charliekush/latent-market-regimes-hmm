from typing import Any, Dict, Sequence, Tuple

import numpy as np


def build_hmm(
    n_states: int,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 0,
) -> Any:
    """Construct an untrained Gaussian HMM model."""
    pass


def fit_hmm(model: Any, X_train: np.ndarray) -> Any:
    """Fit HMM parameters with EM on training data."""
    pass


def score_hmm(model: Any, X: np.ndarray) -> float:
    """Compute log-likelihood of observations under the model."""
    pass


def decode_regimes(model: Any, X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Decode most likely hidden state sequence with Viterbi."""
    pass


def summarize_regimes(model: Any) -> Dict[str, Any]:
    """Summarize regime statistics such as means and transition matrix."""
    pass


def fit_and_evaluate_hmms(
    X_train: np.ndarray, X_test: np.ndarray, state_list: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    """Fit HMMs for each state count and collect train/test scores."""
    pass
