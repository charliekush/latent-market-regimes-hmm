from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import get_project_root
from .data.load_data import compute_log_returns, load_data

PathType = Union[str, Path]


def get_data(dir: Path, ticker: str, start: str, end: str) -> pd.DataFrame:
    df = load_data(dir, ticker, start, end)
    compute_log_returns(df)
    return df


def attach_regimes_to_dataframe(
    df: pd.DataFrame, states: Sequence[int] | np.ndarray, col_name: str = "regime"
) -> pd.DataFrame:
    """Attach decoded regime labels to the DataFrame."""
    pass


def compute_regime_statistics(
    df: pd.DataFrame, return_col: str = "log_return", regime_col: str = "regime"
) -> pd.DataFrame:
    """Compute per-regime return statistics."""
    pass


def plot_price_with_regimes(
    df: pd.DataFrame,
    price_col: str = "Close",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
) -> None:
    """Plot price over time with regimes highlighted."""
    pass


def plot_return_histograms_by_regime(
    df: pd.DataFrame,
    return_col: str = "log_return",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
) -> None:
    """Plot return histograms grouped by regime."""
    pass


def plot_regime_sequence(
    df: pd.DataFrame, regime_col: str = "regime", out_path: Optional[PathType] = None
) -> None:
    """Plot the regime sequence over time."""
    pass


if __name__ == '__main__':
    data_dir : Path =  get_project_root() / "data" / "raw"
    start: str = '1958-01-01'
    end: str = '2025-11-21'
    
    ticker: str = "^GSPC"

    split_date = "1998-01-01"
    df = get_data(data_dir, ticker, start, end)

    
    
