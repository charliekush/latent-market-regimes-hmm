import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path

PathType = Union[str, Path]

def plot_price_with_regimes(
    df: pd.DataFrame,
    price_col: str = "Close",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
) -> None:
    """Plot price over time with regimes highlighted."""
    plt.figure(figsize=(12, 6))
    regimes = sorted(df[regime_col].unique())

    for r in regimes:
        mask = df[regime_col] == r
        plt.plot(df["Date"][mask], df[price_col][mask], ".", label=f"Regime {r}", markersize=2)

    plt.xlabel("Date")
    plt.ylabel(price_col)
    plt.legend()
    plt.title("Price with HMM Regimes")
    if out_path:
        plt.savefig(out_path, dpi=300)
    


def plot_return_histograms_by_regime(
    df: pd.DataFrame,
    return_col: str = "log_return",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
) -> None:

    regimes = sorted(df[regime_col].unique())
    plt.figure(figsize=(10, 6))

    for r in regimes:
        subset = df[df[regime_col] == r]
        plt.hist(subset[return_col], bins=40, alpha=0.5, label=f"Regime {r}")

    plt.xlabel("Log return")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Return Distributions per Regime")

    if out_path:
        plt.savefig(out_path, dpi=300)


def plot_regime_sequence(
    df: pd.DataFrame, regime_col: str = "regime", out_path: Optional[PathType] = None
) -> None:
    """Plot the regime sequence over time."""
    
    plt.figure(figsize=(12, 3))
    plt.plot(df["Date"], df[regime_col], drawstyle="steps-mid")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.title("HMM Regime Sequence")

    if out_path:
        plt.savefig(out_path, dpi=300)
