import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Sequence
from pathlib import Path

PathType = Union[str, Path]

def plot_price_with_regimes(
    df: pd.DataFrame,
    price_col: str = "Close",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
    pgf_data_path: Optional[PathType] = None,
    pgf_downsample: int = 5,
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
    if pgf_data_path:
        pgf_df = df[["Date", price_col, regime_col]].copy()
        pgf_df["Date"] = pd.to_datetime(pgf_df["Date"]).dt.strftime("%Y-%m-%d")
        if pgf_downsample > 1:
            pgf_df = pgf_df.iloc[::pgf_downsample]
        Path(pgf_data_path).parent.mkdir(parents=True, exist_ok=True)
        pgf_df.to_csv(pgf_data_path, index=False)
    


def plot_return_histograms_by_regime(
    df: pd.DataFrame,
    return_col: str = "log_return",
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
    pgf_data_path: Optional[PathType] = None,
    xlim: Optional[tuple[float, float]] = None,
) -> None:

    regimes = sorted(df[regime_col].unique())
    plt.figure(figsize=(10, 6))
    combined_returns = df[return_col].dropna().to_numpy()
    bin_edges = np.histogram_bin_edges(combined_returns, bins=40)
    pgf_rows = []
    colors = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
    )

    # Draw higher-volume regimes first so lower-volume bars appear on top.
    regime_sizes = {
        r: len(df[df[regime_col] == r][return_col].dropna()) for r in regimes
    }
    ordered_regimes = sorted(regimes, key=lambda r: regime_sizes[r], reverse=True)

    for idx, r in enumerate(ordered_regimes):
        subset = df[df[regime_col] == r]
        values = subset[return_col].dropna()
        plt.hist(
            values,
            bins=bin_edges,
            alpha=0.55,
            label=f"Regime {r}",
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
        )

        if pgf_data_path:
            counts, _ = np.histogram(values, bins=bin_edges)
            for left, right, count in zip(bin_edges[:-1], bin_edges[1:], counts):
                center = 0.5 * (left + right)
                pgf_rows.append(
                    {
                        "regime": r,
                        "bin_left": left,
                        "bin_right": right,
                        "bin_center": center,
                        "count": count,
                    }
                )

    plt.xlabel("Log return")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Return Distributions per Regime")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    if xlim:
        plt.xlim(xlim)

    if out_path:
        plt.savefig(out_path, dpi=300)
    if pgf_data_path and pgf_rows:
        pgf_df = pd.DataFrame(pgf_rows)
        Path(pgf_data_path).parent.mkdir(parents=True, exist_ok=True)
        pgf_df.to_csv(pgf_data_path, index=False)


def plot_regime_sequence(
    df: pd.DataFrame,
    regime_col: str = "regime",
    out_path: Optional[PathType] = None,
    pgf_data_path: Optional[PathType] = None,
    pgf_downsample: int = 5,
) -> None:
    """Plot the regime sequence over time."""
    
    plt.figure(figsize=(12, 3))
    plt.plot(df["Date"], df[regime_col], drawstyle="steps-mid")
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.title("HMM Regime Sequence")

    if out_path:
        plt.savefig(out_path, dpi=300)
    if pgf_data_path:
        pgf_df = df[["Date", regime_col]].copy()
        pgf_df["Date"] = pd.to_datetime(pgf_df["Date"]).dt.strftime("%Y-%m-%d")
        if pgf_downsample > 1:
            pgf_df = pgf_df.iloc[::pgf_downsample]
        Path(pgf_data_path).parent.mkdir(parents=True, exist_ok=True)
        pgf_df.to_csv(pgf_data_path, index=False)


def plot_loglik_convergence(
    log_likelihoods: Sequence[float], out_path: Optional[PathType] = None
) -> None:
    """Plot EM log-likelihood trajectory over iterations."""
    if len(log_likelihoods) == 0:
        return
    plt.figure(figsize=(8, 4))
    iterations = np.arange(1, len(log_likelihoods) + 1)
    plt.plot(iterations, log_likelihoods, marker="o", markersize=3)
    plt.xlabel("EM Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs. EM Iteration")
    plt.grid(True, linestyle="--", alpha=0.4)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
