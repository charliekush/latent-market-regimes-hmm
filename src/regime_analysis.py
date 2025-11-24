from .data.load_data import load_data, compute_log_returns

import pandas as pd

def get_data() -> pd.DataFrame:
    df = load_data()
    compute_log_returns(df)
    return df
def attach_regimes_to_dataframe(df, states, col_name="regime"):
    """Attach decoded regime labels to the DataFrame"""
    pass


def compute_regime_statistics(df, return_col="log_return", regime_col="regime"):
    """Compute per-regime return statistics"""
    pass


def plot_price_with_regimes(df, price_col="Close", regime_col="regime", out_path=None):
    """Plot price over time with regimes highlighted"""
    pass


def plot_return_histograms_by_regime(df, return_col="log_return", regime_col="regime", out_path=None):
    """Plot return histograms grouped by regime"""
    pass


def plot_regime_sequence(df, regime_col="regime", out_path=None):
    """Plot the regime sequence over time"""
    pass


if __name__ == '__main__':
    df = get_data()
    
    