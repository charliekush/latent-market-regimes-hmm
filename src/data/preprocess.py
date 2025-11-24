def clean_ohlc_dataframe(df, price_col="Close"):
    """Clean OHLCV data types, sorting, and obvious NaNs"""
    pass


def add_log_returns(df, price_col="Close", return_col="log_return"):
    """Add log returns column based on the given price column"""
    pass


def train_test_split_by_date(df, cutoff_date, date_col="Date"):
    """Split DataFrame into train/test partitions using a cutoff date"""
    pass


def save_processed_data(df, path):
    """Persist processed data subset to disk"""
    pass
