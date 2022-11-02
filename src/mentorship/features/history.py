from typing import Optional, Tuple
import pandas as pd


def cut_history(
    X: pd.DataFrame,
    date_column: str,
    keep_interval: pd.Timedelta,
    y: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    last_date = pd.to_datetime(X[date_column].max())
    X_train = X[X[date_column] >= str(last_date - keep_interval).split(' ')[0]]
    if y is not None:
        assert X.index.equals(y.index)
        y = y.loc[X_train.index]

    return X_train, y
