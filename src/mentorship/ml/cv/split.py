import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class DateTimeSeriesSplit:
    def __init__(self, date_column='date', n_splits=4, max_train_size=365, test_size=16):
        self.date_column = date_column
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        dates = pd.DataFrame(data=X[self.date_column]).drop_duplicates(ignore_index=True)
        dates = dates.sort_values('date')
        tscv = TimeSeriesSplit(gap=0, max_train_size=self.max_train_size, n_splits=self.n_splits,
                               test_size=self.test_size)
        for train_index, test_index in tscv.split(dates):
            train_dates = dates.iloc[train_index][self.date_column]
            test_dates = dates.iloc[test_index][self.date_column]
            X_train_index = np.nonzero(X[self.date_column].isin(train_dates).values)[0]
            X_test_index = np.nonzero(X[self.date_column].isin(test_dates).values)[0]
            yield X_train_index, X_test_index
