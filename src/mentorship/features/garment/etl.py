import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

DATA_ROOT = Path('data', 'kaggle', 'store-sales-time-series-forecasting')


class ETLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date', lags=None, use_final_metric=True, lagged_feature='bigger_than_targeted',
                 rolling_days=None, rolling_aggr=None, rolling_col='bigger_than_targeted'):
        if lags is None:
            lags = []
        self.lags = lags
        self.date_column = date_column
        self.lagged_feature = lagged_feature
        self.use_final_metric = use_final_metric
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.rolling_col = rolling_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, 'bigger_than_targeted'] = (X['targeted_productivity'] <= X['actual_productivity']).astype(int)
        X = X.drop(columns=['actual_productivity'])

        X['date'] = pd.to_datetime(X['date']).dt.strftime('%Y-%m-%d')

        X = X.replace('finishing ', 'finishing')

        X.loc[:, 'over_time'] = X.groupby(['department', 'team'])['over_time'].shift(1)
        X = X.dropna(subset=['over_time']).reset_index(drop=True)

        # lags for train set
        if self.lags:
            for current_lag in self.lags:
                X.loc[:, '{}_lag_{}'.format(self.lagged_feature, current_lag)] = X.groupby(
                    ['department', 'team'])[self.lagged_feature].shift(current_lag)

        X = X.fillna(1)

        # rolling features for train set
        if self.rolling_days:
            for rolling_day in self.rolling_days:
                for rolling_aggr in self.rolling_aggr.items():
                    if rolling_aggr[1] == 'mean':
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_mean'] = \
                            X.groupby(['department', 'team'])[self.rolling_col].transform(lambda s: (s.shift().rolling(rolling_day, min_periods=1).mean()))
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_mean'] = X[f'rolling_{self.rolling_col}_{rolling_day}d_mean'].fillna(0.75)
                    if rolling_aggr[1] == 'std':
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_std'] = \
                            X.groupby(['department', 'team'])[self.rolling_col].transform(lambda s: (s.shift().rolling(rolling_day, min_periods=1).std()))
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_std'] = X[f'rolling_{self.rolling_col}_{rolling_day}d_std'].fillna(0)
                    if rolling_aggr[1] == 'median':
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_median'] = \
                            X.groupby(['department', 'team'])[self.rolling_col].transform(lambda s: (s.shift().rolling(rolling_day, min_periods=1).median()))
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_median'] = X[f'rolling_{self.rolling_col}_{rolling_day}d_median'].fillna(1)
                    if rolling_aggr[1] == 'max':
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_max'] = \
                            X.groupby(['department', 'team'])[self.rolling_col].transform(lambda s: (s.shift().rolling(rolling_day, min_periods=1).max()))
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_max'] = X[f'rolling_{self.rolling_col}_{rolling_day}d_max'].fillna(1)
                    if rolling_aggr[1] == 'min':
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_min'] = \
                            X.groupby(['department', 'team'])[self.rolling_col].transform(lambda s: (s.shift().rolling(rolling_day, min_periods=1).min()))
                        X[f'rolling_{self.rolling_col}_{rolling_day}d_min'] = X[f'rolling_{self.rolling_col}_{rolling_day}d_min'].fillna(0)
        return X
