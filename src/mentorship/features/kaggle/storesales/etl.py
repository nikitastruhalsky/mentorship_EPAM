import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = Path('data', 'kaggle', 'store-sales-time-series-forecasting')


class ETLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='id', date_column='date', lags=None, target_col=None, is_test=False,
                 use_rmsle=True, rolling_days=None, rolling_aggr=None, lagged_feature='sales'):
        self.date_column = date_column
        self.id_column = id_column
        self.oil_data = pd.read_csv(DATA_ROOT / 'oil.csv')
        self.stores_data = pd.read_csv(DATA_ROOT / 'stores.csv')
        self.holidays_data = pd.read_csv(DATA_ROOT / 'holidays_events.csv').drop_duplicates(subset='date')
        self.lags = lags
        self.target_col = target_col
        self.use_rmsle = use_rmsle
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.lagged_feature = lagged_feature
        self.is_test = is_test

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.merge(self.oil_data, on=self.date_column, how='left')
        X['dcoilwtico'] = X['dcoilwtico'].fillna(method='ffill')
        X = X.sort_values(by=[self.id_column], ascending=True, ignore_index=True)
        X = X.drop(columns=[self.id_column])
        X['family'] = X['family'].str.lower()
        X = X.sort_values(by=['date', 'store_nbr', 'family']).reset_index(drop=True)
        if not self.is_test:
            y = X['sales'].copy()

        if self.target_col and self.use_rmsle:
            X = LogTransformer(cols=self.target_col).fit_transform(X)

        # lags for train set
        if self.lags:
            for current_lag in self.lags:
                X.loc[:, '{}_lag_{}'.format(self.lagged_feature, current_lag)] = X.groupby(
                    ['store_nbr', 'family'])[self.lagged_feature].shift(current_lag)

        # rolling features for train set
        if self.rolling_days:
            grp = X[self.target_col].groupby(X['store_nbr'])
            for period in self.rolling_days:
                rolling_slices = [sliding_window_view(v, period) for _, v in grp]
                for function, aggregate in self.rolling_aggr.items():
                    # giving correct indices to the pd.Series with rolling values (to correctly assign them to X)
                    X.loc[period * X['store_nbr'].nunique():, f'rolling_{period}d_{aggregate}'] = pd.Series(
                        function(rolling_slices, -1).ravel(order='F'),
                        index=range(period * X['store_nbr'].nunique(), X.shape[0] + X['store_nbr'].nunique()))
        if self.is_test:
            return X
        else:
            return X, y

    def add_is_holiday_feature(self, X):
        X_copy = X.copy()
        X_copy = X_copy.merge(self.holidays_data, on='date', how='left', indicator=True)
        X_copy['_merge'] = LabelEncoder().fit_transform(X_copy['_merge'].tolist())
        return X_copy['_merge']

    def add_store_features(self, X, columns_to_add):
        stores_data = self.stores_data.drop(columns=[x for x in self.stores_data.columns if x != 'store_nbr'
                                                     and x not in columns_to_add])
        for col in columns_to_add:
            stores_data[col] = LabelEncoder().fit_transform(stores_data[col])
        stores_data = stores_data.rename(columns={x: 'store_' + x for x in columns_to_add})
        X = X.merge(stores_data, on='store_nbr', how='left')
        return X

    def add_time_step_feature(self, X):
        X_dates = X[[self.date_column]].drop_duplicates().sort_values(by=self.date_column, ignore_index=True)
        X_dates = X_dates.reset_index().rename(columns={'index': 'time'})
        X = X.merge(X_dates, on=self.date_column, how='left')
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit_transform(self, X, **fit_params):
        X[self.cols] = np.log1p(X[self.cols])
        return X
