import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = Path('data', 'kaggle', 'store-sales-time-series-forecasting')


class ETLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_column='id', date_column='date', adding_lags=True, lags=[], target_col=None,
                 use_final_metric=True,
                 adding_rolling_features=False, rolling_days=[], rolling_aggr={}):
        self.date_column = date_column
        self.id_column = id_column
        self.oil_data = pd.read_csv(DATA_ROOT / 'oil.csv')
        self.adding_lags = adding_lags
        self.lags = lags
        self.target_col = target_col
        self.use_final_metric = use_final_metric
        self.adding_rolling_features = adding_rolling_features
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.merge(self.oil_data, on=self.date_column, how='left')
        X['dcoilwtico'] = X['dcoilwtico'].fillna(method='ffill')
        X = X.sort_values(by=[self.id_column], ascending=True, ignore_index=True)
        X = X.drop(columns=[self.id_column])
        X['family'] = X['family'].str.lower()

        # lags for train set
        if self.adding_lags:
            X_copy = X.copy()
            if self.use_final_metric:
                X_copy[self.target_col] = np.log(X_copy[self.target_col] + 1)
            for current_lag in self.lags:
                X.loc[:, 'lag_{}'.format(current_lag)] = X_copy.groupby(['store_nbr', 'family'])[self.target_col].shift(
                    current_lag)

        # rolling features for train set
        if self.adding_rolling_features:
            X_copy = X.copy()
            if self.use_final_metric:
                X_copy[self.target_col] = np.log(X_copy[self.target_col] + 1)

            grp = X_copy[self.target_col].groupby(X_copy['store_nbr'])
            for period in self.rolling_days:
                rolling_slices = [sliding_window_view(v, period) for _, v in grp]
                for function, aggregate in self.rolling_aggr.items():
                    # giving correct indices to the pd.Series with rolling values (to correctly assign them to X)
                    X.loc[period * X['store_nbr'].nunique():, f'rolling_{period}d_{aggregate}'] = pd.Series(
                        function(rolling_slices, -1).ravel(order='F'),
                        index=range(period * X['store_nbr'].nunique(), X.shape[0] + X['store_nbr'].nunique()))
        return X, y

    def adding_is_holiday_feature(self, X):
        X_copy = X.copy()
        holidays_data = pd.read_csv(DATA_ROOT / 'holidays_events.csv').drop_duplicates(subset='date')
        X_copy = X_copy.merge(holidays_data, on='date', how='left', indicator=True)
        X_copy['_merge'] = LabelEncoder().fit_transform(X_copy['_merge'].tolist())
        return X_copy['_merge']

    def adding_stores_data(self, X, columns_to_add):
        stores_data = pd.read_csv(DATA_ROOT / 'stores.csv')
        stores_data = stores_data.drop(columns=[x for x in stores_data.columns if x != 'store_nbr'
                                                and x not in columns_to_add])
        stores_data = stores_data.rename(columns={x: 'store_' + x for x in columns_to_add})
        X = X.merge(stores_data, on='store_nbr', how='left')
        return X

    def adding_time_step(self, X):
        X_dates = X[[self.date_column]].drop_duplicates().sort_values(by=self.date_column, ignore_index=True)
        X_dates = X_dates.reset_index().rename(columns={'index': 'time'})
        X = X.merge(X_dates, on=self.date_column, how='left')
        return X
