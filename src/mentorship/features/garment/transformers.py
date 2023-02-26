from pdpipe import PdPipelineStage
import pandas as pd
import numpy as np
import math


class LagComputer(PdPipelineStage):
    def __init__(self, target_col, lags, split_key, level, lagged_feature, date_column='date'):
        self.split_key = split_key
        self.level = level
        self.lags = lags
        self.target_col = target_col
        self.date_column = date_column
        self.lagged_feature = lagged_feature
        self.is_fitted = False
        self.last_days_train = None
        self.current_test_day = 0
        super_kwargs = {
            'exraise': True,
            'desc': 'Pipeline for lags computing',
        }
        super().__init__(**super_kwargs)

    def _prec(self, X: pd.DataFrame) -> bool:
        if self.date_column not in X.columns:
            return False

        return True

    def _fit_transform(self, X, verbose=None):
        X_last_date = pd.to_datetime(X[self.date_column].max())

        if not self.lags:
            self.is_fitted = True
            return X

        # saving last train days for the test data
        max_lag = max(self.lags)
        self.last_days_train = X[X[self.date_column] > str(X_last_date - pd.Timedelta(days=max_lag)).split(' ')[0]]

        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        self.current_test_day += 1
        # filling lags for the test data with the 'target' of last days in X

        for current_lag in self.lags:
            if self.current_test_day <= current_lag:
                for current_level in X[self.level].unique():
                    if len(list(self.last_days_train.loc[self.last_days_train[self.level] == current_level,
                                                         self.target_col])) < current_lag - self.current_test_day + 1:
                        X.loc[X[X[self.level] == current_level].index, f'{self.lagged_feature}_lag_{current_lag}'] = 1
                    else:
                        X.loc[X[X[self.level] == current_level].index, f'{self.lagged_feature}_lag_{current_lag}'] = \
                            list(self.last_days_train.loc[self.last_days_train[self.level] == current_level,
                                                          self.target_col])[self.current_test_day - current_lag - 1]
        return X

    def update_recent_history(self, X):
        self.last_days_train = pd.concat([self.last_days_train, X]).reset_index(drop=True)
        return


class RollingComputer(PdPipelineStage):
    def __init__(self, level='team', rolling_col=None, rolling_aggr=None, rolling_days=None):
        self.level = level
        self.rolling_col = rolling_col
        self.rolling_aggr = rolling_aggr
        self.rolling_days = rolling_days
        self.is_fitted = False
        self.last_days_train = None
        super_kwargs = {
            'exraise': True,
            'desc': 'Pipeline for over_time computing',
        }
        super().__init__(**super_kwargs)

    def _prec(self, X: pd.DataFrame) -> bool:
        return True

    def _fit_transform(self, X, verbose=None):
        self.last_days_train = X
        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        if self.rolling_col is None:
            return X

        for rolling_day in self.rolling_days:
            for function, aggregate in self.rolling_aggr.items():
                for level in X[self.level].unique():
                    X.loc[X[self.level] == level, [f'rolling_{self.rolling_col}_{rolling_day}d_{aggregate}']] = \
                        function(list(self.last_days_train.loc[self.last_days_train[self.level] == level, self.rolling_col])[-rolling_day:])
        return X

    def update_recent_history(self, X):
        self.last_days_train = pd.concat([self.last_days_train, X]).reset_index(drop=True)
        return


class OverTimeComputer(PdPipelineStage):
    def __init__(self, level='team', date_column='date'):
        self.level = level
        self.date_column = date_column
        self.is_fitted = False
        super_kwargs = {
            'exraise': True,
            'desc': 'Pipeline for over_time computing',
        }
        super().__init__(**super_kwargs)

    def _prec(self, X: pd.DataFrame) -> bool:
        if self.date_column not in X.columns:
            return False

        return True

    def _fit_transform(self, X, verbose=None):
        self.previous_data = X
        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        if set(X.index).issubset(self.previous_data.index):
            return X
        for level_value in X[self.level].unique():
            current_overtime_value = np.mean(self.previous_data[self.previous_data[self.level] == level_value]['over_time'][-6:])
            X.loc[X[X[self.level] == level_value].index, 'over_time'] = current_overtime_value
            if math.isnan(current_overtime_value):
                X.loc[X[X[self.level] == level_value].index, 'over_time'] = np.mean(
                    self.previous_data.groupby('date')['over_time'].mean()[-6:])
        self.previous_data = pd.concat([self.previous_data, X])
        return X
