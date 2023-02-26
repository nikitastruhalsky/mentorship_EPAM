from pdpipe import PdPipelineStage
import pandas as pd
from category_encoders import TargetEncoder
from numpy.lib.stride_tricks import sliding_window_view


class LagComputer(PdPipelineStage):
    def __init__(self, target_col, lags, split_key, level, lagged_feature, date_column='date'):
        self.split_key = split_key
        self.level = level
        self.lags = lags
        self.target_col = target_col
        self.date_column = date_column
        self.lagged_feature = lagged_feature
        self.last_days_train = None
        self.is_fitted = False
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
        self.last_days_train = X.loc[X[self.date_column] > str(X_last_date - pd.Timedelta(days=max_lag)).split(' ')[0], :]

        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        # filling lags for the test data with the 'target' of last days in X
        for current_lag in self.lags:
            lagged_date = self.last_days_train[self.date_column].unique()[-current_lag]
            X.loc[:, '{}_lag_{}'.format(self.lagged_feature, current_lag)] = \
                self.last_days_train.loc[self.last_days_train[self.date_column] == lagged_date, self.lagged_feature].tolist()
        return X

    def update_recent_history(self, X):
        self.last_days_train = pd.concat([self.last_days_train, X]).reset_index(drop=True)
        return


class RollingComputer(PdPipelineStage):
    def __init__(self, target_col, rolling_days, rolling_aggr, split_key, level, rolling_feature, date_column='date'):
        self.split_key = split_key
        self.level = level
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.target_col = target_col
        self.date_column = date_column
        self.rolling_feature = rolling_feature
        self.last_days_train = None
        self.is_fitted = False
        super_kwargs = {
            'exraise': True,
            'desc': 'Pipeline for computing rolling features',
        }
        super().__init__(**super_kwargs)

    def _prec(self, X: pd.DataFrame) -> bool:
        if self.date_column not in X.columns:
            return False

        return True

    def _fit_transform(self, X, verbose=None):
        X_last_date = pd.to_datetime(X[self.date_column].max())

        if not self.rolling_days:
            self.is_fitted = True
            return X

        # saving last train days for the test data
        max_rolling_day = max(self.rolling_days)
        self.last_days_train = X[
            X[self.date_column] > str(X_last_date - pd.Timedelta(days=max_rolling_day)).split(' ')[0]]

        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        if self.last_days_train is not None:
            grp = self.last_days_train[self.rolling_feature].groupby(self.last_days_train[self.level])
            for period in self.rolling_days:
                rolling_slices = [sliding_window_view(v, period) for _, v in grp]
                for function, aggregate in self.rolling_aggr.items():
                    # passing last 'X_current_day['store_nbr'].nunique()' (54) rolling values to the X_current_day
                    # last 54 values correspond to the current_day data
                    X.loc[:, f'rolling_{period}d_{aggregate}'] = pd.Series(function(rolling_slices, -1).ravel(order='F')).tail(X['store_nbr'].nunique()).values
        return X

    def update_recent_history(self, X):
        self.last_days_train = pd.concat([self.last_days_train, X]).reset_index(drop=True)
        return


class TargetEncodingTransformer(PdPipelineStage):
    def __init__(self, target_col, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.target_col = target_col
        self.encoder = TargetEncoder()
        self.is_fitted = False
        super_kwargs = {
            'exraise': True,
            'desc': 'Pipeline for target encoding',
        }
        super().__init__(**super_kwargs)

    def _prec(self, X: pd.DataFrame) -> bool:
        return True

    def _fit_transform(self, X, verbose=None):
        if not self.columns_to_encode:
            self.is_fitted = True
            return X

        X[self.columns_to_encode] = self.encoder.fit_transform(X[self.columns_to_encode], X[self.target_col])
        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        if not self.columns_to_encode:
            return X

        X[self.columns_to_encode] = self.encoder.transform(X[self.columns_to_encode])
        return X
