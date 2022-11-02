from pdpipe import PdPipelineStage
import pandas as pd
from category_encoders import TargetEncoder


class LagComputer(PdPipelineStage):
    def __init__(self, target_col, lags, split_key, date_column='date'):
        self.split_key = split_key
        self.lags = lags
        self.target_col = target_col
        self.date_column = date_column
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

        # saving last train days for the test data
        max_lag = max(self.lags)
        self.last_days_train = X[X[self.date_column] > str(X_last_date - pd.Timedelta(days=max_lag)).split(' ')[0]]

        self.is_fitted = True
        return X

    def _transform(self, X, verbose=None):
        if self.target_col not in X.columns:
            # filling lags for the test data with the 'target' of last days in X
            for current_lag in self.lags:
                if str(pd.to_datetime(X[self.date_column].min()) - pd.Timedelta(days=current_lag)).split(' ')[0] in \
                        self.last_days_train[self.date_column].unique():
                    X['lag_{}'.format(current_lag)] = \
                        self.last_days_train[self.last_days_train[self.date_column] ==
                                             str(pd.to_datetime(X[self.date_column].min()) - pd.Timedelta(days=current_lag)).split(' ')[0]][self.target_col].tolist()

        return X


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
