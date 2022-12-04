from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np


class PositiveRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, fit_params=None, predict_negative=False):
        """Regressor that always predicts positive values"""
        self.estimator = estimator
        self.fit_params = fit_params
        self.predict_negative = predict_negative
        if fit_params is None:
            self.fit_params = {}

    def fit(self, X, y=None):
        self.estimator = self.estimator.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        if self.predict_negative is False:
            return np.clip(y_pred, 0, None)
        else:
            return y_pred

    def __getattr__(self, item):
        """
        Return attributes of the underlying estimator
        (for easier hyper-parameter tuning)
        """
        if item in self.__dict__.keys():
            return getattr(self, item)
        else:
            return getattr(self.estimator, item)


class LogTransformRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, fit_params=None, predict_negative=False):
        """Regressor that always predicts positive values"""
        self.estimator = estimator
        self.fit_params = fit_params
        self.predict_negative = predict_negative
        if fit_params is None:
            self.fit_params = {}

    def fit(self, X, y=None):
        y = np.log1p(y)
        self.estimator = self.estimator.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        if self.predict_negative is False:
            return np.clip(y_pred, 0, None)
        else:
            return y_pred

    def __getattr__(self, item):
        """
        Return attributes of the underlying estimator
        (for easier hyper-parameter tuning)
        """
        if item in self.__dict__.keys():
            return getattr(self, item)
        else:
            return getattr(self.estimator, item)
