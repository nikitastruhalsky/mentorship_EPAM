from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np


class PositiveRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, fit_params=None):
        """Regressor that always predicts positive values"""

        if fit_params is None:
            fit_params = {}
        self.estimator = estimator
        self.fit_params = fit_params

    def fit(self, X, y=None):
        self.estimator = self.estimator.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        return np.clip(y_pred, 0, None)

    def __getattr__(self, item):
        """
        Return attributes of the underlying estimator
        (for easier hyper-parameter tuning)
        """
        if item in self.__dict__.keys():
            return getattr(self, item)
        else:
            return getattr(self.estimator, item)
