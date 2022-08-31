from sklearn.base import RegressorMixin, MetaEstimatorMixin, BaseEstimator
import numpy as np


class PositiveRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator):
        """Regressor that always predicts positive values"""
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        return self.estimator.fit(X, y, **fit_params)

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
