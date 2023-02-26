from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, fit_params=None, predict_proba=False):
        """Regressor that always predicts positive values"""
        self.estimator = estimator
        self.fit_params = fit_params
        self.predict_proba = predict_proba
        if fit_params is None:
            self.fit_params = {}

    def fit(self, X, y=None):
        self.estimator = self.estimator.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        if self.predict_proba is False:
            return y_pred
        else:
            y_predict_proba = self.estimator.predict_proba(X)[:, 1]
            return y_pred, y_predict_proba

    def __getattr__(self, item):
        """
        Return attributes of the underlying estimator
        (for easier hyper-parameter tuning)
        """
        if item in self.__dict__.keys():
            return getattr(self, item)
        else:
            return getattr(self.estimator, item)
