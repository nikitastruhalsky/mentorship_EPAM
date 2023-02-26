from sklearn.base import clone, BaseEstimator
import numpy as np
import pandas as pd


class RecursiveTSEstimator(BaseEstimator):
    def __init__(self, split_key, target_col, base_pipelines=None, trend_pred=None, y_detrended=None,
                 detrended_categories=None, trend_models=None, learning_curve=False, class_weight=None,
                 predict_proba=False, zero_categories=None):
        if base_pipelines is None:
            base_pipelines = {}
        if detrended_categories is None:
            detrended_categories = []
        if trend_models is None:
            trend_models = []
        if zero_categories is None:
            zero_categories = []
        self.detrended_categories = detrended_categories
        self.trend_models = trend_models
        self.split_key = split_key
        self.target_col = target_col
        self.base_pipelines = base_pipelines
        self.pipelines_ = {}
        self.trend_pred = trend_pred
        self.y_detrended = y_detrended
        self.learning_curve = learning_curve
        self.class_weight = class_weight
        self.predict_proba = predict_proba
        self.zero_categories = zero_categories

    def fit(self, X, y):
        for key, X_part in X.groupby(self.split_key):
            if self.y_detrended is not None:
                y_part = self.y_detrended.loc[X_part.index]
            else:
                y_part = y.loc[X_part.index]

            # save train data for using it in counting rolling features for the test data in the future
            self.base_pipelines[key].train_data = X_part

            pipeline = clone(self.base_pipelines[key])
            if self.class_weight is not None:
                pipeline.estimator.estimator.set_params(**{'class_weight': self.class_weight[key]})
            pipeline = pipeline.fit(X_part.drop(columns=[self.split_key]), y_part)

            self.pipelines_[key] = pipeline
        return self

    def predict(self, X):
        y_preds = {}
        y_proba = {}
        y_predict_proba = []
        for split_key_value, X_part in X.groupby(self.split_key):
            if split_key_value in self.zero_categories:
                y_preds[split_key_value] = pd.Series(data=0, index=X_part.index, name='forecast')
                continue

            pipeline = self.pipelines_[split_key_value]

            # sort dates of X_part for safety
            sorted_dates = X_part[pipeline.date_column].unique().tolist()
            sorted_dates.sort()

            for current_day_number, current_day in enumerate(sorted_dates):
                X_current_day = X_part.loc[X_part[pipeline.date_column] == current_day, :]

                if self.predict_proba:
                    pred = pipeline.predict(X_current_day.drop(columns=[self.split_key]))
                    y_pred = pred[0]
                    y_predict_proba = pred[1]
                else:
                    y_pred = pipeline.predict(X_current_day.drop(columns=[self.split_key]))

                y_preds[(split_key_value, current_day)] = pd.Series(data=y_pred, index=X_current_day.index,
                                                                    name='forecast')
                if self.predict_proba:
                    y_proba[(split_key_value, current_day)] = pd.Series(data=y_predict_proba, index=X_current_day.index,
                                                                        name='forecast')

                X_current_day.loc[:, self.target_col] = y_pred
                if pipeline.lags:
                    pipeline.pipeline[0].update_recent_history(X_current_day.drop(columns=[self.split_key]))
                if pipeline.rolling_days:
                    pipeline.pipeline[1].update_recent_history(X_current_day.drop(columns=[self.split_key]))

            if pipeline.use_rmsle:
                for current_key in [key for key in y_preds.keys() if key[0] == split_key_value]:
                    y_preds[current_key] = np.exp(y_preds[current_key]) - 1

        y_pred = pd.concat(y_preds.values()).loc[X.index]
        if self.predict_proba:
            y_predict_proba = pd.concat(y_proba.values()).loc[X.index]
            if self.trend_pred is None:
                return y_pred, y_predict_proba
            else:
                return y_pred + self.trend_pred.loc[y_pred.index], y_predict_proba
        else:
            if self.trend_pred:
                return y_pred + self.trend_pred.loc[y_pred.index], y_predict_proba
            else:
                return y_pred
