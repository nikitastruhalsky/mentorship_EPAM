from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import clone, BaseEstimator
import numpy as np
import pandas as pd


class RecursiveTSEstimator(BaseEstimator):
    def __init__(self, split_key, target_col, base_pipelines=None, trend_pred=None, y_detrended=None,
                 detrended_categories=None, trend_models=None):
        if base_pipelines is None:
            base_pipelines = {}
        if detrended_categories is None:
            detrended_categories = []
        if trend_models is None:
            trend_models = []
        self.detrended_categories = detrended_categories
        self.trend_models = trend_models
        self.split_key = split_key
        self.target_col = target_col
        self.base_pipelines = base_pipelines
        self.pipelines_ = {}
        self.trend_pred = trend_pred
        self.y_detrended = y_detrended

    def fit(self, X, y):
        for key, X_part in X.groupby(self.split_key):
            if self.y_detrended is not None:
                y_part = self.y_detrended.loc[X_part.index]
            else:
                y_part = y.loc[X_part.index]

            # save train data for using it in counting rolling features for the test data in the future
            self.base_pipelines[key].train_data = X_part

            pipeline = clone(self.base_pipelines[key])
            pipeline = pipeline.fit(X_part.drop(columns=[self.split_key]), y_part)

            self.pipelines_[key] = pipeline
        return self

    def predict(self, X):
        y_preds = {}
        for split_key_value, X_part in X.groupby(self.split_key):
            pipeline = self.pipelines_[split_key_value]

            # all train set + processed test days (for counting rolling features)
            X_part_all_rows = pipeline.train_data

            # sort dates of X_part for safety
            sorted_dates = X_part[pipeline.date_column].unique().tolist()
            sorted_dates.sort()

            for current_day_number, current_day in enumerate(sorted_dates):
                X_current_day = X_part[X_part[pipeline.date_column] == current_day]
                for lag in pipeline.lags:

                    # filling lags with the predictions of previous test days
                    if current_day_number + 1 > lag:
                        X_current_day['lag_{}'.format(lag)] = y_preds[
                            (split_key_value, sorted_dates[current_day_number - lag])].tolist()

                # filling rolling features for current day
                grp = X_part_all_rows[self.target_col].groupby(X_part_all_rows[pipeline.level])
                for period in pipeline.rolling_days:
                    rolling_slices = [sliding_window_view(v, period) for _, v in grp]

                    for function, aggregate in pipeline.rolling_aggr.items():

                        # passing last 'X_current_day['store_nbr'].nunique()' (54) rolling values to the X_current_day
                        # last 54 values correspond to the current_day data
                        if f'rolling_{period}d_{aggregate}' not in pipeline.drop_columns:
                            X_current_day.loc[:, f'rolling_{period}d_{aggregate}'] = pd.Series(
                                function(rolling_slices, -1).ravel(order='F')).tail(
                                X_current_day['store_nbr'].nunique())

                # making predictions and saving them (also to the X_part_all_rows for using them while
                # counting rolling features for next test days)
                # also adding new test day to the X_part_all_rows

                y_pred = pipeline.predict(X_current_day.drop(columns=[self.split_key]))
                X_part_all_rows = pd.concat([X_part_all_rows, X_current_day])

                # for learning curves (while making predictions on the train set)
                X_part_all_rows = X_part_all_rows.drop_duplicates(
                    subset=[pipeline.date_column, pipeline.level, self.split_key])
                X_part_all_rows.loc[X_current_day.index, 'sales'] = y_pred
                y_preds[(split_key_value, current_day)] = pd.Series(data=y_pred, index=X_current_day.index,
                                                                    name='forecast')

            if pipeline.use_final_metric:
                for current_key in [key for key in y_preds.keys() if key[0] == split_key_value]:
                    y_preds[current_key] = np.exp(y_preds[current_key]) - 1

        y_pred = pd.concat(y_preds.values()).loc[X.index]
        if self.trend_pred is None:
            return y_pred
        else:
            return y_pred + self.trend_pred.loc[y_pred.index]
