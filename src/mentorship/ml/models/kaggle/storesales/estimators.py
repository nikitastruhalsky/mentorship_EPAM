from numpy.lib.stride_tricks import sliding_window_view
from mentorship.ml.models.common import RecursiveTSEstimator
import numpy as np
import pandas as pd


class RecursiveTSEstimatorWithZeroCategories(RecursiveTSEstimator):
    def __init__(self, split_key, target_col, trend_models=None, detrended_categories=None, base_pipelines=None, zero_categories=None,
                 trend_pred=None, y_detrended=None):
        super().__init__(split_key, target_col, base_pipelines)
        if zero_categories is None:
            zero_categories = []
        self.zero_categories = zero_categories
        self.trend_pred = trend_pred
        self.y_detrended = y_detrended
        self.base_pipelines = base_pipelines
        self.detrended_categories = detrended_categories
        self.trend_models = trend_models

    def predict(self, X):
        y_preds = {}
        y_trends = {}
        for split_key_value, X_part in X.groupby(self.split_key):
            if split_key_value in self.zero_categories:
                y_preds[split_key_value] = pd.Series(0, index=X_part.index, name='forecast')
                if self.target_col not in X_part.columns:
                    y_trends[split_key_value] = pd.Series(0, index=X_part.index, name='forecast')
                continue

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
                for period in pipeline.rolling_days:
                    grp = X_part_all_rows[self.target_col].groupby(X_part_all_rows['store_nbr'])
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
                if self.target_col not in X_current_day.columns:
                    if split_key_value in self.detrended_categories:
                        y_trend_pred = self.trend_models[split_key_value][0].predict(
                            pd.DataFrame(self.trend_models[split_key_value][1].iloc[[current_day_number]])
                        )
                        y_trends[(split_key_value, current_day)] = pd.Series(data=[y_trend_pred[0]] * len(X_current_day.index),
                                                                             index=X_current_day.index, name='forecast')
                    else:
                        y_trends[(split_key_value, current_day)] = pd.Series(data=0, index=X_current_day.index,
                                                                             name='forecast')

                y_pred = pipeline.predict(X_current_day.drop(columns=[self.split_key]))
                X_part_all_rows = pd.concat([X_part_all_rows, X_current_day])

                # for learning curves (while making predictions on the train set)
                X_part_all_rows = X_part_all_rows.drop_duplicates(
                    subset=[col for col in [pipeline.date_column, pipeline.level, self.split_key] if col is not None])
                X_part_all_rows.loc[X_current_day.index, self.target_col] = y_pred
                y_preds[(split_key_value, current_day)] = pd.Series(data=y_pred, index=X_current_day.index,
                                                                    name='forecast')

            if pipeline.use_final_metric:
                for current_key in [key for key in y_preds.keys() if key[0] == split_key_value]:
                    y_preds[current_key] = np.exp(y_preds[current_key]) - 1

        y_pred = pd.concat(y_preds.values()).loc[X.index]
        if self.trend_pred is None:
            if self.target_col not in X.columns:
                y_trend = pd.concat(y_trends.values()).loc[X.index]
                return y_pred + y_trend
            else:
                return y_pred
        else:
            y_final = y_pred + self.trend_pred.loc[y_pred.index]
            y_final[y_final < 0] = 0
            return y_final
