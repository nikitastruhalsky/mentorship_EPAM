import pdpipe as pdp
from mentorship.ml.models.reg import PositiveRegressor
from mentorship.features.transformers import LagComputer
from sklearn.linear_model import LinearRegression
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LinearPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, cols_to_scale=[], cols_to_encode=[], drop_columns=[], date_column='date',
                 lags=[], fit_params={}, use_final_metric=True, rolling_days=[], rolling_aggr={}, train_data=None):
        self.cols_to_scale = cols_to_scale
        self.cols_to_encode = cols_to_encode
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.use_final_metric = use_final_metric
        self.train_data = train_data
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column),
            pdp.Scale('MinMaxScaler', self.cols_to_scale),
            pdp.OneHotEncode(self.cols_to_encode),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        model = PositiveRegressor(LinearRegression(), fit_params=self.fit_params)

        super().__init__(pipeline=pipeline, estimator=model)
