import pdpipe as pdp
from mentorship.ml.models.reg import PositiveRegressor
from mentorship.ml.models.reg import LogTransformRegressor
from mentorship.features.transformers import LagComputer
from sklearn.linear_model import LinearRegression
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LinearPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, level=None, cols_to_scale=None, cols_to_encode=None, drop_columns=None,
                 date_column='date', lags=None, fit_params=None, use_final_metric=True, rolling_days=None,
                 rolling_aggr=None, train_data=None, predict_negative=False):
        if rolling_aggr is None:
            rolling_aggr = {}
        if rolling_days is None:
            rolling_days = []
        if fit_params is None:
            fit_params = {}
        if lags is None:
            lags = []
        if drop_columns is None:
            drop_columns = []
        if cols_to_encode is None:
            cols_to_encode = []
        if cols_to_scale is None:
            cols_to_scale = []
        self.cols_to_scale = cols_to_scale
        self.cols_to_encode = cols_to_encode
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.level = level
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.use_final_metric = use_final_metric
        self.train_data = train_data
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.predict_negative = predict_negative

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column, level=self.level),
            pdp.Scale('MinMaxScaler', self.cols_to_scale),
            pdp.OneHotEncode(self.cols_to_encode),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        if use_final_metric:
            model = LogTransformRegressor(LinearRegression(), fit_params=self.fit_params, predict_negative=self.predict_negative)
        else:
            model = PositiveRegressor(LinearRegression(), fit_params=self.fit_params, predict_negative=self.predict_negative)

        super().__init__(pipeline=pipeline, estimator=model)
