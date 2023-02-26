import pdpipe as pdp
from mentorship.ml.models.reg import PositiveRegressor
from mentorship.ml.models.reg import LogTransformRegressor
from mentorship.features.kaggle.storesales.transformers import LagComputer, RollingComputer
from sklearn.linear_model import LinearRegression
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LinearPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, level=None, cols_to_scale=None, cols_to_encode=None, drop_columns=None,
                 date_column='date', lags=None, fit_params=None, use_rmsle=True, rolling_days=None,
                 rolling_aggr=None, predict_negative=False, lagged_feature='sales', rolling_feature='sales'):
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
        self.use_rmsle = use_rmsle
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.predict_negative = predict_negative
        self.lagged_feature = lagged_feature
        self.rolling_feature = rolling_feature

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column, level=self.level, lagged_feature=self.lagged_feature),
            RollingComputer(target_col=self.target_col, rolling_days=self.rolling_days,
                            rolling_aggr=self.rolling_aggr, split_key=self.split_key, level=self.level,
                            rolling_feature=self.rolling_feature),
            pdp.Scale('MinMaxScaler', self.cols_to_scale),
            pdp.OneHotEncode(self.cols_to_encode),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        if self.use_rmsle:
            model = LogTransformRegressor(LinearRegression(), fit_params=self.fit_params, predict_negative=self.predict_negative)
        else:
            model = PositiveRegressor(LinearRegression(), fit_params=self.fit_params, predict_negative=self.predict_negative)

        super().__init__(pipeline=pipeline, estimator=model)
