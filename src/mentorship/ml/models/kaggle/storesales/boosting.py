from lightgbm import LGBMRegressor
import pdpipe as pdp
from mentorship.ml.models.reg import PositiveRegressor
from mentorship.ml.models.reg import LogTransformRegressor
from mentorship.features.kaggle.storesales.transformers import LagComputer, RollingComputer
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LGBMPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, level=None, drop_columns=None, date_column='date', lags=None,
                 use_rmsle=True, fit_params=None, params=None, rolling_days=None, rolling_aggr=None,
                 predict_negative=False, lagged_feature='sales', rolling_feature='sales'):
        if rolling_aggr is None:
            rolling_aggr = {}
        if rolling_days is None:
            rolling_days = []
        if params is None:
            params = {}
        if lags is None:
            lags = []
        if drop_columns is None:
            drop_columns = []
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.level = level
        self.params = params
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
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        lgbm = LGBMRegressor(**self.params)
        if self.use_rmsle:
            model = LogTransformRegressor(lgbm, fit_params=self.fit_params, predict_negative=self.predict_negative)
        else:
            model = PositiveRegressor(lgbm, fit_params=self.fit_params, predict_negative=self.predict_negative)

        super().__init__(pipeline=pipeline, estimator=model)
