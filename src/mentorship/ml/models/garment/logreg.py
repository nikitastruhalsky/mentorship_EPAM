import pdpipe as pdp
from mentorship.ml.models.clf import Classifier
from mentorship.features.garment.transformers import OverTimeComputer, RollingComputer, LagComputer
from sklearn.linear_model import LogisticRegression
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LogRegPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, level=None, cols_to_scale=None, cols_to_encode=None, drop_columns=None,
                 date_column='date', lags=None, fit_params=None, rolling_days=None, cols_to_standard_scale=None,
                 rolling_aggr=None, predict_proba=False, params=None, lagged_feature='bigger_than_targeted',
                 rolling_col=None):
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
        if params is None:
            params = {}
        self.cols_to_scale = cols_to_scale
        self.cols_to_encode = cols_to_encode
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.level = level
        self.use_rmsle = False    # crutch
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.predict_proba = predict_proba
        self.params = params
        self.lagged_feature = lagged_feature
        self.cols_to_standard_scale = cols_to_standard_scale
        self.rolling_col = rolling_col

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column, level=self.level, lagged_feature=self.lagged_feature),
            RollingComputer(level=self.level, rolling_col=self.rolling_col, rolling_days=self.rolling_days,
                            rolling_aggr=self.rolling_aggr),
            OverTimeComputer(level=self.level, date_column=self.date_column),
            pdp.Scale('MinMaxScaler', self.cols_to_scale),
            pdp.Scale('StandardScaler', self.cols_to_standard_scale),
            pdp.OneHotEncode(self.cols_to_encode),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        model = Classifier(LogisticRegression(**self.params), fit_params=self.fit_params, predict_proba=self.predict_proba)

        super().__init__(pipeline=pipeline, estimator=model)
