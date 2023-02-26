from lightgbm import LGBMClassifier
import pdpipe as pdp
from mentorship.ml.models.clf import Classifier
from mentorship.features.garment.transformers import LagComputer, RollingComputer
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LGBMPipelineClf(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, level=None, drop_columns=None, date_column='date', lags=None,
                 fit_params=None, params=None, rolling_days=None, rolling_aggr=None, rolling_col='bigger_than_targeted',
                 predict_proba=False, cols_to_encode=None, lagged_feature='bigger_than_targeted'):
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
        if cols_to_encode is None:
            cols_to_encode = []
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.level = level
        self.use_rmsle = False    # crutch
        self.params = params
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr
        self.predict_proba = predict_proba
        self.cols_to_encode = cols_to_encode
        self.lagged_feature = lagged_feature
        self.rolling_col = rolling_col

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column, level=self.level, lagged_feature=self.lagged_feature),
            RollingComputer(level=self.level, rolling_col=self.rolling_col, rolling_days=self.rolling_days,
                            rolling_aggr=self.rolling_aggr),
            pdp.Encode(self.cols_to_encode),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        model = Classifier(LGBMClassifier(**self.params), fit_params=self.fit_params, predict_proba=self.predict_proba)

        super().__init__(pipeline=pipeline, estimator=model)
