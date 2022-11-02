from lightgbm import LGBMRegressor
import pdpipe as pdp
from mentorship.ml.models.reg import PositiveRegressor
from mentorship.features.transformers import LagComputer
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator


class LGBMPipeline(PdPipelineAndSklearnEstimator):
    def __init__(self, split_key, target_col, drop_columns=[], date_column='date', lags=[], use_final_metric=True,
                 fit_params={}, params={}, rolling_days=[], rolling_aggr={}, train_data=None):
        self.date_column = date_column
        self.drop_columns = drop_columns
        self.lags = lags
        self.split_key = split_key
        self.target_col = target_col
        self.fit_params = fit_params
        self.params = params
        self.use_final_metric = use_final_metric
        self.train_data = train_data
        self.rolling_days = rolling_days
        self.rolling_aggr = rolling_aggr

        pipeline = pdp.PdPipeline([
            LagComputer(target_col=self.target_col, lags=self.lags, split_key=self.split_key,
                        date_column=self.date_column),
            pdp.ColDrop(self.drop_columns + [self.target_col, self.date_column], errors='ignore'),
        ])

        lgbm = LGBMRegressor(**self.params)
        model = PositiveRegressor(lgbm, fit_params=self.fit_params)

        super().__init__(pipeline=pipeline, estimator=model)
