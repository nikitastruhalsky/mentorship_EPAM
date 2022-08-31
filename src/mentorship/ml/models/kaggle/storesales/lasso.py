import pdpipe as pdp
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator
from mentorship.ml.models.reg import PositiveRegressor
from sklearn.linear_model import Lasso


class PipelineLassoV1(PdPipelineAndSklearnEstimator):
    def __init__(self, num_columns, cat_columns, best_params, date_column='date'):
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.best_params = best_params
        self.date_column = date_column

        pipeline = pdp.PdPipeline([
            pdp.Scale('MinMaxScaler', self.num_columns),
            pdp.OneHotEncode(self.cat_columns),
            pdp.ColDrop([self.date_column, 'family']),
        ])
        model = PositiveRegressor(Lasso(alpha=best_params))
        super().__init__(pipeline=pipeline, estimator=model)
