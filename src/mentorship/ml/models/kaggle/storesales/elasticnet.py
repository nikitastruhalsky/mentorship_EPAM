import pdpipe as pdp
from pdpipe.skintegrate import PdPipelineAndSklearnEstimator
from mentorship.ml.models.reg import PositiveRegressor
from sklearn.linear_model import ElasticNet


class PipelineElasticNetV1(PdPipelineAndSklearnEstimator):
    def __init__(self, num_columns, cat_columns, alpha, l1_ratio, date_column='date'):
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.date_column = date_column

        pipeline = pdp.PdPipeline([
            pdp.Scale('MinMaxScaler', self.num_columns),
            pdp.OneHotEncode(self.cat_columns),
            pdp.ColDrop([self.date_column, 'family']),
        ])
        model = PositiveRegressor(ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio))
        super().__init__(pipeline=pipeline, estimator=model)
