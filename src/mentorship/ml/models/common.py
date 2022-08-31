from sklearn.base import clone


class SplitPipeline:
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.pipelines_ = {}

    def fit(self, X, y):
        for family in X['family'].unique():
            indexer = X[X['family'] == family].index
            X_part = X[X['family'] == family]
            y_part = y.loc[indexer]

            pipeline = clone(self.base_pipeline)
            pipeline = pipeline.fit(X_part, y_part)

            self.pipelines_[family] = pipeline
        return self

    def predict(self, X):
        X.loc[:, 'forecast'] = 0
        for family in X['family'].unique():
            X_part = X[X['family'] == family]
            X_part = X_part.drop('forecast', axis=1)
            pipeline = self.pipelines_[family]
            X.loc[X_part.index, 'forecast'] = pipeline.predict(X_part)

        y_pred = X['forecast']
        X = X.drop(columns='forecast')
        return y_pred

    def get_params(self, deep=True):
        return {'base_pipeline': self.base_pipeline}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
