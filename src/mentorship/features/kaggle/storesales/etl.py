import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

DATA_ROOT = Path('data', 'kaggle', 'store-sales-time-series-forecasting')


class ETLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, id_column):
        self.date_column = date_column
        self.id_column = id_column
        self.oil_data = pd.read_csv(DATA_ROOT / 'oil.csv')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.merge(self.oil_data, on=self.date_column, how='left')
        X['dcoilwtico'] = X['dcoilwtico'].fillna(method='ffill')
        X = X.sort_values(by=[self.id_column], ascending=True, ignore_index=True)
        X = X.drop(columns=[self.id_column])
        X['family'] = X['family'].str.lower()
        return X, y
