import pandas as pd
from pathlib import Path

DATA_ROOT = Path('data', 'kaggle', 'store-sales-time-series-forecasting')


def make_submission_file(test_data, model, output_path):
    submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    submission['sales'] = model.predict(test_data)
    submission.to_csv(DATA_ROOT / output_path, index=False)
