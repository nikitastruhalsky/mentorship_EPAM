import numpy as np


def print_cv_test_scores(scores):
    for metric_name, metric_values in scores.items():
        if metric_name.startswith('test_'):
            metric_name = metric_name[len('test_'):]
            if metric_name.startswith('neg_'):
                metric_name = metric_name[len('neg_'):]
                metric_values = -metric_values.copy()

            if metric_name == 'mean_squared_log_error':
                metric_name = 'root_mean_squared_log_error'
                metric_values = np.sqrt(metric_values)

            print(f'{metric_name}: {metric_values.mean():.3f} ± {metric_values.std():.3f}')