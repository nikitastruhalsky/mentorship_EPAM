import numpy as np
from matplotlib import pyplot as plt


def print_cv_test_scores(scores):
    scores_RMSLE = []
    for metric_name, metric_values in scores.items():
        if metric_name.startswith('test_'):
            metric_name = metric_name[len('test_'):]
            if metric_name.startswith('neg_'):
                metric_name = metric_name[len('neg_'):]
                metric_values = -metric_values.copy()

            if metric_name == 'mean_squared_log_error':
                metric_name = 'root_mean_squared_log_error'
                metric_values = np.sqrt(metric_values)
                scores_RMSLE = metric_values

            print(f'{metric_name}: {metric_values.mean():.3f} ± {metric_values.std():.3f}')
    plt.plot(scores_RMSLE)
    plt.xlabel('Fold number')
    plt.ylabel('RMSLE value')


def save_cv_test_scores(scores):
    mean_scores, std_scores = {}, {}
    for metric_name, metric_values in scores.items():
        if metric_name.startswith('test_'):
            metric_name = metric_name[len('test_'):]
            if metric_name.startswith('neg_'):
                metric_name = metric_name[len('neg_'):]
                metric_values = -metric_values.copy()

            if metric_name == 'mean_squared_log_error':
                metric_name = 'root_mean_squared_log_error'
                metric_values = np.sqrt(metric_values)

            mean_scores[metric_name] = metric_values.mean()
            std_scores[metric_name] = metric_values.std()

    return mean_scores, std_scores
