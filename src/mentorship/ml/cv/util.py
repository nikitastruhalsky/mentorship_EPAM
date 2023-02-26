import numpy as np
from matplotlib import pyplot as plt


def format_cv_test_scores(scores, metrics_to_plot=None, save_scores=False, print_scores=True, save_train_scores=False):
    if metrics_to_plot is None:
        metrics_to_plot = []
    mean_train_scores, mean_test_scores, std_scores = {}, {}, {}
    for metric_name, metric_values in scores.items():
        if metric_name.startswith('test_'):
            metric_name = metric_name[len('test_'):]
            if metric_name.startswith('neg_'):
                metric_name = metric_name[len('neg_'):]
                metric_values = -metric_values.copy()

            if metric_name == 'mean_squared_log_error':
                metric_name = 'root_mean_squared_log_error'
                metric_values = np.sqrt(metric_values)

            mean_test_scores[metric_name] = np.mean(metric_values)
            std_scores[metric_name] = np.std(metric_values)

            if print_scores:
                print(f'{metric_name}: {np.mean(metric_values):.3f} ± {np.std(metric_values):.3f}')

            if metric_name in metrics_to_plot:
                plt.figure(figsize=(5, 4))
                plt.plot(metric_values, marker='o')
                plt.xlabel('Fold number')
                plt.ylabel(f'{metric_name} value')

        if metric_name.startswith('train_'):
            metric_name = metric_name[len('train_'):]
            if metric_name.startswith('neg_'):
                metric_name = metric_name[len('neg_'):]
                metric_values = -metric_values.copy()

            if metric_name == 'mean_squared_log_error':
                metric_name = 'root_mean_squared_log_error'
                metric_values = np.sqrt(metric_values)

            mean_train_scores[metric_name] = np.mean(metric_values)

    if save_scores:
        if save_train_scores:
            return mean_train_scores, mean_test_scores, std_scores
        else:
            return mean_test_scores, std_scores
