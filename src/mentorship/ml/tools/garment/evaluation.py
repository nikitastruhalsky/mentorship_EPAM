import os
import numpy as np
import pandas as pd
import seaborn as sns
import tempfile
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc


def plot_confusion_matrix(X=None, y=None, modelling_pipeline=None, folds_indices=None, n_folds=4, y_true=None,
                          y_pred=None, save_plot=False):
    if y_true is not None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted labels', fontsize=10)
        plt.ylabel('True labels', fontsize=10)
        plt.title('test set')
        if save_plot:
            fp = tempfile.TemporaryFile(dir=os.getcwd(), suffix='.jpg', delete=False)
            plt.savefig(fp, format='jpg')
            return fp
        return

    plt.figure(figsize=(15, 10))
    for current_fold in range(n_folds):
        plt.subplot(2, 2, current_fold + 1)
        modelling_pipeline.fit(X.loc[folds_indices[current_fold][0], :], y.loc[folds_indices[current_fold][0]])
        y_pred_current_fold = modelling_pipeline.predict(X.loc[folds_indices[current_fold][1], :])
        y_true_current_fold = y.loc[folds_indices[current_fold][1]]
        cm = confusion_matrix(y_true_current_fold, y_pred_current_fold, labels=[0, 1])
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted labels', fontsize=10)
        plt.ylabel('True labels', fontsize=10)
        plt.title(f'{current_fold + 1} fold')

    if save_plot:
        fp = tempfile.TemporaryFile(dir=os.getcwd(), suffix='.jpg', delete=False)
        plt.savefig(fp, format='jpg')
        return fp


def plot_roc_auc(models, X_test, y_test, save_plot=False):
    plt.figure(figsize=(10, 6))
    for current_model in models.keys():
        y_proba = models[current_model].predict(X_test)[1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{current_model} auc: {round(score, 3)}')
        plt.legend()

    if save_plot:
        plt.savefig('roc_auc_plot.png')


def plot_precision_recall_auc(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for m in models.keys():
        proba = models[m].predict(X_test)[1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        score = auc(recall, precision)
        plt.title('Precision-Recall curves for all models')
        plt.plot(recall, precision, lw=2, label=f'{m}  auc: {round(score, 3)}')
    baseline = y_test.mean()
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()


def plot_lift_curve(y_val, y_pred, step=0.1, title='Lift curve'):

    # Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    # Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_val
    aux_lift['predicted'] = 1 - y_pred
    # Order the values for the predicted probability column:
    aux_lift.sort_values('predicted', ascending=False, inplace=True)

    # Create the values that will go into the X axis of our plot
    x_val = np.round(np.arange(step, 1 + step, step), 1)
    # Calculate the ratio of ones in our data
    ratio_zeros = 1 - aux_lift['real'].mean()
    # Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []

    # Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(
            np.ceil(x * len(aux_lift)))  # The ceil function returns the closest integer bigger than our number
        data_here = aux_lift.iloc[:num_data, :]  # ie. np.ceil(1.4) = 2
        ratio_zeros_here = 1 - data_here['real'].mean()
        y_v.append(ratio_zeros_here / ratio_zeros)

    # Plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40, 40)
    axis.plot(x_val, y_v, 'g-', linewidth=3, markersize=5)
    axis.plot(x_val, np.ones(len(x_val)), 'k--', label='baseline')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_cumulative_gain_curve(y_true, y_proba, rare_class_prop, step=0.1, title='Cumulative Gains Curve', ax=None,
                               figsize=None, title_fontsize='large', text_fontsize='medium'):
    x_v = np.round(np.arange(0, 1 + step, step), 1)
    zeros_number = len(y_true[y_true == 0])

    for i, current_proba in enumerate(y_proba):
        aux_lift = pd.DataFrame()
        aux_lift['real'] = y_true
        aux_lift['predicted'] = current_proba
        aux_lift.sort_values('predicted', ascending=False, inplace=True)
        y_v = []

        for x in x_v:
            num_data = int(np.ceil(x * len(aux_lift)))
            data_here = aux_lift.iloc[:num_data, :]
            zeros_number_here = (data_here['real'] == 0).sum()
            y_v.append(zeros_number_here / zeros_number)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(x_v, y_v, lw=2, label=f'Model {i + 1}')

    ax.plot([0, rare_class_prop, 1], [0, 1, 1], 'y--', lw=2, label='Perfect model')
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(x_v, x_v)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')
    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Positive responses, %', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)
