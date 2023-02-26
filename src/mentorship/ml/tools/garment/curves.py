import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from mentorship.ml.models.common import RecursiveTSEstimator


def plot_learning_curve(estimator, title, X, y, cv=None, scoring=None, negate_scores=True, left_border=0.7):
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=-1,
                                                            error_score='raise',
                                                            train_sizes=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if negate_scores:
        train_scores_mean = np.mean(-train_scores, axis=1)
        test_scores_mean = np.mean(-test_scores, axis=1)
    else:
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.ylim(left_border, 1)
    plt.legend(loc='best')
    return plt


def plot_learning_curver(X, y, tscv_inner, split_key='department', target_col='bigger_than_targeted',
                         base_pipelines=None, class_weight=None, left_border=0.7):
    """
    Function for building learning curves for each department (on the train set of the first cv fold)
    Metric: 'f1'
    """

    for i, current_department in enumerate(X[split_key].unique()):
        X_copy, y_copy = X.copy(), y.copy()
        gscv = []
        for (train_dates, val_dates) in tscv_inner:
            train_ind = X_copy[(X_copy['date'].isin(train_dates)) & (X_copy['department'] == current_department)].index
            test_ind = X_copy[(X_copy['date'].isin(val_dates)) & (X_copy['department'] == current_department)].index
            gscv.append((train_ind, test_ind))

        plt.figure(figsize=(10, 5))

        # learning curves (f1)
        base_pipeline = {current_department: base_pipelines[current_department]}
        modelling_pipeline = RecursiveTSEstimator(base_pipelines=base_pipeline, split_key=split_key,
                                                  target_col=target_col, learning_curve=True, class_weight=class_weight)
        plot_learning_curve(modelling_pipeline,
                            title=f'{current_department}: Learning Curves (Logistic Regression), F1',
                            X=X_copy, y=y_copy, cv=gscv, scoring='f1', negate_scores=False, left_border=left_border)
        plt.show()
