import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from mentorship.ml.cv.split import DateTimeSeriesSplit
from mentorship.ml.models.kaggle.storesales.boosting import LGBMPipeline
from mentorship.ml.models.common import RecursiveTSEstimator


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, scoring=None,
                        train_sizes=None, negate_scores=True):

    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, scoring=scoring, cv=cv,
                                                            train_sizes=train_sizes, n_jobs=n_jobs, error_score='raise')
    if negate_scores:
        train_scores_mean = np.mean(-train_scores, axis=1)
        test_scores_mean = np.mean(-test_scores, axis=1)
    else:
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


def plot_learning_curver(X, y, tscv_inner, train_sizes, split_key='family', target_col='sales', lags=None, params=None):
    """
    Function for building learning curves for each family (on the train set of the first cv fold)
    Metric: MSLE
    """

    # train set formation
    if params is None:
        params = {}
    if lags is None:
        lags = []
    splitter = DateTimeSeriesSplit()
    train_indices = next(splitter.split(X, y))[0]
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]

    if params == {}:
        params = {family: {} for family in X[split_key].unique()}

    for i, current_family in enumerate(X_train[split_key].unique()):
        X_current_family = X_train[X_train[split_key] == current_family]
        y_current_family = y_train.loc[X_current_family.index]

        plt.figure(figsize=(10, 5))

        # learning curves (MSLE)
        fit_params = {'categorical_feature': [0]}
        base_pipeline = {current_family: LGBMPipeline(lags=lags, split_key=split_key, target_col=target_col,
                                                      fit_params=fit_params, params=params[current_family])}
        modelling_pipeline = RecursiveTSEstimator(base_pipelines=base_pipeline, split_key=split_key,
                                                  target_col=target_col)
        plot_learning_curve(modelling_pipeline,
                            title=f'{current_family}: Learning Curves (default LGBM Regressor), RMSLE',
                            X=X_current_family, y=y_current_family, cv=tscv_inner, scoring='neg_mean_squared_log_error',
                            n_jobs=-1, train_sizes=train_sizes)
        plt.show()
