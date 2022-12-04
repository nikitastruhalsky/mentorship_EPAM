from sklearn.base import clone
import optuna
import numpy as np
from sklearn.model_selection import cross_validate
from mentorship.ml.models.common import RecursiveTSEstimator
from mentorship.ml.models.kaggle.storesales.estimators import RecursiveTSEstimatorWithZeroCategories
from mentorship.ml.cv.util import format_cv_test_scores
from mentorship.ml.models.kaggle.storesales.boosting import LGBMPipeline


def objective(trial, X, y, base_pipeline, tscv_inner, zero_categories, split_key='family', target_col='sales',
              scoring='neg_mean_squared_log_error'):
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 200, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_data_in_leaf', 10, 510, step=50),
        'reg_alpha': trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
        'min_split_gain': trial.suggest_float('min_gain_to_split', 0, 15),
    }

    base_pipeline.params = param_grid
    base_pipeline = {X[split_key].unique()[0]: clone(base_pipeline)}
    if zero_categories is not []:
        modelling_pipeline = RecursiveTSEstimatorWithZeroCategories(base_pipelines=base_pipeline, split_key=split_key,
                                                                    target_col=target_col, zero_categories=zero_categories)
    else:
        modelling_pipeline = RecursiveTSEstimator(base_pipelines=base_pipeline, split_key=split_key, target_col=target_col)

    scores = cross_validate(modelling_pipeline, X, y, cv=tscv_inner, scoring=[scoring],
                            return_estimator=True, error_score='raise', n_jobs=-1)

    return np.mean(format_cv_test_scores(scores, save_scores=True, print_scores=False)[0]['root_mean_squared_error'])


def tune_hyperparams(X, y, split_key='family', target_col='sales', drop_columns=None, tscv_inner=None,
                     lags=None, rolling_days=None, rolling_aggr=None, level=None, zero_categories=None,
                     predict_negative=False, use_final_metric=True, scoring='neg_mean_squared_log_error'):
    """
    Nested cross-validation on the first fold train set only (to save time);
    these best params will be used in the cross validation later
    """
    if zero_categories is None:
        zero_categories = []
    if level is None:
        level = []
    if lags is None:
        lags = []
    if rolling_days is None:
        rolling_days = []
    if rolling_aggr is None:
        rolling_aggr = {}
    if drop_columns is None:
        drop_columns = {}
    best_params = {}
    for current_split_key in X[split_key].unique():
        print(current_split_key)

        X_current_split_key = X[X[split_key] == current_split_key]
        y_current_split_key = y.loc[X_current_split_key.index]

        fit_params = {'categorical_feature': [0]}
        if drop_columns == {}:
            drop_columns = {current_split_key: [] for current_split_key in X[split_key].unique()}

        if 'store_nbr' in drop_columns[current_split_key]:
            fit_params = {}

        base_pipeline = LGBMPipeline(lags=lags, split_key=split_key, target_col=target_col, fit_params=fit_params,
                                     drop_columns=drop_columns[current_split_key], rolling_days=rolling_days,
                                     rolling_aggr=rolling_aggr, level=level, predict_negative=predict_negative,
                                     use_final_metric=use_final_metric)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', study_name=f'LGBM Regressor for {current_split_key}')
        func = lambda trial: objective(trial, X_current_split_key, y_current_split_key, base_pipeline,
                                       target_col=target_col, zero_categories=zero_categories, tscv_inner=tscv_inner,
                                       scoring=scoring)
        study.optimize(func, n_trials=50)

        best_params[current_split_key] = study.best_params
        print(f'best_params: {best_params[current_split_key]}')
        print(f'best_value: {study.best_value} \n')
    return best_params
