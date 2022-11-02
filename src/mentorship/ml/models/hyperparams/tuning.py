from sklearn.base import clone
import optuna
import time
import numpy as np
from sklearn.model_selection import cross_validate
from mentorship.ml.models.common import RecursiveTSEstimator
from mentorship.ml.cv.util import save_cv_test_scores
from mentorship.ml.models.kaggle.storesales.boosting import LGBMPipeline


def objective(trial, X, y, base_pipeline, tscv_inner):
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 200, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'num_leaves': trial.suggest_int('num_leaves', 5, 1005, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_data_in_leaf', 10, 510, step=50),
        'reg_alpha': trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
        'min_split_gain': trial.suggest_float('min_gain_to_split', 0, 15),
    }

    base_pipeline.params = param_grid
    base_pipeline = {X['family'].unique()[0]: clone(base_pipeline)}
    modelling_pipeline = RecursiveTSEstimator(base_pipelines=base_pipeline, split_key='family', target_col='sales')
    scores = cross_validate(modelling_pipeline, X, y, cv=tscv_inner, scoring=['neg_mean_squared_log_error'],
                            return_estimator=True, error_score='raise', n_jobs=-1)

    return np.mean(save_cv_test_scores(scores)[0]['root_mean_squared_log_error'])


def hyperparams_tuning(X, y, split_key='family', target_col='sales', drop_columns={}, tscv_inner=None,
                       lags=[], rolling_days=[], rolling_aggr={}):
    """
    Nested cross-validation on the first fold train set only (to save time);
    these best params will be used in the cross validation later
    """
    best_params = {family: {} for family in X[split_key].unique()}
    for current_family in X[split_key].unique():
        start = time.time()
        print(current_family)

        X_current_family = X[X[split_key] == current_family]
        y_current_family = y.loc[X_current_family.index]

        fit_params = {'categorical_feature': [0]}
        if drop_columns == {}:
            drop_columns = {family: [] for family in X[split_key].unique()}

        if 'store_nbr' in drop_columns[current_family]:
            fit_params = {}

        base_pipeline = LGBMPipeline(lags=lags, split_key=split_key, target_col=target_col, fit_params=fit_params,
                                     drop_columns=drop_columns[current_family], rolling_days=rolling_days,
                                     rolling_aggr=rolling_aggr)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', study_name='LGBM Regressor')
        func = lambda trial: objective(trial, X_current_family, y_current_family, base_pipeline, tscv_inner=tscv_inner)
        study.optimize(func, n_trials=50)

        end = time.time()
        print(f'Tuning took {end - start} seconds.')

        best_params[current_family] = study.best_params
        print(f'best_params: {best_params[current_family]}')
        print(f'best_value: {study.best_value} \n')
    return best_params
