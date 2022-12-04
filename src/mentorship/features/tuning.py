import numpy as np
import lightgbm as lgb
from boruta import BorutaPy


def boruta_features_tuning(X, y, split_key='family', model=lgb.LGBMRegressor(), n_estimators=100, max_iter=100,
                           verbose=-1):
    """
    Choosing best features with Boruta (on the train set of the first cv fold)
    """
    best_features = {split_key_value: {} for split_key_value in X[split_key].unique()}
    for current_split_key_value in X[split_key].unique():
        print(current_split_key_value)
        X_current_split_key_value = X[X[split_key] == current_split_key_value].drop(columns=[split_key])
        y_current_split_key_value = y.loc[X_current_split_key_value.index]
        boruta = BorutaPy(estimator=model, n_estimators=n_estimators, max_iter=max_iter, verbose=verbose)
        boruta.fit(np.array(X_current_split_key_value), np.array(y_current_split_key_value))
        best_features[current_split_key_value]['green_area'] = X_current_split_key_value.columns[boruta.support_].to_list()
        best_features[current_split_key_value]['blue_area'] = X_current_split_key_value.columns[boruta.support_weak_].to_list()
        print('\n')
    return best_features
