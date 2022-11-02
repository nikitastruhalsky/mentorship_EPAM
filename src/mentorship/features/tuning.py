import numpy as np
import lightgbm as lgb
from boruta import BorutaPy


def boruta_features_tuning(X, y, split_key='family'):
    """
    Choosing best features with Boruta (on the train set of the first cv fold)
    """
    best_features = {family: {} for family in X[split_key].unique()}
    for current_family in X[split_key].unique():
        print(current_family)
        X_current_family = X[X[split_key] == current_family].drop(columns=[split_key])
        y_current_family = y.loc[X_current_family.index]
        model = lgb.LGBMRegressor()
        boruta = BorutaPy(estimator=model, n_estimators=100, max_iter=100, verbose=2)
        boruta.fit(np.array(X_current_family), np.array(y_current_family))
        best_features[current_family]['green_area'] = X_current_family.columns[boruta.support_].to_list()
        best_features[current_family]['blue_area'] = X_current_family.columns[boruta.support_weak_].to_list()
        print('\n')
    return best_features
