from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def grid_search(X, y, model, params, fit_params=None, cv=None, scoring='f1'):
    if fit_params is None:
        fit_params = {}
    gs = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=-1, scoring=scoring)
    gs.fit(X, y, **fit_params)
    return gs.best_params_


def tune_hyperparams(X, y, model, params, fit_params, drop_columns, inner_folds, scoring='f1'):
    best_params = {}
    for department in X['department'].unique():
        X_train_copy, y_train_copy = X.copy(), y.copy()
        X_train_copy['day'] = LabelEncoder().fit_transform(X_train_copy['day'])

        gscv = []
        for (train_dates, val_dates) in inner_folds:
            train_ind = X_train_copy[(X_train_copy['date'].isin(train_dates)) & (X_train_copy['department'] == department)].index
            test_ind = X_train_copy[(X_train_copy['date'].isin(val_dates)) & (X_train_copy['department'] == department)].index
            gscv.append((train_ind, test_ind))

        drop_cols = drop_columns[department]
        drop_cols.extend(['date', 'bigger_than_targeted', 'department'])
        X_train_copy = X_train_copy.drop(columns=drop_cols)

        best_params[department] = grid_search(X_train_copy, y_train_copy, model, params, cv=gscv, scoring=scoring,
                                              fit_params=fit_params)
    return best_params
