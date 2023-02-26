import numpy as np
import pandas as pd


def splitter(X, n_folds=4, test_size=6):
    folds_indices = []
    for i in range(n_folds):
        date_range_train = X['date'].unique()[test_size:((i - n_folds) * test_size)]
        train_ind = X[X['date'].isin(date_range_train)].index
        date_range_test = X['date'].unique()[((i - n_folds) * test_size):((i - n_folds + 1) * test_size)]
        if i == n_folds - 1:
            date_range_test = X['date'].unique()[((i - n_folds) * test_size):]
        test_ind = X[X['date'].isin(date_range_test)].index
        folds_indices.append((train_ind, test_ind))

    return folds_indices
