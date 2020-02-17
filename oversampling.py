from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def oversample(X_train, y_train):
    """Function tranforming train data using undersampling and oversampling
    Keyword arguments:
    X_train: pandas df. X train
    y_train: pandas df or list. y train
    Returns:
    os_X_train, os_y_train. As pandas dataframes
    """
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0].ravel()

    # count_dict = {x: y_train.count(x) for x in y_train}
    # y_keys, y_counts = count_dict.keys(), count_dict.values()
    unique, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(unique, counts))
    y_keys, y_counts = count_dict.keys(), count_dict.values()
    sorted_counts = sorted(y_counts, reverse=True)
    highest = sorted_counts[0]
    second_highest = sorted_counts[1]
    ratio = highest / second_highest
    # under sample then oversample
    if (ratio > 4):
        under = RandomUnderSampler(sampling_strategy=0.5)
        over = SMOTE()
        steps = [('u', under), ('o', over)]
        pipeline = Pipeline(steps=steps)
        os_X, os_y = pipeline.fit_resample(X_train, y_train)
    else:
        oversample = SMOTE()
        os_X, os_y = oversample.fit_resample(X_train, y_train)
    return os_X, os_y
