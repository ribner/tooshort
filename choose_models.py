import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, SGDRegressor, Ridge, ElasticNet, Lasso
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier


def choose_models(y, prediction_type):
    """Function giving you suggested sklearn models based on prediction type and size of data
    Keyword arguments:
    y = pdDataframe or list. Target values
    prediction_type: str. String representing the type of predictive model. Either "regression" or "classification"
    Returns:
    List of sklearn model classes
    """
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].ravel()
    n_samples = len(y)
    if (prediction_type == "regression"):
        if (n_samples > 100000):
            return [
                LinearRegression,
                SGDRegressor,
                SVR,
                Ridge,
                ElasticNet,
                Lasso,
                LinearSVC
            ]
        # SGD regressor needs more than 100k samples
        else:
            return [
                LinearRegression,
                SVR,
                Ridge,
                ElasticNet,
                Lasso,
                LinearSVC
            ]
    if (prediction_type == "classification"):
        if (n_samples > 100000):
            return [
                # SVC,
                RandomForestClassifier,
                KNeighborsClassifier,
                SGDClassifier
            ]
        else:
            return [
                # SVC,
                RandomForestClassifier,
                KNeighborsClassifier,
            ]
    raise Exception("prediction_type must be categorical or regression")
