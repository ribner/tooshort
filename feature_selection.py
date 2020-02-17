from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd


def select_features(X_train, y_train, X_test, prediction_type):
    if (prediction_type == "regression"):
        selector = SelectFromModel(
            estimator=LinearRegression()).fit(X_train, y_train)
        selected_X_train = pd.DataFrame(data=selector.transform(X_train))
        selected_X_test = pd.DataFrame(data=selector.transform(X_test))
        support = selector.get_support()
    if (prediction_type == "classification"):
        selector = SelectFromModel(
            estimator=KNeighborsClassifier()).fit(X_train, y_train)
        selected_X_train = pd.DataFrame(data=selector.transform(X_train))
        selected_X_test = pd.DataFrame(data=selector.transform(X_test))
        support = selector.get_support()

    # restore column names
    if (isinstance(X_train, pd.DataFrame) and len(X_train.columns) == len(support)):
        filtered_columns = []
        for i in range(len(support)):
            if (support[i] == True):
                filtered_columns.append(X_train.columns[i])
        selected_X_train.columns = filtered_columns
    if (isinstance(X_test, pd.DataFrame) and len(X_test.columns) == len(support)):
        filtered_columns = []
        for i in range(len(support)):
            if (support[i] == True):
                filtered_columns.append(X_test.columns[i])
        selected_X_test.columns = filtered_columns
    return selected_X_train, selected_X_test
