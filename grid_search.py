from too_short import get_param_grid
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
import pandas as pd


def search(models, X, y, X_test=None, y_test=None, scoring=None):
    results = {}
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].ravel()
    for model in models:
        param_grid = get_param_grid(model)
        total_params = sum(map(lambda x: len(x), param_grid.keys()))
        if (total_params > 17):
            search = RandomizedSearchCV(
                model(), param_grid, n_jobs=-1, scoring=scoring)
        else:
            search = GridSearchCV(model(), param_grid,
                                  n_jobs=-1, scoring=scoring)
        search.fit(X, y)
        test_score = 'N/A'
        if (isinstance(X_test, pd.DataFrame)):
            test_score = search.score(X_test, y_test)
        results[model.__name__] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'test_score': test_score
        }
    return results
