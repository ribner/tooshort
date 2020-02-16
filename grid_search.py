from too_short import get_param_grid
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pandas as pd


def search(X, y, models):
    master_param_grid = {}
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].ravel()
    for model in models:
        param_grid = get_param_grid(model)
        param_grid_keys = param_grid.keys()
        for param_key in param_grid_keys:
            master_param_grid[f"{model.__name__}__{param_key}"] = param_grid[param_key]
    pipeline_steps = list(map(lambda x: (x.__name__, x()), models))
    pipe = Pipeline(steps=pipeline_steps)
    search = GridSearchCV(pipe, master_param_grid, n_jobs=-1)
    search.fit(X, y)
    print(search.best_score_)
    print(search.best_params_)
