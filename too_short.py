import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_boston, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def get_param_grid(model):
    """Function providing a hyperparam grid to be used in sklearn hyperparameter optimizatoin
    Keyword arguments:
    SKlearn model, .__name__ should equal one of the following 
    SVC
    RandomForest
    KNN
    SGD
    Kmeans
    SGDRegressor
    SVR
    Ridge
    ElasticNet
    Lasso
    BayesianGaussianMixture
    LinearSVC
    Returns:
    Dictionary containing sklearn params as keys and list of param options as values
    Example:
    get_param_grid(LinearRegression)
    >>{'normalize': [True, False]}
    """
    name = model.__name__
    if (name == "LogisticRegression"):
        penalty = ['l1', 'l2']
        C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        class_weight = [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6},
                        {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}]
        solver = ['liblinear', 'saga']

        return dict(penalty=penalty,
                    C=C,
                    class_weight=class_weight,
                    solver=solver)
    if (name == "SGDClassifier"):
        loss = ['hinge', 'log', 'modified_huber',
                'squared_hinge', 'perceptron']
        penalty = ['l1', 'l2', 'elasticnet']
        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
        class_weight = [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6},
                        {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}]
        eta0 = [1, 10, 100]

        return dict(loss=loss,
                    penalty=penalty,
                    alpha=alpha,
                    learning_rate=learning_rate,
                    class_weight=class_weight,
                    eta0=eta0)
    if (name == "RandomForestClassifier"):
        n_estimators = [int(x)
                        for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        return dict(n_estimators=n_estimators,
                    max_features=max_features,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap)
    if (name == "LinearRegression"):
        normalize = [True, False]
        return dict(
            normalize=normalize
        )
    if (name == "KNeighborsClassifier"):
        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]
        return dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    if (name == "SVC"):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        return dict(C=Cs,
                    gamma=gammas,
                    # TODO - add kernels?
                    kernel='rbf'
                    )
    if (name == "SVR"):
        kernel = ['linear', 'rbf', 'poly']
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        epsilon = [0.1, 0.2, 0.5, 0.3]
        return dict(kernel=kernel,
                    C=Cs,
                    gamma=gammas,
                    epsilon=epsilon)
    if (name == "SGDRegressor"):
        alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        loss = ['squared_loss', 'huber', 'epsilon_insensitive',
                'squared_epsilon_insensitive']
        penalty = ['l1', 'l2']
        epsilon = [0.1, 0.2, 0.5, 0.3]
        return dict(alpha=alpha,
                    loss=loss,
                    pentalty=penalty,
                    epsilon=epsilon
                    )
    if (name == "KMeans"):
        n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        return dict(
            n_clusters=n_clusters
        )
    if (name == "AdaBoostClassifier"):
        base_estimator__criterion = ["gini", "entropy"]
        base_estimator__splitter = ["best", "random"]
        return dict(
            base_estimator__criterion=base_estimator__criterion,
            base_estimator__splitter=base_estimator__splitter
        )
    if (name == "GradientBoostingClassifier"):
        max_depth = range(5, 16, 2)
        min_samples_split = range(200, 1001, 200)
        return dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    if (name == "Ridge"):
        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        return dict(
            alpha=alpha
        )
    if (name == "ElasticNet"):
        max_iter = [1, 5, 10]
        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        l1_ratio = np.arange(0.0, 1.0, 0.1)
        return dict(
            max_iter=max_iter,
            alpha=alpha,
            l1_ratio=l1_ratio
        )
    if (name == "Lasso"):
        alpha = [0.02, 0.024, 0.025, 0.026, 0.03]
        return dict(
            alpha=alpha
        )
    if (name == "BayesianGaussianMixture"):
        covariance_type = ['full', 'tied', 'diag', 'spherical']
        weight_concentration_prior_type = ['dirichlet_process',
                                           'dirichlet_distribution']
        dict(
            covariance_type=covariance_type,
            weight_concentration_prior_type=weight_concentration_prior_type
        )
    if (name == "LinearSVC"):
        dual = [True, False]
        C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        return dict(
            dual=dual,
            C=C
        )
    print(name)
    raise Exception("Model not supported.")


def preproc(X, OHE=[], standard_scale=[], numerical_impute=[], categegorical_impute=[], label_encode={}):
    """Prerocesses data frames, including onehot encoding, scaling, and imputation, and label encoding
    Keyord arguments:
    X: list. List of pandas data frame type. Can be one data frame, or multiple of same structure (i.e. train set and test test). First element must be train set. - Required
    OHE: Array of columns to be processed with sklearn OneHotEncoder, this accepts non numerical categorical rows without need for label encoding. - Default []
    standard_scale: list. List of columns to be processes with standard scalar. - Defualt []
    numerical_impute: list. list of column names that should be imputed using mean method. - Default []
    categorical_impute: list. list of column names that should be imputed to 'missing'. - Default []
    label_encode: dict. Keys in the dict should be the column names to transform, the values should be lists that
    contain the various values in the column, the order of the values will determine the encoding (1st element will be 0 etc.). - Default {}
    Returns:
    List of processed pandas dataframes
    """
    transformer = ColumnTransformer(
        transformers=[
            ('cat_imputer',
             SimpleImputer(strategy='constant', fill_value='missing'),
             categegorical_impute
             ),
            ('num_imputer',
             SimpleImputer(strategy='median'),
             numerical_impute
             ),
            ("one_hot",
             OneHotEncoder(),
             OHE
             ),
            ("standard_scalar",
             StandardScaler(),
             standard_scale
             )
        ],
        remainder='passthrough'  # donot apply anything to the remaining columns
    )
    column_names = X[0].columns
    # fit transformation to the train X set
    transformer.fit(X[0])
    # fetch newly created ohe transformed columns
    if (len(OHE) > 0):
        ohe_columns = transformer.named_transformers_.one_hot.get_feature_names(
            OHE)
    else:
        ohe_columns = []
    # get rid of original OHE columns
    column_names = list(filter(lambda x: x not in OHE, column_names))
    # reset column names with newly created ohe column names
    column_names = np.append(ohe_columns, column_names)
    # transform test and validation sets if they are included
    for i in range(0, len(X)):
        X[i] = pd.DataFrame(data=transformer.transform(
            X[i]), columns=column_names)
    # label encoding tranformation
    le_columns = label_encode.keys()
    for column in le_columns:
        le = LabelEncoder()
        le.fit(label_encode[column])
        for df in X:
            df[column] = le.transform(df[column])
    return X
