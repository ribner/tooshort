import unittest

from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.svm import SVC
from choose_models import choose_models
from too_short import preproc
from too_short import get_param_grid
from grid_search import search
from sklearn.neighbors import KNeighborsClassifier


def get_iris():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y


class TestEndToEnd(unittest.TestCase):
    def testCatSmallEndToEnd(self):
        X, y = get_iris()
        result = preproc([X],
                         standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                         'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                         'proanthocyanins', 'color_intensity', 'hue',
                                         'od280/od315_of_diluted_wines', 'proline'])
        X = result[0]
        models = choose_models(y, prediction_type="classification")
        result = search(X, y, models)


class TestGridSearch(unittest.TestCase):

    def test_basic_grid_search(self):
        X, y = get_iris()
        result = preproc([X],
                         standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                         'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                         'proanthocyanins', 'color_intensity', 'hue',
                                         'od280/od315_of_diluted_wines', 'proline'])
        X = result[0]
        result = search(X, y, [KNeighborsClassifier])


class TestChooseModels(unittest.TestCase):
    def test_returns_regression_models_small_samples(self):
        iris = load_iris()
        y = pd.DataFrame(iris.target)
        y.columns = ["target"]
        result = choose_models(y, prediction_type="regression")
        self.assertIn(LinearRegression, result)
        self.assertNotIn(SGDRegressor, result)

    def test_returns_regression_models_many_samples(self):
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        result = choose_models(y, prediction_type="regression")
        self.assertIn(LinearRegression, result)
        self.assertIn(SGDRegressor, result)

    def test_returns_classification_models_small_samples(self):
        iris = load_iris()
        y = pd.DataFrame(iris.target)
        y.columns = ["target"]
        result = choose_models(y, prediction_type="classification")
        self.assertIn(SVC, result)
        self.assertNotIn(SGDClassifier, result)

    def test_returns_classification_models_many_samples(self):
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        result = choose_models(y, prediction_type="classification")
        self.assertIn(SVC, result)
        self.assertIn(SGDClassifier, result)


class TestGetHyperParamGrids(unittest.TestCase):
    def test_returns_linear_regression_params(self):
        lr_params = {
            'normalize': [True, False]
        }
        result = get_param_grid(LinearRegression)
        self.assertEqual(result, lr_params)


class TestPreproc(unittest.TestCase):
    def test_does_not_alter_original_df(self):
        wine = load_wine()
        X = pd.DataFrame(wine.data)
        X.columns = wine.feature_names
        y = pd.DataFrame(wine.target)
        y.columns = ['TARGET']
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        X_copy = X.copy()
        preproc([X], OHE=np.array(
            ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
            label_encode={
            'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
        },
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        assert_frame_equal(X, X_copy)

    def test_create_ohe(self):
        wine = load_wine()
        X = pd.DataFrame(wine.data)
        X.columns = wine.feature_names
        y = pd.DataFrame(wine.target)
        y.columns = ['TARGET']
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        result = preproc([X], OHE=np.array(
            ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
            label_encode={
            'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
        },
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        result_df = result[0]
        self.assertCountEqual(result_df.columns[0:11], ['A_FAKE_CAT_0', 'A_FAKE_CAT_1', 'A_FAKE_CAT_2', 'A_FAKE_CAT_3',
                                                        'B_FAKE_CAT_0', 'B_FAKE_CAT_1', 'B_FAKE_CAT_2', 'B_FAKE_CAT_3',
                                                        'C_FAKE_CAT_SOUR', 'C_FAKE_CAT_SWEET', 'C_FAKE_CAT_TART'])
        self.assertFalse('A_FAKE_CAT' in result_df.columns)
        self.assertIn(result_df['A_FAKE_CAT_0'][0], [0.0, 1.0])

    def test_standard_scaled(self):
        wine = load_wine()
        X = pd.DataFrame(wine.data)
        X.columns = wine.feature_names
        y = pd.DataFrame(wine.target)
        y.columns = ['TARGET']
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        result = preproc([X],
                         standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                         'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                         'proanthocyanins', 'color_intensity', 'hue',
                                         'od280/od315_of_diluted_wines', 'proline'])
        result_df = result[0]
        self.assertAlmostEqual(
            result_df['alcohol'].mean(), result_df['malic_acid'].mean())


if __name__ == '__main__':
    unittest.main()
