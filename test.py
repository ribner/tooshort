import unittest

from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.svm import SVC
#from choose_models import choose_models
from too_short import TooShort
#from too_short import preproc
#from too_short import get_param_grid
#from grid_search import search
#from feature_selection import select_features
#from oversampling import oversample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter


def get_iris():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y


def get_boston():
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    X.columns = boston.feature_names
    y = pd.DataFrame(boston.target)
    y.columns = ["target"]
    return X, y


class TestFeatureSelection(unittest.TestCase):
    def testBasicFeatureSelection(self):
        X, y = get_iris()
        too_short = TooShort(prediction_type="classification")
        result = too_short.preproc([X],
                                   standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                                   'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                                   'proanthocyanins', 'color_intensity', 'hue',
                                                   'od280/od315_of_diluted_wines', 'proline'])
        X = result[0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y["target"].ravel(), test_size=0.33, random_state=42)
        X_train_filtered, X_test_filtered = too_short.select_features(
            X_train, y_train, X_test)
        self.assertTrue(len(X_train.columns) > len(X_train_filtered.columns))
        self.assertTrue(len(X_test.columns) > len(X_test_filtered.columns))


class TestEDA(unittest.TestCase):
    def testBasicEDA(self):
        return None


class TestOversampling(unittest.TestCase):
    def testBasicOversamplingNoDfWithUndersample(self):
        too_short = TooShort()
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
        os_X, os_y = too_short.oversample(X, y)
        count = Counter(os_y)
        self.assertTrue(count[0] == 200)
        self.assertTrue(count[1] == 200)

    def testBasicOversamplingNoDfNoUndersampling(self):
        too_short = TooShort()
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.75], flip_y=0, random_state=1)
        os_X, os_y = too_short.oversample(X, y)
        count = Counter(os_y)
        print(count)
        self.assertTrue(count[0] == 7500)
        self.assertTrue(count[1] == 7500)


class TestEndToEnd(unittest.TestCase):
    def testCatSmallEndToEnd(self):
        too_short = TooShort(prediction_type="classification")
        X, y = get_iris()
        result = too_short.preproc([X],
                                   standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                                   'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                                   'proanthocyanins', 'color_intensity', 'hue',
                                                   'od280/od315_of_diluted_wines', 'proline'])
        X = result[0]

        models = too_short.choose_models(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y["target"].ravel(), test_size=0.33, random_state=42)
        result = too_short.search(models, X_train, y_train, X_test, y_test)
        model_keys = result.keys()
        self.assertIn('SVC', model_keys)

    def testRegressionSmallEndToEnd(self):
        too_short = TooShort(prediction_type="regression")
        X, y = get_boston()
        result = too_short.preproc([X],
                                   standard_scale=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                                   'PTRATIO', 'B', 'LSTAT'])
        X = result[0]

        models = too_short.choose_models(y)
        print(models)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y["target"].ravel(), test_size=0.33, random_state=42)
        result = too_short.search(models, X_train, y_train, X_test, y_test)
        model_keys = result.keys()
        print(result)
        self.assertIn('Ridge', model_keys)


class TestGridSearch(unittest.TestCase):

    def test_basic_grid_search(self):
        too_short = TooShort()
        X, y = get_iris()
        result = too_short.preproc([X],
                                   standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                                   'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                                   'proanthocyanins', 'color_intensity', 'hue',
                                                   'od280/od315_of_diluted_wines', 'proline'])
        X = result[0]
        result = too_short.search([KNeighborsClassifier], X, y)
        model_keys = result.keys()
        self.assertIn('KNeighborsClassifier', model_keys)


class TestChooseModels(unittest.TestCase):
    def test_returns_regression_models_small_samples(self):
        too_short = TooShort(prediction_type="regression")
        iris = load_iris()
        y = pd.DataFrame(iris.target)
        y.columns = ["target"]
        result = too_short.choose_models(y)
        self.assertIn(LinearRegression, result)
        self.assertNotIn(SGDRegressor, result)

    def test_returns_regression_models_many_samples(self):
        too_short = TooShort(prediction_type="regression")
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        result = too_short.choose_models(y)
        self.assertIn(LinearRegression, result)
        self.assertIn(SGDRegressor, result)

    def test_returns_classification_models_small_samples(self):
        too_short = TooShort(prediction_type="classification")
        iris = load_iris()
        y = pd.DataFrame(iris.target)
        y.columns = ["target"]
        result = too_short.choose_models(y)
        self.assertIn(SVC, result)
        self.assertNotIn(SGDClassifier, result)

    def test_returns_classification_models_many_samples(self):
        too_short = TooShort(prediction_type="classification")
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        result = too_short.choose_models(y)
        self.assertIn(SVC, result)
        self.assertIn(SGDClassifier, result)


class TestGetHyperParamGrids(unittest.TestCase):
    def test_returns_linear_regression_params(self):
        too_short = TooShort()
        lr_params = {
            'normalize': [True, False]
        }
        result = too_short.get_param_grid(LinearRegression)
        self.assertEqual(result, lr_params)


class TestPreproc(unittest.TestCase):
    def test_does_not_alter_original_df(self):
        too_short = TooShort()
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
        too_short.preproc([X], OHE=np.array(
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
        too_short = TooShort()
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
        result = too_short.preproc([X], OHE=np.array(
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
        too_short = TooShort()
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
        result = too_short.preproc([X],
                                   standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                                   'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                                   'proanthocyanins', 'color_intensity', 'hue',
                                                   'od280/od315_of_diluted_wines', 'proline'])
        result_df = result[0]
        self.assertAlmostEqual(
            result_df['alcohol'].mean(), result_df['malic_acid'].mean())


if __name__ == '__main__':
    unittest.main()
