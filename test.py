import unittest

from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from too_short import TooShort
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_iris():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


def get_boston():
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    X.columns = boston.feature_names
    y = pd.DataFrame(boston.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


def get_wine():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


class TestFeatureSelection(unittest.TestCase):
    def testBasicFeatureSelection(self):
        X, y = get_iris()
        too_short = TooShort(X, y, prediction_type="classification")
        X_train, X_test = too_short.preproc(
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        X_train_filtered, X_test_filtered = too_short.select_features()
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
        too_short.set_attributes(X_train=X, y_train=y)
        os_X, os_y = too_short.oversample()
        count = Counter(os_y)
        self.assertTrue(count[0] == 200)
        self.assertTrue(count[1] == 200)

    def testBasicOversamplingNoDfNoUndersampling(self):
        too_short = TooShort()
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.75], flip_y=0, random_state=1)
        too_short.set_attributes(X_train=X, y_train=y)
        os_X, os_y = too_short.oversample()
        count = Counter(os_y)
        self.assertTrue(count[0] == 7500)
        self.assertTrue(count[1] == 7500)

    # slow
    # def testCreditDatasetEndToEnd(self):
    #     df = pd.read_excel(
    #         "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", encoding="utf-8", skiprows=1)
    #     df = df.rename(
    #         columns={'default payment next month': 'DEFAULT_PAYMENT_NEXT_MONTH', 'PAY_0': 'PAY_1'})
    #     y = df['DEFAULT_PAYMENT_NEXT_MONTH'].ravel()
    #     X = df.drop(['DEFAULT_PAYMENT_NEXT_MONTH'], axis=1)
    #     too_short = TooShort(X, y, prediction_type="classification")
    #     too_short.oversample()
    #     too_short.preproc(standard_scale=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
    #                                       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    #                                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    #                                       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    #     too_short.choose_models()
    #     result = too_short.search()
    #     print(result)

    # slow
    # def testCreditDatasetAlternateScoringEndToEnd(self):
    #     df = pd.read_excel(
    #         "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", encoding="utf-8", skiprows=1)
    #     df = df.rename(
    #         columns={'default payment next month': 'DEFAULT_PAYMENT_NEXT_MONTH', 'PAY_0': 'PAY_1'})
    #     y = df['DEFAULT_PAYMENT_NEXT_MONTH'].ravel()
    #     X = df.drop(['DEFAULT_PAYMENT_NEXT_MONTH'], axis=1)
    #     too_short = TooShort(X, y, prediction_type="classification")
    #     too_short.oversample()
    #     too_short.preproc(standard_scale=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
    #                                       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    #                                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    #                                       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    #     too_short.select_features()
    #     too_short.choose_models()
    #     result = too_short.search(scoring="recall")
    #     print(result)


class TestEndToEnd(unittest.TestCase):
    def testCatSmallEndToEnd(self):
        X, y = get_iris()
        too_short = TooShort(X, y, prediction_type="classification")
        result = too_short.preproc(
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        models = too_short.choose_models()
        result = too_short.search()
        model_keys = result.keys()
        self.assertIn('SVC', model_keys)

    def testRegressionSmallEndToEnd(self):
        X, y = get_boston()
        too_short = TooShort(X, y, prediction_type="regression")
        result = too_short.preproc(
            standard_scale=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                                   'PTRATIO', 'B', 'LSTAT'])

        models = too_short.choose_models()
        result = too_short.search()
        model_keys = result.keys()
        self.assertIn('Ridge', model_keys)


class TestGridSearch(unittest.TestCase):

    def test_basic_custom_grid_search(self):
        X, y = get_iris()
        too_short = TooShort(X, y)
        result = too_short.preproc(
            standard_scale=too_short.X_train.columns)
        too_short.set_attributes(models=[KNeighborsClassifier])
        result = too_short.search()
        model_keys = result.keys()
        self.assertEqual(len(model_keys), 1)
        self.assertIn('KNeighborsClassifier', model_keys)


class TestChooseModels(unittest.TestCase):
    def test_returns_regression_models_small_samples(self):
        X, y = get_iris()
        too_short = TooShort(X, y, prediction_type="regression")
        result = too_short.choose_models()
        self.assertIn(LinearRegression, result)
        self.assertNotIn(SGDRegressor, result)

    def test_returns_regression_models_many_samples(self):
        too_short = TooShort(prediction_type="regression")
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        too_short.set_attributes(y_train=y)
        result = too_short.choose_models()
        self.assertIn(LinearRegression, result)
        self.assertIn(SGDRegressor, result)

    def test_returns_classification_models_small_samples(self):
        X, y = get_iris()
        too_short = TooShort(X, y, prediction_type="classification")
        result = too_short.choose_models()
        self.assertIn(SVC, result)
        self.assertNotIn(SGDClassifier, result)

    def test_returns_classification_models_many_samples(self):
        too_short = TooShort(prediction_type="classification")
        y = np.random.choice([0, 1, 2, 3, 4], 110000)
        too_short.set_attributes(y_train=y)
        result = too_short.choose_models()
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
        X, y = get_wine()
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        X_copy = X.copy()
        too_short = TooShort(X, y)
        too_short.preproc(OHE=np.array(
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
        X, y = get_wine()
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        too_short = TooShort(X, y)
        X_train, X_test = too_short.preproc(OHE=np.array(
            ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
            label_encode={
            'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
        },
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        result_df = X_train
        self.assertCountEqual(result_df.columns[0:11], ['A_FAKE_CAT_0', 'A_FAKE_CAT_1', 'A_FAKE_CAT_2', 'A_FAKE_CAT_3',
                                                        'B_FAKE_CAT_0', 'B_FAKE_CAT_1', 'B_FAKE_CAT_2', 'B_FAKE_CAT_3',
                                                        'C_FAKE_CAT_SOUR', 'C_FAKE_CAT_SWEET', 'C_FAKE_CAT_TART'])
        self.assertFalse('A_FAKE_CAT' in result_df.columns)
        self.assertIn(result_df['A_FAKE_CAT_0'][0], [0.0, 1.0])

    def test_standard_scaled(self):
        X, y = get_wine()
        X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
        X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X['D_FAKE_LABEL_CAT'] = np.random.choice(
            ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
        too_short = TooShort(X, y)
        result = too_short.preproc(
            standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                            'proanthocyanins', 'color_intensity', 'hue',
                            'od280/od315_of_diluted_wines', 'proline'])
        result_df = result[0]
        self.assertAlmostEqual(
            result_df['alcohol'].mean(), result_df['malic_acid'].mean())


if __name__ == '__main__':
    unittest.main()
