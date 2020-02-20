# TOO SHORT

## Description:

This is a package intended to make your life easier, and automate a lot of the common things dealt with in supervised learning. Specifically with building and testing sklearn models. Although, it also relies on some other useful tools like SMOTE for imabalanced data.

In short:

- Chooses relevant models for you based on the type of problem (classification or regression)
- Provides param grids for the chosen models
- Wraps all the preprocessing into one function so you can do somewhat customized preprocessing with a oneliner
- Wraps Oversampling and undersampling using the imbalanced learn package to simplify the process
- Feature selection based on a model input
- Automatically splitting train and test sets, as well as evaluting them seperately.
- Most importantly, provides a simple method to search and compare all the relevant models and grids with either your original data, or the transformed version set in the preprocessing, oversampling, or feature selection step.

## Class methods

The module exposes one class, too_short.

---

### init

`def __init__(self, X=None, y=None, prediction_type=None):`

**init function**

**Args:**

- X (pd.dataframe): Df with full X data (will be split within init) - optional (you will need to include this for most functionality)
- y (list): list of targets (will be split within init) - optional (you will need to include this for most functionality)
- prediction_type (string): String of either "classification" or "regression" - optional (you will need to include this for most functionality)

**Returns:**
None

---

### set_attributes

`def set_attributes(self, X=None, y=None, X_test=None, y_test=None, X_train=None, y_train=None, prediction_type=None, imb_pipeline_steps=None, models=None):`

**Function for setting class attributes manually. Useful for overriding defaults if you are trying to test different settings**

**Args:**

- X (Pd dataframe): containing X - optional
- y (list): target list - optional
- X_test (pd dataframe): X_test - optional
- y_test (pd dataframe): y_test - optional
- X_train (pd dataframe): X_train - optional
- y_train (pd dataframe): y_train - optional
- prediction_type (str): Either "classification" or "regression" - optional
- imb_pipline_steps (list): Containing imbalanced learn Pipeline steps. ex [('smote', SMOTE(random_state=random_state))]. If you want to use this within the search method  
  dont add a sklearn model step to the end of this, that will be done automatically in the the search function. - optional
- models (list): List of models to be used in in the search function. This would be set automatically in choose_models function, but you can override here. Do not instantiate the models within the list. - optional

**Returns:**
None

---

### get_param_grid

`def get_param_grid(self, model, prepend=None):`

**Function providing a hyperparam grid to be used in sklearn hyperparameter optimizatoin. This is automatically called internally in the search function, the user need not call this directly.**

**Args:**

- model (sklearn model): models' `.__name__` property - Required
- prepend (string): To be prepended to grid keys for grid search along with to underscores. this will generally be the model name as a string. ie "LogisticRegression" - optional

**Returns:**
Dictionary containing sklearn params as keys and list of param options as values

**Example:**

```
get_param_grid(LinearRegression)
> > {'normalize': [True, False]}
```

---

### preproc

`def preproc(self, OHE=[], standard_scale=[], numerical_impute=[], categegorical_impute=[], label_encode={}):`

**Prerocesses data frames, including onehot encoding, scaling, and imputation, and label encoding**

**Args:**

- OHE (list): columns to be processed with sklearn OneHotEncoder, this accepts non numerical categorical rows without need for label encoding. - Default []
- standard_scale (list): List of columns to be processes with standard scalar. - Defualt []
- numerical_impute (list): List of column names that should be imputed using mean method. - Default []
- categorical_impute (list): List of column names that should be imputed to 'missing'. - Default []
- label_encode (dict): Keys in the dict should be the column names to transform, the values should be lists that
  contain the various values in the column, the order of the values will determine the encoding (1st element will be 0 etc.). - Default {}

**Returns:**
List of processed pandas dataframes. Processed dfs will overwrite self.X_train and self.X_test.

---

### choose_models

`def choose_models(self):`

**Function giving you suggested sklearn models based on prediction type and size of data. classification_type must be set during the class instantiation or using set_attributes.**

**Args:**
None

**Returns:**
List of sklearn model classes. Models saved to class instance under self.models, they will be automatically passed to
the search method.

---

### oversample

`def oversample(self):`

**Function tranforming train data using undersampling and oversampling. Uses undersampling as well as oversampling if the ratio between classes is highly imbalanced. Otherwise only oversampling will be used**

**Args:**
None

**Returns:**
os_X_train, os_y_train. As matrices (as returned by SMOTE). os_X_train and os_y_train are saved to class and will be automatically applied during the search method, if oversample method is run first.

---

### select_features

`def select_features(self, model=None):`

**Select best features from X_test and X_train**

**Keyword args:**

- model (sklearn model): - The model that will be used in sklearns SelectFromModel method Needs to
  be instantiated - Default is LinearRegression if prediction_type is "regression", otherwise if prediction_type is "classification" defaults to LinearSVC()

**Returns:**
limited_X_train, limited_X_test - Also replaces self.X_test and self.X_train with the limited features.

---

### search

`def search(self, scoring=None):`

**Function performing a grid search on a list of predefined models**

**Keyword Args:**

- Scoring (string): Scoring param as allowed by grid search cv - optional

**Returns:**
Dict containing each model, and within each model a sub dict containing the best grid search cv scores, best params, and test score.

## Examples

### Basic model training with some preprocessing

```

X, y = get_iris()
too_short = TooShort(X, y, prediction_type="classification")
result = too_short.preproc(
standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
'proanthocyanins', 'color_intensity', 'hue',
'od280/od315_of_diluted_wines', 'proline'])

models = too_short.choose_models()
result = too_short.search()
print(result)

```

#### output

```

{'SVC': {'best_score': 0.9833333333333332, 'best_params': {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}, 'test_score': 1.0}, 'RandomForestClassifier': {'best_score': 0.9833333333333332, 'best_params': {'RandomForestClassifier**bootstrap': False, 'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 32}, 'test_score': 1.0}, 'KNeighborsClassifier': {'best_score': 1.0, 'best_params': {'p': 1, 'n_neighbors': 17, 'leaf_size': 21}, 'test_score': 0.9830508474576272}, 'LinearSVC': {'best_score': 0.9833333333333332, 'best_params': {'C': 0.0001, 'dual': True}, 'test_score': 1.0}, 'LogisticRegression': {'best_score': 0.9833333333333332, 'best_params': {'LogisticRegression**class_weight': {1: 0.5, 0: 0.5}, 'LogisticRegression\_\_C': 100, 'solver': 'liblinear', 'penalty': 'l2'}, 'test_score': 1.0}}

```

---

### Choosing your own models

```

X, y = get_iris()
too_short = TooShort(X, y)
result = too_short.preproc(
standard_scale=too_short.X_train.columns)
too_short.set_attributes(models=[KNeighborsClassifier])
result = too_short.search()
print(result)

```

#### output

```

{'KNeighborsClassifier': {'best_score': 0.9833333333333332, 'best_params': {'p': 1, 'n_neighbors': 20, 'leaf_size': 14}, 'test_score': 1.0}}

```

---

### More involved preprocessing

```

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
print(too_short.X_train.columns)

```

#### output

You new column names A_FAKE_CAT_0 etc have been one hot encoded and renamed. The final column has been label encoded (dont apply this for OHE as a first step). The other columns have been applied with the standard scalar.

```

A_FAKE_CAT_0 A_FAKE_CAT_1 A_FAKE_CAT_2 A_FAKE_CAT_3 B_FAKE_CAT_0 B_FAKE_CAT_1 ... proanthocyanins color_intensity hueod280/od315_of_diluted_wines proline D_FAKE_LABEL_CAT

```

---

### Oversampling, feature selection and customer scoring (recall)

```

df = pd.read_excel(
"https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", encoding="utf-8", skiprows=1)
df = df.rename(
columns={'default payment next month': 'DEFAULT_PAYMENT_NEXT_MONTH', 'PAY_0': 'PAY_1'})
y = df['DEFAULT_PAYMENT_NEXT_MONTH'].ravel()
X = df.drop(['DEFAULT_PAYMENT_NEXT_MONTH'], axis=1)
too_short = TooShort(X, y, prediction_type="classification")
too_short.oversample()
too_short.preproc(standard_scale=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
too_short.select_features()
too_short.choose_models()
result = too_short.search(scoring="recall")

```

---

### Detailed example using settattributes, search, and feature selection

First, we get the wine dataset from sklearn datasets package, add a ordinal category and three non ordinal categories for examples sake. The added columns will provide no predictive use, but will illustrate the preproccessing step, and feature selection step.

```
wine = load_wine()
X = pd.DataFrame(wine.data)
X.columns = wine.feature_names
y = wine.target
X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
X['D_FAKE_LABEL_CAT'] = np.random.choice(
    ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
```

Ok now lets use the TooShort module.

```
too_short = TooShort(X, y, prediction_type="classification")
too_short.preproc(OHE=np.array(
    ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
    label_encode={
    'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
},
    standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                    'proanthocyanins', 'color_intensity', 'hue',
                    'od280/od315_of_diluted_wines', 'proline'])
print(too_short.X_train.columns)
>> ['A_FAKE_CAT_0', 'A_FAKE_CAT_1', 'A_FAKE_CAT_2', 'A_FAKE_CAT_3',
       'B_FAKE_CAT_0', 'B_FAKE_CAT_1', 'B_FAKE_CAT_2', 'B_FAKE_CAT_3',
       'C_FAKE_CAT_SOUR', 'C_FAKE_CAT_SWEET', 'C_FAKE_CAT_TART', 'alcohol',
       'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
       'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
       'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline',
       'D_FAKE_LABEL_CAT']
```

Above we can see the new rows created through one hot encoding. The final column has the original name but its contents Will be label encoded on a scale of 0 to 4 based on the arrray passed in above. The rest of the rows had standard scalar applied

Lets set the models, based on the size of the dataset and the prediction type (classification in this case).

```
too_short.choose_models()
print(too_short.models)
>> >> [<class 'sklearn.svm._classes.SVC'>, <class 'sklearn.ensemble._forest.RandomForestClassifier'>, <class 'sklearn.neighbors._classification.KNeighborsClassifier'>, <class 'sklearn.svm._classes.LinearSVC'>, <class 'sklearn.linear_model._logistic.LogisticRegression'>]
```

The models property has been set with the list of applicable models, these models are uninstantiated.

Next we can search using the models above, using a grid search. We will automatically apply seperate param grids that are specific to each model for tuning.

```
result = too_short.search()
print(result)
>> {'SVC': {'best_search_score': 0.9666666666666668, 'best_params': {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}, 'test_score': 1.0}, 'RandomForestClassifier': {'best_search_score': 0.9666666666666668, 'best_params': {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 98, 'bootstrap': True}, 'test_score': 0.9830508474576272}, 'KNeighborsClassifier': {'best_search_score': 0.9663043478260871, 'best_params': {'p': 2, 'n_neighbors': 19, 'leaf_size': 1}, 'test_score': 0.9661016949152542}, 'LinearSVC': {'best_search_score': 0.975, 'best_params': {'C': 0.01, 'dual': True}, 'test_score': 1.0}, 'LogisticRegression': {'best_search_score': 0.9666666666666668, 'best_params': {'solver': 'saga', 'penalty': 'l2', 'class_weight': {1: 0.5, 0: 0.5}, 'C': 0.1}, 'test_score': 0.9661016949152542}}
```

Above we can see the scores for each model, along with the params used, and the test score as well.

Lets grab the best logistic regression params, or any other model you prefer and perform feature selection on that model and params We could have called select_features() with no params before the search method, but we should get a better result by using the specific model we have chosen for evaluation.

```
best_lr_params = result['LogisticRegression']['best_params']
too_short.select_features(LogisticRegression(**best_lr_params))
print(too_short.X_train.columns)
>> ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'total_phenols',
        'flavanoids', 'proanthocyanins', 'color_intensity', 'hue',
        'od280/od315_of_diluted_wines', 'proline']
```

Since the added categorical columns were created at random, they had no predicted value and were automatically dropped. Aside from the fake columns, it looks like we got rid of the magnesium feature as well.

Now that we have narrowed down the features under specific model and params, lets manually set the model to logistic regression and see if we can do any better. We could call search directly, without setting the model but this will save us time from running the grid search again on the other models.

```
too_short.set_attributes(models=[LogisticRegression])
results = too_short.search()
print(results)
>> {'LogisticRegression': {'best_search_score': 0.9833333333333334, 'best_params': {'solver': 'saga', 'penalty': 'l1', 'class_weight': {1: 0.4, 0: 0.6}, 'C': 100}, 'test_score': 0.9830508474576272}}
```

It looks like the new data set with feature selection performed sligthly better (96.6% to 98.3). If we wanted to dig deeper we could try some other things like over and under sampling, or alternate preprocessing methods.

## Further examples

**for more example check <a href="./test.py">The test file</a>**
