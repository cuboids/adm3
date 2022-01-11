""""Finding exoplanets in the Kepler data!

Note: Python 3.8+ is required.
"""

from collections import OrderedDict
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional, Union

import numpy as np
import pandas as pd; pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer  # noqa
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


N_FOLDS = 5  # Number of folds to use in cross-validation

METHODS = {
    'svm': SVC(
        random_state=0,
        probability=True),

    'rfc': RandomForestClassifier(
        random_state=0),

    'xgb': xgb.XGBClassifier(
        random_state=0,
        use_label_encoder=False,
        eval_metric='logloss')}

SVM_GRID = {
    # Kernel type to be used in the algorithm
    'svc__kernel': ['rbf', 'poly', 'sigmoid'],

    # Regularization parameter
    'svc__C': [.01, .1, 1, 10, 100],

    # Kernel coefficient
    'svc__gamma': [.001, .01, .1, 1]}

RFC_GRID = {
    # Number of trees in the random forest
    'n_estimators': list(range(200, 2200, 200)),

    # Number of features to consider at every split
    'max_features': ['auto', 'sqrt'],

    # Maximum number of levels in each tree
    'max_depth': list(range(10, 110, 10)) + [None],

    # Minimum number of samples to split a node
    'min_samples_split': [2, 5, 10],

    # Minimum number of samples required at each leaf node
    'min_samples_leaf': [1, 2, 4],

    # Method of selecting samples for training each tree
    'bootstrap': [True, False]}

# For compatibility with SK Learn pipes.
RFC_GRID = {'randomforestclassifier__' + k: v for k, v in RFC_GRID.items()}

XGB_GRID = {
    # Number of estimators
    'n_estimators': list(range(50, 550, 50)),

    # Learning rate
    'learning_rate': [.01, .03, .1, .3, .6],

    # Minimum loss reduction required for partitioning
    'gamma': [0, 1, 10, 100, 1000],

    # Maximum depth of the tree
    'max_depth': [4, 6, 8, 10],

    # L1 regularization
    'reg_alpha': [.01, .1, 1, 10, 100],

    # L2 regularization
    'reg_lambda': [.01, .1, 1, 10, 100]

}

XGB_GRID = {'xgbclassifier__' + k: v for k, v in XGB_GRID.items()}

# Test grids with fewer parameter values to test the pipeline

SVM_TEST_GRID = {
    'svc__kernel': ['rbf', 'poly'],
    'svc__C': [10, 100],
    'svc__gamma': [.001, .01]}

RFC_TEST_GRID = {
    'randomforestclassifier__n_estimators': [800, 1200],
    'randomforestclassifier__max_features': ['auto', 'sqrt'],
    'randomforestclassifier__max_depth': [50, 60],
    'randomforestclassifier__min_samples_split': [2, 10],
    'randomforestclassifier__min_samples_leaf': [1, 4],
    'randomforestclassifier__bootstrap': [True, False]
}

XGB_TEST_GRID = {
    'xgbclassifier__n_estimators': [100, 300],
    'xgbclassifier__learning_rate': [.01, .6],
    'xgbclassifier__gamma': [1, 10],
    'xgbclassifier__max_depth': [4, 10],
    'xgbclassifier__reg_alpha': [1, 10],
    'xgbclassifier__reg_lambda': [1, 10]
}

# The parameters of the methods below are the output of the
# optimal values according to `cv_grid_search` below.
#
# Base model accuracies: svm = .923, rfc = .921, xgb = .926
# Optimized accuracies: svm = .923, rfc = .924, xgb = .931

METHODS_OPTIMAL = {
    'svm': SVC(
        C=100,
        kernel='rbf',
        gamma=.01,
        random_state=0,
        probability=True),

    'rfc': RandomForestClassifier(
        n_estimators=2000,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='auto',
        max_depth=80,
        bootstrap=False,
        random_state=0),

    'xgb': xgb.XGBClassifier(
        reg_lambda=1,
        reg_alpha=.1,
        n_estimators=500,
        max_depth=8,
        learning_rate=.1,
        gamma=0,
        random_state=0,
        use_label_encoder=False,
        eval_metric='logloss')}


def import_data(fp: str) -> pd.DataFrame:
    """Import Kepler data from csv
    source: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

    Args:
        fp: file path to Kepler data
    """
    return pd.read_csv(fp, comment='#')


def first_impression(df: pd.DataFrame) -> None:
    """View basic descriptive statistics"

    Args:
        df: Pandas dataframe
    """
    print(df.head())
    print(df.describe())
    df.hist()
    plt.show()


def basic_data_clean_up(raw: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Select columns for the feature matrix and code target variable as y

    Args:
        raw: kepler data
        cols: optional list of columns to be included
    """

    # More information about the meaning of each column:
    # https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
    cols = cols if cols is not None else [
        'kepoi_name',  # KOI Name
        'koi_disposition',  # Exoplanet Archive Disposition
        'koi_period',  # Orbital Period
        'koi_time0bk',  # Transit Epoch [BKJD]
        'koi_impact',  # Impact Parameter
        'koi_duration',  # Transit Duration [hrs]
        'koi_depth',  # Transit Depth [ppm]
        'koi_prad',  # Planetary Radius [Earth radii]
        'koi_teq',  # Equilibrium Temperature [K]
        'koi_insol',  # Insolation Flux [Earth flux]
        'koi_model_snr',  # Transit Signal-to-Noise
        'koi_tce_plnt_num',  # TCE Planet Number
        'koi_steff',  # Stellar Effective Temperature [K]
        'koi_slogg',  # Stellar Surface Gravity [log10(cm/s**2)]
        'koi_srad',  # Stellar Radius [Solar radii]
        'ra',  # RA [decimal degrees]
        'dec',  # Dec [decimal degrees]
        'koi_kepmag']  # Kepler-band [mag]

    # Check that `cols` includes the planet name and disposition
    assert 'kepoi_name' in cols and 'koi_disposition' in cols

    # Only preserve the columns in cols
    df = raw[cols]

    # Planet names need to be unique
    assert len(df['kepoi_name']) == len(set(df['kepoi_name']))

    # Set planet name as row index
    df = df.set_index('kepoi_name')

    # Filter out exoplanet candidates
    df = df[df['koi_disposition'] != 'CANDIDATE']

    # Ensure a planet is either CONFIRMED or FALSE POSITIVE, not a CANDIDATE
    assert len(set(df['koi_disposition'])) == 2

    # Code the target variable as binary
    df['y'] = [int(b) for b in df['koi_disposition'] == 'CONFIRMED']
    df = df.drop('koi_disposition', axis=1)

    return df


def baseline_score(y_train: np.array) -> float:

    # Get predictions on the training set
    baseline_predictions = np.zeros(y_train.shape)

    # Compute MAE
    acc = accuracy_score(y_train, baseline_predictions)
    print(f'Baseline Accuracy: {acc:.3f}')
    print()

    return acc


def split_df(df: pd.DataFrame, test_size: float = .3) -> tuple[np.array]:
    """Split dataframe into train and test set

    Args:
        df: kepler data
        test_size: relative size of the test set"""

    X, y = df.drop('y', axis=1).to_numpy(), df['y'].to_numpy()
    return train_test_split(X, y, test_size=test_size, random_state=0)


def cv_grid_search(X_train: np.array, y_train: np.array, method: str, test: bool = True,  # noqa
                   verbose: Optional[bool] = True) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Use cross-validated grid search to find best-fitting parameters

    Args:
        X_train: input variables
        y_train: output variables
        method: classification methods to use ('svm' or 'rfc')
        test: use reduced model for testing purposes
        verbose: print intermediate output
    """

    assert method in ['svm', 'rfc', 'xgb']

    verbose = test if verbose is None else verbose

    estimator = DecisionTreeRegressor(random_state=0)  # Estimator used for imputation
    cachedir = mkdtemp()  # noqa; we cache transformers to avoid repeated computation

    pipe = make_pipeline(
        IterativeImputer(random_state=0, estimator=estimator),
        preprocessing.PowerTransformer(),
        METHODS[method],
        memory=cachedir,
        verbose=verbose)

    # Cross validation settings
    cv = KFold(N_FOLDS, shuffle=True, random_state=0)

    if method == 'svm':
        search = GridSearchCV(
            estimator=pipe,
            param_grid=SVM_TEST_GRID if test else SVM_GRID,
            n_jobs=-1,
            cv=cv,
            verbose=verbose)

    else:
        # Select appropriate grid.
        if method == 'rfc':
            param_distributions = RFC_TEST_GRID if test else RFC_GRID
        else:
            param_distributions = XGB_TEST_GRID if test else XGB_GRID
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=10 if test else 100,
            n_jobs=-1,
            cv=cv,
            verbose=verbose,
            random_state=0)

    search.fit(X_train, y_train)
    rmtree(cachedir)
    print(f'Best parameters (CV score={search.best_score_:.3f})')
    print(search.best_params_)

    return search


def evaluate(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array,
             verbose: bool = True) -> dict[str]:
    """Evaluate optimal models on test set"""

    estimator = DecisionTreeRegressor(random_state=0)  # Estimator used for imputation
    cachedir = mkdtemp()  # noqa; we cache transformers to avoid repeated computation

    scores = {'svm': None, 'rfc': None, 'xgb': None}
    for method in ['svm', 'rfc', 'xgb']:

        pipe = make_pipeline(
            IterativeImputer(random_state=0, estimator=estimator),
            preprocessing.PowerTransformer(),
            METHODS_OPTIMAL[method],
            memory=cachedir,
            verbose=verbose)

        pipe.fit(X_train, y_train)
        scores[method] = pipe.score(X_test, y_test)

    return scores


def main(method: str = 'xgb', test: bool = True, compute_optimal_parameters: bool = False) -> None:
    """

    Args:
        method: Method to use ('svm', 'rfc', or 'xgb')
        test: if true, use a reduced grid for testing the pipeline
        compute_optimal_parameters: recalculate (if true) or evaluate (if false) optimal parameters

    Note: recomputing the optimal parameters on a non-test grid might take 30+ min.

    """

    # Specify file path
    fp = 'cumulative_2021.12.05_03.56.43.csv'

    # Import data
    raw = import_data(fp)

    # Basic data clean-up
    df = basic_data_clean_up(raw)

    # Train-test split
    X_train, X_test, y_train, y_test = split_df(df)  # noqa

    if test:
        print('Descriptive statistics:\n')
        first_impression(df)
        baseline_score(y_test)

    if compute_optimal_parameters:
        cv_grid_search(X_train, y_train, method=method, test=test)
        return

    scores = evaluate(X_train, y_train, X_test, y_test)
    print(scores)


if __name__ == '__main__':
    main(method='svm', test=False, compute_optimal_parameters=False)

