"""Finding exoplanets in the Kepler data!"""

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
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


N_FOLDS = 5  # Number of folds to use in cross-validation

METHODS = {
    'svm': SVC(random_state=0),
    'rfc': RandomForestClassifier(random_state=0),
    'xgb': xgb.XGBClassifier(random_state=0)}

SVM_GRID = {
    # ...
    'svc__kernel': ['rbf', 'poly', 'sigmoid'],

    # ...
    'svc__C': [.01, .1, 1, 10, 100],

    # ...
    'svc__gamma': [.001, .01, .1, 1]}

SVM_TEST_GRID = {
    'svc__kernel': ['rbf', 'poly'],
    'svc__C': [10, 100],
    'svc__gamma': [.001, .01]}

RFC_GRID = {
    # Number of trees in the random forest
    'randomforestclassifier__n_estimators': list(range(200, 2200, 200)),

    # Number of features to consider at every split
    'randomforestclassifier__max_features': ['auto', 'sqrt'],

    # Maximum number of levels in each tree
    'randomforestclassifier__max_depth': list(range(10, 110, 10)) + [None],

    # Minimum number of samples to split a node
    'randomforestclassifier__min_samples_split': [2, 5, 10],

    # Minimum number of samples required at each leaf node
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],

    # Method of selecting samples for training each tree
    'randomforestclassifier__bootstrap': [True, False]}

XGB_GRID = {
    # ... TODO: add rest of XGB grid
    'xgbclassifier__n_estimators': list(range(200, 2200, 200))}


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


def split_df(df: pd.DataFrame, test_size: float = .3) -> tuple[np.array]:
    """Split dataframe into train and test set

    Args:
        test_size: relative size of the test set"""

    X, y = df.drop('y', axis=1).to_numpy(), df['y'].to_numpy()
    return train_test_split(X, y, test_size=test_size, random_state=0)


def cv_grid_search(X_train: np.array, y_train: np.array,
                   method: str, test: bool = True) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Use cross-validated grid search to find best-fitting parameters

    Args:
        X_train: input variables
        y_train: output variables
        method: classification methods to use ('svm' or 'rfc')
        test: use reduced model for testing purposes

    """

    assert method in ['svm', 'rfc', 'xgb']

    estimator = DecisionTreeRegressor(random_state=0)  # Estimator used for imputation
    cachedir = mkdtemp()  # noqa; we cache transformers to avoid repeated computation

    pipe = make_pipeline(
        IterativeImputer(random_state=0, estimator=estimator),
        preprocessing.PowerTransformer(),
        METHODS[method],
        memory=cachedir,
        verbose=test)

    # Cross validation settings
    cv = KFold(N_FOLDS, shuffle=True, random_state=0)

    if method == 'svm':
        search = GridSearchCV(
            estimator=pipe,
            param_grid=SVM_TEST_GRID if test else SVM_GRID,
            n_jobs=-1,
            cv=cv,
            verbose=test)

    else:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=RFC_GRID if method == 'rfc' else XGB_GRID,
            n_iter=10 if test else 100,
            n_jobs=-1,
            cv=cv,
            verbose=test,
            random_state=0)

    search.fit(X_train, y_train)  
    rmtree(cachedir)
    if test:
        print(f'Best parameters (CV score={search.best_score_:.3f})')
        print(search.best_params_)

    return search


def main():
    # Specify file path
    fp = 'cumulative_2021.12.05_03.56.43.csv'

    # Import data
    raw = import_data(fp)

    # Basic data clean-up
    df = basic_data_clean_up(raw)

    if not ...:
        #  Descriptive stats
        first_impression(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_df(df)  # noqa

    # Missing data imputation, scaling & SVM
    cv_grid_search(X_train, y_train, method='svm')

    return df


if __name__ == '__main__':
    df = main()
