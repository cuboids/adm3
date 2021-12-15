"""
Steps:
* (1) csv -> pandas dataframe
* (2) basic data cleanup
* (3) target = "disposition using kepler data"
  code target as 0/1
* (4) Calibration/validation split
* (5) Scaling?
* (6) Train/test split of calibration set
* (7) SVM
* (8) Random Forests
* (9) XGBoost
* (10) Dimension reduction and data viz
"""

from typing import Optional

import numpy as np
import pandas as pd; pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import xgboost


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
    """Select columns and code target variable as y

    Args:
        raw: raw dataframe with kepler data
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
        'koi_kepmag',  # Kepler-band [mag]
    ]

    # Check that `cols` includes the planet name and disposition
    assert 'kepoi_name' in cols and 'koi_disposition' in cols

    df = raw[cols]

    # Make sure that planet names are unique
    assert len(df['kepoi_name']) == len(set(df['kepoi_name']))

    # Set planet name as row index
    df = df.set_index('kepoi_name')

    # Filter out exoplanet candidates
    df = df[df['koi_disposition'] != 'CANDIDATE']

    # Ensure a planet is either CONFIRMED or FALSE POSITIVE
    assert len(set(df['koi_disposition'])) == 2

    # Code the target variable as binary
    df['y'] = [int(b) for b in df['koi_disposition'] == 'CONFIRMED']
    df = df.drop('koi_disposition', axis=1)

    return df


def split_df(df: pd.DataFrame, test_size=.3) -> tuple[pd.DataFrame]:
    """"""

    X, y = df.drop('y', axis=1).to_numpy(), df['y'].to_numpy()
    return train_test_split(X, y, test_size=test_size, random_state=0)


def split_into_folds(df: pd.DataFrame):
    """"""

    kf = KFold(n_splits=5)


def svm_pipeline(X_train: np.array, y_train: np.array,
                 X_test: np.array, y_test: np.array) -> float:
    """Impute missing data, scale, and fit SVM model.

    Returns:
        score: the score of X_test, y_test
    """

    pipe = make_pipeline(
        IterativeImputer(random_state=0),
        preprocessing.StandardScaler(),
        SVC(gamma='auto')
    )
    pipe.fit(X_train, y_train)
    return pipe.score(X_test, y_test)


def main():
    # Specify file path
    fp = 'cumulative_2021.12.05_03.56.43.csv'

    # Import data
    raw = import_data(fp)

    # Basic data clean-up
    df = basic_data_clean_up(raw)

    # Train-test split
    X_train, X_test, y_train, y_test = split_df(df)  # noqa

    # Missing data imputatin, scaling & SVM
    score = svm_pipeline(X_train, y_train, X_test, y_test)
    print(score)
    
    return df


if __name__ == '__main__':
    df = main()
