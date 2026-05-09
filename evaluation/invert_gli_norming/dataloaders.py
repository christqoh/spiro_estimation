import os

import pandas as pd


def load_spline_lookup_table(sex: str, predictor: str) -> pd.DataFrame:
    """

    :param sex: sex in {males, females}
    :param predictor: in {FVC, FEV1}
    :return:
    """
    splines = pd.read_csv(os.getcwd() + '/evaluation/invert_gli_norming/lookuptables/' + predictor + '_' + sex + '.csv', skiprows=1)
    cols = list(splines.columns)
    splines.columns = [col.lower().replace(' ', '_').replace(':', '') for col in cols]
    splines.set_index('age', inplace=True)
    splines = splines[['lspline', 'mspline', 'sspline']]
    return splines


def load_coefficients(sex: str, predictor: str, coefficient: str, value_column: str) -> pd.DataFrame:
    """

    :param sex: sex in {males, females}
    :param predictor: in {FVC, FEV1}
    :param coefficient: in {M, L, S}
    :param value_column: according value columns
    :return:
    """
    c = pd.read_csv(os.getcwd() + '/evaluation/invert_gli_norming/lookuptables/' + predictor + '_' + sex + '.csv', skiprows=2)
    cols = list(c.columns)
    c.columns = [col.lower().replace(' ', '_').replace(':', '') for col in cols]
    c.set_index(coefficient, inplace=True)
    c = c.iloc[:15]  # just in case there is any noise further down
    c = c[value_column].dropna()
    c = c.iloc[:-1]  # drop last row with spline id
    c = c.astype('float32')
    return c
