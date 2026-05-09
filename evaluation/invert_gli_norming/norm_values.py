from copy import deepcopy

import numpy as np
import pandas as pd


def get_l_value(coeffs: pd.DataFrame, l_spline_ref: pd.Series, age_table: np.ndarray):
    l_spline_table = deepcopy(age_table)

    for col in range(age_table.shape[1]):
        age = age_table[~np.isnan(age_table[:, col]), col]
        l_spline_table[~np.isnan(l_spline_table[:, col]), col] = l_spline_ref[age]

    l_values = coeffs['q0'] + coeffs['q1'] * np.log(age_table) + l_spline_table
    return l_values


def get_m_value(coeffs: pd.DataFrame, m_spline_ref: pd.Series, age_table: np.ndarray, height_table: np.ndarray,
                caucasian: int = 0, african_american: int = 0, nea: int = 0, sea: int = 0, other: int = 0) -> np.ndarray:
    m_spline_table = deepcopy(age_table)

    assert sum([caucasian, african_american, nea, sea, other]) == 1

    for col in range(age_table.shape[1]):
        age = age_table[~np.isnan(age_table[:, col]), col]
        m_spline_table[~np.isnan(m_spline_table[:, col]), col] = m_spline_ref[age]

    # m_spline = get_m_spline(coeffs, age)
    m_values = np.exp(coeffs['a0'] +
                      coeffs['a1'] * np.log(height_table) +
                      coeffs['a2'] * np.log(age_table) +
                      coeffs['a3'] * african_american +
                      coeffs['a4'] * nea +
                      coeffs['a5'] * sea +
                      coeffs['a6'] * other +
                      m_spline_table)

    return m_values


def get_s_value(coeffs: pd.DataFrame, s_spline_ref: pd.Series, age_table: np.ndarray,
                caucasian: int = 0, african_american: int = 0, nea: int = 0, sea: int = 0, other: int = 0) -> np.ndarray:
    s_spline_table = deepcopy(age_table)

    assert sum([caucasian, african_american, nea, sea, other]) == 1

    for col in range(age_table.shape[1]):
        age = age_table[~np.isnan(age_table[:, col]), col]
        s_spline_table[~np.isnan(s_spline_table[:, col]), col] = s_spline_ref[age]

    s_values = np.exp(coeffs['p0'] +
                      coeffs['p1'] * np.log(age_table) +
                      coeffs['p2'] * african_american +
                      coeffs['p3'] * nea +
                      coeffs['p4'] * sea +
                      coeffs['p5'] * other +
                      s_spline_table)

    return s_values


def get_l_value_long(coeffs: pd.DataFrame, l_spline_ref: pd.Series, age: pd.Series):
    l_spline = pd.merge(left=age, right=l_spline_ref, left_on=age.values, right_index=True, how='outer')
    l_spline.dropna(subset=['age'], inplace=True)
    l_values = coeffs['q0'] + coeffs['q1'] * np.log(age).to_numpy() + l_spline.loc[:, 'lspline'].to_numpy()
    return l_values


def get_m_value_long(coeffs: pd.DataFrame, m_spline_ref: pd.Series, age: pd.Series, height_table: pd.Series,
                     black: int = 0, nea: int = 0, sea: int = 0, other: int = 1) -> float:
    m_spline = pd.merge(left=age, right=m_spline_ref, left_on=age.values, right_index=True, how='outer')
    m_spline.dropna(subset=['age'], inplace=True)

    m_values = np.exp(coeffs['a0'] +
                      coeffs['a1'] * np.log(height_table).to_numpy() +
                      coeffs['a2'] * np.log(age).to_numpy() +
                      coeffs['a3'] * black +
                      coeffs['a4'] * nea +
                      coeffs['a5'] * sea +
                      coeffs['a6'] * other +
                      m_spline.loc[:, 'mspline'].to_numpy())
    return m_values


def get_s_value_long(coeffs: pd.DataFrame, s_spline_ref: pd.Series, age: pd.Series,
                     black: int = 0, nea: int = 0, sea: int = 0, other: int = 1) -> float:
    s_spline = pd.merge(left=age, right=s_spline_ref, left_on=age.values, right_index=True, how='outer')
    s_spline.dropna(subset=['age'], inplace=True)

    s_values = np.exp(coeffs['p0'] +
                      coeffs['p1'] * np.log(age).to_numpy() +
                      coeffs['p2'] * black +
                      coeffs['p3'] * nea +
                      coeffs['p4'] * sea +
                      coeffs['p5'] * other +
                      s_spline.loc[:, 'sspline'].to_numpy())
    return s_values
