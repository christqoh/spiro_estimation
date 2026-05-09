from typing import List

import pandas as pd
import numpy as np

from evaluation.invert_gli_norming.dataloaders import load_spline_lookup_table, load_coefficients
from evaluation.invert_gli_norming.norm_values import get_l_value, get_m_value, get_s_value
from evaluation.invert_gli_norming.utils import pivot_table, augment


def invert_z_score(df: pd.DataFrame, df_demographic: pd.DataFrame, z_score: pd.DataFrame, predictor: str,
                   standard: str =  'other', backward: bool = False):
    """

    :param df:
    :param df_demographic:
    :param z_score:
    :param predictor:
    :param standard:
    :return:
    """
    birthdays = df_demographic.loc[:, '02_q_2adate_dem_date_of_birth_numeric']
    df_visit_dates = df.pivot(columns='visit_number_unified', values='18_visit_date')
    df_visit_dates.dropna(inplace=True, axis=1, how='all')

    idx = df_visit_dates.index.intersection(birthdays.dropna().index).unique()
    df_visit_dates = df_visit_dates.loc[idx, :]
    df_age_precise = (df_visit_dates.loc[idx, :] - birthdays.loc[idx].to_numpy()[:, None]) / pd.to_timedelta(365.25, 'D')
    age = np.round(df_age_precise.to_numpy() * 4, 0) / 4
    height = np.repeat(df_demographic.loc[idx, '03_q_3a_exm_height'].to_numpy()[:, None],
                       repeats=df_visit_dates.shape[1], axis=1)

    # male
    male_rows = df_demographic.loc[idx, '02_q_3a_dem_sex_at_birth'] == 'Male'
    splines_male = load_spline_lookup_table('males', predictor)
    coeffs_m_male = load_coefficients('males', predictor, 'm', 'unnamed_8')
    coefficients_s_male = load_coefficients('males', predictor, 's', 'unnamed_11')
    coefficients_l_male = load_coefficients('males', predictor, 'l', 'unnamed_14')

    if standard == 'other':
        l_male = get_l_value(coefficients_l_male, splines_male['lspline'], age)
        m_male = get_m_value(coeffs_m_male, splines_male.loc[:, 'mspline'], age, height, other=1)
        s_male = get_s_value(coefficients_s_male, splines_male['sspline'], age, other=1)
    elif standard == 'african_american':
        l_male = get_l_value(coefficients_l_male, splines_male['lspline'], age)
        m_male = get_m_value(coeffs_m_male, splines_male.loc[:, 'mspline'], age, height, african_american=1, other=0)
        s_male = get_s_value(coefficients_s_male, splines_male['sspline'], age, african_american=1, other=0)
    else:
        raise Exception(f'Unknown standard: {standard}')

    # female
    female_rows = df_demographic.loc[idx, '02_q_3a_dem_sex_at_birth'] == 'Female'

    splines_female = load_spline_lookup_table('females', predictor)
    coefficients_m_female = load_coefficients('females', predictor, 'm', 'unnamed_8')
    coefficients_s_female = load_coefficients('females', predictor, 's', 'unnamed_11')
    coefficients_l_female = load_coefficients('females', predictor, 'l', 'unnamed_14')

    if standard == 'other':
        l_female = get_l_value(coefficients_l_female, splines_female['lspline'], age)
        m_female = get_m_value(coefficients_m_female, splines_female['mspline'], age, height, other=1)
        s_female = get_s_value(coefficients_s_female, splines_female['sspline'], age, other=1)
    elif standard == 'african_american':
        l_female = get_l_value(coefficients_l_female, splines_female['lspline'], age)
        m_female = get_m_value(coefficients_m_female, splines_female['mspline'], age, height, african_american=1, other=0)
        s_female = get_s_value(coefficients_s_female, splines_female['sspline'], age, african_american=1, other=0)
    else:
        raise Exception(f'Unknown standard: {standard}')

    l_combined = l_male
    l_combined[female_rows.to_numpy(), :] = l_female[female_rows.to_numpy(), :]

    m_combined = m_male
    m_combined[female_rows.to_numpy(), :] = m_female[female_rows.to_numpy(), :]

    s_combined = s_male
    s_combined[female_rows.to_numpy(), :] = s_female[female_rows.to_numpy(), :]

    # backward:
    if backward:
        z_score = z_score.loc[:, df_visit_dates.columns]
        measured = np.power((z_score * (l_combined * s_combined) + 1), (1/l_combined)) * m_combined
        df_r = measured.reset_index().melt(id_vars='person_id_complete').dropna()
        df_r.columns = ['index', 'visit_number_unified', predictor + '_measured']
    else:
        df_r = z_score.reset_index().melt(id_vars='person_id_complete').dropna()
        df_r.columns = ['index', 'visit_number_unified', predictor + '_measured']
        measured = z_score

    predicted_value = pd.DataFrame(m_combined)
    predicted_value.set_index(df_visit_dates.index, inplace=True)
    predicted_value.columns = df_visit_dates.columns
    df_pred = predicted_value.reset_index().melt(id_vars='index').dropna()
    df_pred.columns = ['index', 'visit_number_unified', predictor + '_predicted']

    lln = np.exp(np.log(1 - 1.644 * l_combined * s_combined)/l_combined + np.log(m_combined))
    lln = pd.DataFrame(lln).set_index(df_visit_dates.index)
    lln.columns = df_visit_dates.columns
    df_lln = lln.reset_index().melt(id_vars='index').dropna()
    df_lln.columns = ['index', 'visit_number_unified', predictor + '_lln']

    # forward:
    z_score = (np.power((measured.loc[idx, :].to_numpy()/m_combined), l_combined) - 1) / (l_combined * s_combined)
    z_score = pd.DataFrame(z_score).set_index(df_visit_dates.index)
    z_score.columns = df_visit_dates.columns
    df_z = z_score.reset_index().melt(id_vars='index').dropna()
    df_z.columns = ['index', 'visit_number_unified', predictor + '_z_score']

    df = df_z.merge(df_lln, how='left', on=['index', 'visit_number_unified'])
    df = df.merge(df_pred, how='left', on=['index', 'visit_number_unified'])
    df = df.merge(df_r, how='left', on=['index', 'visit_number_unified'])

    return df

