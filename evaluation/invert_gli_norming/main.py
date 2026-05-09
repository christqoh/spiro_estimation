from tqdm import tqdm

import pandas as pd

from evaluation.invert_gli_norming.demographics import extract_demographic_data
from evaluation.invert_gli_norming.utils import preprocess_spirometry_data, pivot_table
from evaluation.invert_gli_norming.predictors_and_scores import invert_z_score
from evaluation.invert_gli_norming.add_impair_sever import process_df_impairment_severity


def invert_z_scores(df, standard: str, ending: str = ''):
    df_s = df.set_index('person_id_complete')
    df_demo = extract_demographic_data(df_s)

    cols = ['fev1_z', 'fvc_z']
    vals = ['FEV1', 'FVC']

    for c, v in zip(cols, vals):
        df_wide = pivot_table(df.set_index('person_id_complete'), c)
        r = invert_z_score(df.set_index('person_id_complete'), df_demo, df_wide, predictor=v, standard=standard, backward=True)

        df = df.merge(r, left_on=['person_id_complete', 'visit_number_unified'], right_on=['index', 'visit_number_unified'], how='left')

        df = df.copy()

    df.dropna(subset=['fev1_z', 'fvc_z'], inplace=True)
    df.sort_index(inplace=True, axis=1)

    df_fev1 = pivot_table(df.set_index('person_id_complete'), 'FEV1_measured')
    df_fvc = pivot_table(df.set_index('person_id_complete'), 'FVC_measured')
    df_wide = df_fev1 / df_fvc

    ratio = invert_z_score(df.set_index('person_id_complete'), df_demo, df_wide, predictor='FEV1FVC', standard=standard)
    df = df.merge(ratio, left_on=['person_id_complete', 'visit_number_unified'],
                         right_on=['index', 'visit_number_unified'], how='left')

    rename_dict = {'FEV1_lln': '18_q_8a_sprb_expiratory_volume_gli_lower_limit_normality',
                   'FEV1_measured': '18_q_8a_sprb_expiratory_volume',
                   'person_id_complete': 'person_id_complete',
                   'visit_number_unified': 'visit_number_unified',
                   'FEV1_z_score': '18_q_8a_sprb_expiratory_volume_gli_z_score',
                   'FVC_lln': '18_q_7a_sprb_vital_capacity_gli_lower_limit_normality',
                   'FVC_measured': '18_q_7a_sprb_vital_capacity',
                   'FVC_z_score': '18_q_7a_sprb_vital_capacity_gli_z_score',
                   'FEV1FVC_lln': '18_q_8X_sprb_ratio_fev1_fvc_gli_lower_limit_normality',
                   'FEV1FVC_measured': '18_q_8X_sprb_ratio_fev1_fvc'}

    df_renamed = df.rename(columns=rename_dict)
    df_renamed = process_df_impairment_severity(df_renamed, cut_mild_mod=-2.5, cut_mod_sev=-4.0, kind='ats_ers_2022', standard='other')
    df_renamed.set_index('person_id_complete', inplace=True)
    return df_renamed.copy()
