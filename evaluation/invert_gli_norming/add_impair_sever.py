import numpy as np
import pandas as pd

from typing import List


def get_lung_impairment_mask(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                             df_fev1: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                             df_fev1_fvc: pd.DataFrame, df_tiff_lln: pd.DataFrame):
    mask_lung_impairment = np.logical_or(df_fvc < df_fvc_lln, df_fev1 < df_fev1_lln)
    mask_lung_impairment = np.logical_or(mask_lung_impairment, df_fev1_fvc < df_tiff_lln)
    return mask_lung_impairment.astype(bool)


def get_restriction_masks(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                          df_fev1_obs: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                          df_ratio: pd.DataFrame, df_ratio_lln: pd.DataFrame,
                          columns: List) -> List:

    mask_lung_impairment = get_lung_impairment_mask(df_fvc=df_fvc, df_fvc_lln=df_fvc_lln,
                                                    df_fev1=df_fev1_obs, df_fev1_lln=df_fev1_lln,
                                                    df_fev1_fvc=df_ratio, df_tiff_lln=df_ratio_lln)

    mask_restriction_general = np.logical_and(df_fvc < df_fvc_lln, df_ratio >= df_ratio_lln)
    mask_restriction_general = np.logical_and(mask_lung_impairment, mask_restriction_general)

    mild = np.logical_and(mask_restriction_general, df_fvc >= 0.85 * df_fvc_lln).astype(bool)
    moderate = np.logical_and(df_fvc >= 0.55 * df_fvc_lln, df_fvc < 0.85 * df_fvc_lln)
    moderate = np.logical_and(mask_restriction_general, moderate).astype(bool)
    severe = np.logical_and(mask_restriction_general, df_fvc < 0.55 * df_fvc_lln).astype(bool)

    return [mild.loc[:, columns], moderate.loc[:, columns], severe.loc[:, columns]]


def get_restriction_masks_z_score(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                                  df_fev1_obs: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                                  df_fev1_fvc_ratio: pd.DataFrame, df_fev1_fvc_ratio_lln: pd.DataFrame,
                                  df_fvc_z_score: pd.DataFrame, columns: List,
                                  cut_mild_mod: float = -2.5, cut_mod_sev: float = -4.0) -> List:

    mask_lung_impairment = get_lung_impairment_mask(df_fvc=df_fvc, df_fvc_lln=df_fvc_lln,
                                                    df_fev1=df_fev1_obs, df_fev1_lln=df_fev1_lln,
                                                    df_fev1_fvc=df_fev1_fvc_ratio, df_tiff_lln=df_fev1_fvc_ratio_lln)

    mask_restriction = np.logical_and(df_fvc < df_fvc_lln, df_fev1_fvc_ratio >= df_fev1_fvc_ratio_lln)
    mask_restriction = np.logical_and(mask_lung_impairment, mask_restriction)

    mild = np.logical_and(mask_restriction, df_fvc_z_score >= cut_mild_mod).astype(bool)
    moderate = np.logical_and(df_fvc_z_score >= cut_mod_sev, df_fvc_z_score < cut_mild_mod)
    moderate = np.logical_and(mask_restriction, moderate).astype(bool)
    severe = np.logical_and(mask_restriction, df_fvc_z_score < cut_mod_sev).astype(bool)

    return [mild.loc[:, columns], moderate.loc[:, columns], severe.loc[:, columns]]


def get_obstruction_masks(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                          df_fev1_obs: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                          df_fev1_fvc_ratio: pd.DataFrame, df_fev1_fvc_ratio_lln: pd.DataFrame,
                          df_fev1_z_score: pd.DataFrame, columns: List,
                          cut_mild_mod: float = -2, cut_mod_sev: float = -2.5) -> List:

    mask_lung_impairment = get_lung_impairment_mask(df_fvc=df_fvc, df_fvc_lln=df_fvc_lln,
                                                    df_fev1=df_fev1_obs, df_fev1_lln=df_fev1_lln,
                                                    df_fev1_fvc=df_fev1_fvc_ratio, df_tiff_lln=df_fev1_fvc_ratio_lln)

    mask_obstruction = np.logical_and(df_fvc >= df_fvc_lln, df_fev1_fvc_ratio < df_fev1_fvc_ratio_lln)
    mask_obstruction = np.logical_and(mask_lung_impairment, mask_obstruction)

    mild = np.logical_and(mask_obstruction, df_fev1_z_score >= cut_mild_mod).astype(bool)
    moderate = np.logical_and(df_fev1_z_score >= cut_mod_sev, df_fev1_z_score < cut_mild_mod)
    moderate = np.logical_and(mask_obstruction, moderate).astype(bool)
    severe = np.logical_and(mask_obstruction, df_fev1_z_score < cut_mod_sev).astype(bool)

    return [mild.loc[:, columns], moderate.loc[:, columns], severe.loc[:, columns]]


def get_mixed_masks(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                    df_fev1_obs: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                    df_fev1_fvc: pd.DataFrame, df_fev1_fvc_lln: pd.DataFrame,
                    df_fev1_z_score: pd.DataFrame, columns: List,
                    cut_mild_mod: float = -2, cut_mod_sev: float = -2.5) -> List:

    mask_lung_impairment = get_lung_impairment_mask(df_fvc=df_fvc, df_fvc_lln=df_fvc_lln,
                                                    df_fev1=df_fev1_obs, df_fev1_lln=df_fev1_lln,
                                                    df_fev1_fvc=df_fev1_fvc, df_tiff_lln=df_fev1_fvc_lln)

    # mixed
    mask_mixed = np.logical_and(df_fvc < df_fvc_lln, df_fev1_fvc < df_fev1_fvc_lln)
    mask_mixed = np.logical_and(mask_lung_impairment, mask_mixed)

    mild = np.logical_and(mask_mixed, df_fev1_z_score >= cut_mild_mod).astype(bool)
    moderate = np.logical_and(df_fev1_z_score >= cut_mod_sev, df_fev1_z_score < cut_mild_mod)
    moderate = np.logical_and(mask_mixed, moderate).astype(bool)
    severe = np.logical_and(mask_mixed, df_fev1_z_score <= cut_mod_sev).astype(bool)

    return [mild.loc[:, columns], moderate.loc[:, columns], severe.loc[:, columns]]


def get_other_masks(df_fvc: pd.DataFrame, df_fvc_lln: pd.DataFrame,
                    df_fev1_obs: pd.DataFrame, df_fev1_lln: pd.DataFrame,
                    df_fev1_fvc: pd.DataFrame, df_fev1_fvc_lln: pd.DataFrame,
                    df_fev1_z_score: pd.DataFrame, columns: List,
                    cut_mild_mod: float = -2, cut_mod_sev: float = -2.5) -> List:

    mask_lung_impairment = get_lung_impairment_mask(df_fvc=df_fvc, df_fvc_lln=df_fvc_lln,
                                                    df_fev1=df_fev1_obs, df_fev1_lln=df_fev1_lln,
                                                    df_fev1_fvc=df_fev1_fvc, df_tiff_lln=df_fev1_fvc_lln)

    mask_fvc_not_impaired = np.logical_and(df_fev1_fvc >= df_fev1_fvc_lln, df_fvc >= df_fvc_lln)
    mask_other = np.logical_and(df_fev1_obs < df_fev1_lln, mask_fvc_not_impaired)
    mask_other = np.logical_and(mask_lung_impairment, mask_other)

    mild = np.logical_and(mask_other, df_fev1_z_score >= cut_mild_mod).astype(bool)
    moderate = np.logical_and(df_fev1_z_score >= cut_mod_sev, df_fev1_z_score < cut_mild_mod)
    moderate = np.logical_and(mask_other, moderate).astype(bool)

    return [mild.loc[:, columns], moderate.loc[:, columns]]


def process_df_impairment_severity(df: pd.DataFrame,
                                   cut_mild_mod: float = -2.0, cut_mod_sev: float = -2.5,
                                   kind: str = '', standard: str = '') -> pd.DataFrame:

    df.set_index(['person_id_complete', 'visit_number_unified'], inplace=True)

    df = df.copy()
    df['18_q_8a_sprb_impairment_phenotype_' + kind] = np.NAN
    df['18_q_8a_sprb_impairment_phenotype_' + kind] = df['18_q_8a_sprb_impairment_phenotype_' + kind].astype(str)

    df = df.copy()
    df['18_q_8a_sprb_impairment_severity_' + kind] = np.NAN
    df['18_q_8a_sprb_impairment_severity_' + kind] = df['18_q_8a_sprb_impairment_severity_' + kind].astype(str)

    df_fvc = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                    values='18_q_7a_sprb_vital_capacity')
    df_fvc_lln = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                        values='18_q_7a_sprb_vital_capacity_gli_lower_limit_normality')
    df_fvc_z = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                      values='18_q_7a_sprb_vital_capacity_gli_z_score')

    df_fev1 = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                     values='18_q_8a_sprb_expiratory_volume')
    df_fev1_lln = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                         values='18_q_8a_sprb_expiratory_volume_gli_lower_limit_normality')
    df_fev1_z = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                       values='18_q_8a_sprb_expiratory_volume_gli_z_score')

    df_ratio = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                     values='18_q_8X_sprb_ratio_fev1_fvc')
    df_ratio_lln = df.reset_index().pivot(index='person_id_complete', columns='visit_number_unified',
                                         values='18_q_8X_sprb_ratio_fev1_fvc_gli_lower_limit_normality')

    # get masks
    mask_impair = get_lung_impairment_mask(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln, df_ratio, df_ratio_lln)
    mask_measure_exists = df_fvc.notna()
    mask_gli_data_exists = df_fvc_lln.notna()
    mask_possible_to_diagnose = np.logical_and(mask_measure_exists, mask_gli_data_exists)
    mask_no_impair = mask_possible_to_diagnose * ~mask_impair
    mask_no_impair = mask_no_impair.loc[:, df_fvc.columns]

    # restriction
    if kind == 'ats_ers_2022':
        mask_restriction_mild, mask_restriction_moderate, mask_restriction_severe = \
            get_restriction_masks_z_score(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln, df_ratio, df_ratio_lln,
                                          df_fvc_z, cut_mild_mod=cut_mild_mod, cut_mod_sev=cut_mod_sev,
                                          columns=df_fvc.columns)
    elif kind == 'rachow_bl':
        mask_restriction_mild, mask_restriction_moderate, mask_restriction_severe = \
            get_restriction_masks(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln, df_ratio, df_ratio_lln,
                                  columns=df_fvc.columns)
    else:
        raise Exception('Spiro impairment standards miss-specified')

    # obstruction
    mask_obstruction_mild, mask_obstruction_moderate, mask_obstruction_severe = \
        get_obstruction_masks(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln, df_ratio, df_ratio_lln,
                              df_fev1_z,
                              cut_mild_mod=cut_mild_mod, cut_mod_sev=cut_mod_sev,
                              columns=df_fvc.columns)

    # mixed
    mask_mixed_mild, mask_mixed_moderate, mask_mixed_severe = \
        get_mixed_masks(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln, df_ratio, df_ratio_lln, df_fev1_z,
                        cut_mild_mod=cut_mild_mod, cut_mod_sev=cut_mod_sev,
                        columns=df_fvc.columns)

    # other
    mask_other_mild, mask_other_moderate = get_other_masks(df_fvc, df_fvc_lln, df_fev1, df_fev1_lln,
                                                           df_ratio, df_ratio_lln, df_fev1_z,
                                                           cut_mild_mod=cut_mild_mod, cut_mod_sev=cut_mod_sev,
                                                           columns=df_fvc.columns)

    # add to large df
    phenotypes = ['no_impairment',
                  'restriction', 'restriction', 'restriction',
                  'obstruction', 'obstruction', 'obstruction',
                  'mixed', 'mixed', 'mixed',
                  'other', 'other']
    severities = ['none',
                  'mild', 'moderate', 'severe',
                  'mild', 'moderate', 'severe',
                  'mild', 'moderate', 'severe',
                  'mild', 'moderate']

    impairments = [mask_no_impair,
                   mask_restriction_mild, mask_restriction_moderate, mask_restriction_severe,
                   mask_obstruction_mild, mask_obstruction_moderate, mask_obstruction_severe,
                   mask_mixed_mild, mask_mixed_moderate, mask_mixed_severe,
                   mask_other_mild, mask_other_moderate]

    for df_imp, ph, sev in zip(impairments, phenotypes, severities):
        df_imp_long = pd.melt(df_imp.reset_index(), id_vars='person_id_complete')
        df_imp_long = df_imp_long.loc[df_imp_long.loc[:, 'value'] == True, :]
        df_imp_long.set_index(['person_id_complete', 'visit_number_unified'], inplace=True)
        df = df.copy()
        df.loc[df_imp_long.index, '18_q_8a_sprb_impairment_phenotype_' + kind + '_' + standard] = ph
        df = df.copy()
        df.loc[df_imp_long.index, '18_q_8a_sprb_impairment_severity_' + kind + '_' + standard] = sev
        df = df.copy()

    df.reset_index(inplace=True)

    return df.copy()
