import pandas as pd

from evaluation.table_supp_62_performance_time import evaluate_performance_by_time
from evaluation.table_4_performance_pathology import evaluate_performance_by_pathology
from evaluation.table_supp_61_performance_site import evaluate_performance_by_site_cutoff


def generate_performance_tables(run_evaluated: str, data_src, path, df_tbs: pd.DataFrame, n_iter: int = 1000):
    # assemble all of this into a dataframe
    df = pd.DataFrame()
    df['pid'] = data_src.get(run_evaluated).get('pid')
    df['visit'] = data_src.get(run_evaluated).get('visit')

    df['fev1_label'] = data_src.get(run_evaluated).get('target_cont')[:, 0]
    df['fev1_label_bool'] = data_src.get(run_evaluated).get('label_tgt')[:, 0]

    df['fvc_label'] = data_src.get(run_evaluated)['target_cont'][:, 1]
    df['fvc_label_bool'] = data_src.get(run_evaluated)['label_tgt'][:, 1]
    df.set_index('pid', inplace=True)

    df['fev1'] = data_src.get(run_evaluated).get('estimate_mean_weighted')[:, 0]
    df['fev1_std'] = data_src.get(run_evaluated)['estimate_std_weighted'][:, 0]
    df['fvc'] = data_src.get(run_evaluated).get('estimate_mean_weighted')[:, 1]
    df['fvc_std'] = data_src.get(run_evaluated)['estimate_std_weighted'][:, 1]

    df_dem = df_tbs.loc[:, ['site_id', '02_q_3a_dem_sex_at_birth', 'hiv_positive']].dropna()
    df = df.merge(df_dem, left_on=df.index, right_on=df_dem.index, how='outer')

    df_cxr = df_tbs.loc[:, ['visit_number_unified', '15_q_6a_cxr_result', '15_q_8a_cxr_cavities',
                            '15_q_9d1_cxr_d_calcifications_status',
                            '15_q_9g1_cxr_g_infiltration_status',
                            '15_q_9i1_cxr_i_lobar_volume_loss_colapse_bronchiectasis_status',
                            '18_q_8a_sprb_impairment_phenotype_ats_ers_2022_other',
                            '18_q_8a_sprb_impairment_severity_ats_ers_2022_other']].dropna(thresh=2, axis=0)
    df_cxr.loc[(df_cxr.loc[:, '15_q_6a_cxr_result'] == 'Normal') * (df_cxr.loc[:, '15_q_8a_cxr_cavities'].isna()), '15_q_8a_cxr_cavities'] = 'No'
    df_cxr.loc[(df_cxr.loc[:, '15_q_6a_cxr_result'] == 'Normal') * (df_cxr.loc[:, '15_q_9g1_cxr_g_infiltration_status'].isna()), '15_q_9g1_cxr_g_infiltration_status'] = 'No'
    df_cxr.loc[(df_cxr.loc[:, '15_q_6a_cxr_result'] == 'Normal') * (df_cxr.loc[:, '15_q_9d1_cxr_d_calcifications_status'].isna()), '15_q_9d1_cxr_d_calcifications_status'] = 'No'
    df_cxr.loc[(df_cxr.loc[:, '15_q_6a_cxr_result'] == 'Normal') * (df_cxr.loc[:, '15_q_9i1_cxr_i_lobar_volume_loss_colapse_bronchiectasis_status'].isna()), '15_q_9i1_cxr_i_lobar_volume_loss_colapse_bronchiectasis_status'] = 'No'
    df_cxr.replace({'Yes': True, 'No': False}, inplace=True)
    df_cxr.reset_index(inplace=True)
    df_cxr = df_cxr.rename(columns={'person_id_complete': 'pid',
                          'visit_number_unified': 'visit',
                          '15_q_6a_cxr_result': 'result',
                          '15_q_8a_cxr_cavities': 'cavity_present',
                          '15_q_9d1_cxr_d_calcifications_status': 'calcifications',
                          '15_q_9g1_cxr_g_infiltration_status': 'infiltrations',
                          '15_q_9i1_cxr_i_lobar_volume_loss_colapse_bronchiectasis_status': 'lobar_volume_loss',
                           '18_q_8a_sprb_impairment_phenotype_ats_ers_2022_other': 'phenotype',
                           '18_q_8a_sprb_impairment_severity_ats_ers_2022_other': 'severity'})
    df_cxr.replace({'Mth 30': 'M30',
                    'Mth 36': 'M36',
                    'Mth 42': 'M42',
                    'Mth 48': 'M48',
                    'Mth 24': 'M24',
                    'Baseline': 'D0',
                    'Mth 06': 'M6'}, inplace=True)

    df = df.merge(df_cxr, left_on=['key_0', 'visit'], right_on=['pid', 'visit'], how='left')

    df.replace({'M30': 'M24', 'M36': 'M24', 'M42': 'M24', 'M48': 'M24', 'S2V1': 'M24', 'D0': 'M00',
                'M6': 'M06'}, inplace=True)

    df.set_index('pid', inplace=True)

    df.loc[df.loc[:, 'hiv_positive'] == True, 'hiv_positive'] = 'HIV Status: Positive'
    df.loc[df.loc[:, 'hiv_positive'] == False, 'hiv_positive'] = 'HIV Status: Negative'

    df.loc[df.loc[:, 'result'] == 'Normal', 'result'] = 'CXR Normal'
    df.loc[df.loc[:, 'result'] == 'Abnormal', 'result'] = 'CXR Abnormal'

    df.loc[df.loc[:, 'cavity_present'] == True, 'cavity_present'] = 'Cavities: True'
    df.loc[df.loc[:, 'cavity_present'] == False, 'cavity_present'] = 'Cavities: False'

    df.loc[df.loc[:, 'calcifications'] == True, 'calcifications'] = 'Calcification'
    df.loc[df.loc[:, 'calcifications'] == False, 'calcifications'] = 'No Calcification'

    df.loc[df.loc[:, 'infiltrations'] == True, 'infiltrations'] = 'Infiltration: True'
    df.loc[df.loc[:, 'infiltrations'] == False, 'infiltrations'] = 'Infiltration: False'

    df.loc[df.loc[:, 'lobar_volume_loss'] == True, 'lobar_volume_loss'] = 'Loss of Lobar Volume: True'
    df.loc[df.loc[:, 'lobar_volume_loss'] == False, 'lobar_volume_loss'] = 'Loss of Lobar Volume: False'

    evaluate_performance_by_pathology(df.copy(), path, run=run_evaluated, n_iter=n_iter)

    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.0, z_score_tolerance=0.0)
    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.6, z_score_tolerance=0.1)
    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.7, z_score_tolerance=0.1)
    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.8, z_score_tolerance=0.1)
    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.6, z_score_tolerance=0.2)
    evaluate_performance_by_time(df.copy(), path, n_iter=n_iter, prob_threshold=0.7, z_score_tolerance=0.2)

    evaluate_performance_by_site_cutoff(df.copy(), path, run=run_evaluated, n_iter=n_iter, prob_threshold=0.0, z_score_tolerance=0.0)
    evaluate_performance_by_site_cutoff(df.copy(), path, run=run_evaluated, n_iter=n_iter, prob_threshold=0.7, z_score_tolerance=0.1)
    evaluate_performance_by_site_cutoff(df.copy(), path, run=run_evaluated, n_iter=n_iter, prob_threshold=0.8, z_score_tolerance=0.1)

    pass
