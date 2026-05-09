from tqdm import tqdm

import pandas as pd
import numpy as np

from evaluation.utils import bootstrap_confidence_intervals_roc_auc, estimates_to_probs


def evaluate_performance_by_time(df: pd.DataFrame, path: str, n_iter: int,
                                 prob_threshold: float = 0.0, z_score_tolerance: float = 0.0):
    rows = ['Overall',
            'CXR Normal', 'CXR Abnormal',
            'Cavities: True', 'Cavities: False',
            'Infiltration: True', 'Infiltration: False',
            'Loss of Lobar Volume: True', 'Loss of Lobar Volume: False',
            'no_impairment', 'restriction', 'obstruction', 'mixed', 'Female', 'Male',
            'HIV Status: Positive', 'HIV Status: Negative']  #, 'none', 'mild', 'moderate', 'severe']
    params = ['fev1', 'fvc']

    df_tgt = pd.DataFrame()

    for r in tqdm(rows, desc='Performance Evaluation'):
        for visit in ['M00', 'M06', 'M24']:
            for param in params:
                # for c in columns:
                if r in ['Overall']:
                    mask = (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['CXR Normal', 'CXR Abnormal']:
                    mask = (df.loc[:, 'result'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = - 2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['Cavities: True', 'Cavities: False']:
                    mask = (df.loc[:, 'cavity_present'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = - 2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['Infiltration: True', 'Infiltration: False']:
                    mask = (df.loc[:, 'infiltrations'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = - 2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['Loss of Lobar Volume: True', 'Loss of Lobar Volume: False']:
                    mask = (df.loc[:, 'lobar_volume_loss'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = - 2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['no_impairment', 'restriction', 'obstruction', 'mixed']: #, cut in  zip(['no_impairment', 'restriction', 'obstruction', 'mixed'], [-1.64, -2.5, -2.5, -2.5]):
                    mask = (df.loc[:, 'phenotype'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['Female', 'Male']:
                    mask = (df.loc[:, '02_q_3a_dem_sex_at_birth'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['HIV Status: Positive', 'HIV Status: Negative']:
                    mask = (df.loc[:, 'hiv_positive'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low

                elif r in ['none']:
                    mask = (df.loc[:, 'severity'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -1.64
                    cut_high = 10
                    ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                    tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

                elif r in ['mild']:
                    mask = (df.loc[:, 'severity'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    cut_high = -2.0
                    ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                    tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

                elif r in ['moderate']:
                    mask = (df.loc[:, 'severity'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -4.0
                    cut_high = -2.5
                    ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                    tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

                elif r in ['severe']:
                    mask = (df.loc[:, 'severity'] == r) & (df.loc[:, 'visit'] == visit)
                    df_s = df.loc[mask, :]
                    cut_low = -10.0
                    cut_high = -4.0
                    ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                    tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

                else:
                    raise Exception('miss-specified')

                # threshold based on CDF
                m = df_s.loc[:, param].to_numpy()
                s = df_s.loc[:, param + '_std'].to_numpy()

                lower_threshold = -2.5 - z_score_tolerance
                upper_threshold = -2.5 + z_score_tolerance

                if df_s.shape[0] <= 5:
                    for metric, k in zip(['AUC'], ['auc']):
                        df_tgt.loc[r + ' ' + param.upper(), visit] = 'NA (n=' + str(df_s.shape[0]) + ')'
                    continue

                mask = estimates_to_probs(mean=m.squeeze(), scale=s.squeeze(),
                                          lower_threshold=lower_threshold, upper_threshold=upper_threshold,
                                          prob_threshold=prob_threshold)

                estimate = df_s.loc[mask, param].to_numpy()
                tgt = tgt.loc[mask].to_numpy()

                res = bootstrap_confidence_intervals_roc_auc(estimates=estimate, targets=tgt,
                                                             threshold=cut_low*np.ones_like(estimate),
                                                             threshold_up=ch, n_iter=n_iter)

                # for metric, k in zip(['AUC', 'F1', 'Precision', 'Recall'], ['auc', 'f1', 'prec', 'rec']):
                for metric, k in zip(['AUC'], ['auc']):
                    if np.isnan(res.get(k + '_mean')):
                        df_tgt.loc[r + ' ' + param.upper(), visit] = 'NA (n=' + str(df_s.shape[0]) + ')'
                    else:
                        df_tgt.loc[r + ' ' + param.upper() , visit] = (
                                    str(np.round(res.get(k + '_mean'), 3))
                                    + ' (' + str(np.round(res.get(k + '_low'), 3)) + '-'
                                    + str(np.round(res.get(k + '_high'), 3)) + ')'
                                    + ' (n=' + str(df_s.shape[0]) + ')')

    df_tgt = df_tgt.reindex(sorted(df_tgt.columns), axis=1)

    df_tgt.to_csv(path + '/model_performance_by_pathology_time_thresholded_' + str(prob_threshold) + '_' + str(z_score_tolerance) + '.csv')
