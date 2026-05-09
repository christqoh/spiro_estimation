from tqdm import tqdm

import pandas as pd
import numpy as np

from evaluation.utils import bootstrap_confidence_intervals_roc_auc, estimates_to_probs


def evaluate_performance_by_site_cutoff(df: pd.DataFrame, path: str, run: str, n_iter: int,
                                        prob_threshold: float = 0.8, z_score_tolerance: float = 0.2):
    rows = ['Overall',
            'Cavities: True', 'Cavities: False',
            'Infiltration: True', 'Infiltration: False',
            'Loss of Lobar Volume: True', 'Loss of Lobar Volume: False',
            'Female', 'Male', 'HIV Status: Positive', 'HIV Status: Negative']
    params = ['fev1', 'fvc']

    df_tgt = pd.DataFrame()

    for r in tqdm(rows, desc='Performance Evaluation'):
        for site in df.loc[:, 'site_id'].unique():
            for param in params:
                if r in ['Overall']:
                    mask = (df.loc[:, 'site_id'] == site)
                    df_s = df.loc[mask, :]
                    cut_low = -2.5
                    ch = None
                    tgt = df_s.loc[:, param + '_label'] > cut_low
                else:
                    raise Exception('miss-specified')

                # threshold based on CDF
                m = df_s.loc[:, param].to_numpy()
                s = df_s.loc[:, param + '_std'].to_numpy()

                lower_threshold = -2.5 - z_score_tolerance
                upper_threshold = -2.5 + z_score_tolerance

                if mask.sum() <= 5:
                    for metric, k in zip(['AUC'], ['auc']):
                        df_tgt.loc[r + ' ' + param.upper(), site] = 'NA (n=' + str(df_s.shape[0]) + ')'
                    continue

                mask = estimates_to_probs(mean=m.squeeze(), scale=s.squeeze(),
                                          lower_threshold=lower_threshold, upper_threshold=upper_threshold,
                                          prob_threshold=prob_threshold)

                estimate = df_s.loc[mask, param].to_numpy()
                tgt = tgt.loc[mask].to_numpy()

                res = bootstrap_confidence_intervals_roc_auc(estimates=estimate, targets=tgt,
                                                             threshold=cut_low*np.ones_like(estimate),
                                                             threshold_up=ch, n_iter=n_iter)

                for metric, k in zip(['AUC'], ['auc']):
                    if np.isnan(res.get(k + '_mean')):
                        df_tgt.loc[r + ' ' + param.upper(), site] = 'NA (n=' + str(df_s.shape[0]) + ')'
                    else:
                        df_tgt.loc[r + ' ' + param.upper() , site] = (
                                    str(np.round(res.get(k + '_mean'), 3))
                                    + ' (' + str(np.round(res.get(k + '_low'), 3)) + '-'
                                    + str(np.round(res.get(k + '_high'), 3)) + ')'
                                    + ' (n=' + str(mask.sum()) + ')')

    df_tgt = df_tgt.reindex(sorted(df_tgt.columns), axis=1)

    df_tgt.to_excel(path + '/model_performance_by_pathology_site_thresholded_' + str(prob_threshold) + '_' + str(z_score_tolerance) + '.xlsx')
    df_tgt.to_csv(path + '/model_performance_by_pathology_site_thresholded_' + str(prob_threshold) + '_' + str(z_score_tolerance) + '.csv')
