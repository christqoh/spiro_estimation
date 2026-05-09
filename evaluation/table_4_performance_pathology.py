from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from evaluation.utils import bootstrap_confidence_intervals_roc_auc


def evaluate_performance_by_pathology(df: pd.DataFrame, path: str, run: str, n_iter: int):
    rows = ['Overall', 'CXR Normal', 'CXR Abnormal', 'Cavities: True', 'Cavities: False',
            'Infiltration: True', 'Infiltration: False',
            'Loss of Lobar Volume: True', 'Loss of Lobar Volume: False',
            'no_impairment', 'restriction', 'obstruction', 'mixed', 'Female', 'Male',
            'HIV Status: Positive', 'HIV Status: Negative']  #, 'none', 'mild', 'moderate', 'severe']
    params = ['fev1', 'fvc']

    df_tgt = pd.DataFrame()

    for r in tqdm(rows, desc='Performance Evaluation'):
        for param in params:
            # for c in columns:
            if r in ['Overall']:
                # mask = np.ones(shape=df.shape[0]).astype(bool)
                df_s = df
                cut_low = -2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['CXR Normal', 'CXR Abnormal']:
                mask = (df.loc[:, 'result'] == r)
                df_s = df.loc[mask, :]
                cut_low = - 2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['Cavities: True', 'Cavities: False']:
                mask = (df.loc[:, 'cavity_present'] == r)
                df_s = df.loc[mask, :]
                cut_low = - 2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['Infiltration: True', 'Infiltration: False']:
                mask = (df.loc[:, 'infiltrations'] == r)
                df_s = df.loc[mask, :]
                cut_low = - 2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['Loss of Lobar Volume: True', 'Loss of Lobar Volume: False']:
                mask = (df.loc[:, 'lobar_volume_loss'] == r)
                df_s = df.loc[mask, :]
                cut_low = - 2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low


            elif r in ['no_impairment', 'restriction', 'obstruction', 'mixed']: #, cut in  zip(['no_impairment', 'restriction', 'obstruction', 'mixed'], [-1.64, -2.5, -2.5, -2.5]):
                mask = (df.loc[:, 'phenotype'] == r)
                df_s = df.loc[mask, :]
                cut_low = -2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['Female', 'Male']:
                mask = (df.loc[:, '02_q_3a_dem_sex_at_birth'] == r)
                df_s = df.loc[mask, :]
                cut_low = -2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['HIV Status: Positive', 'HIV Status: Negative']:
                mask = (df.loc[:, 'hiv_positive'] == r)
                df_s = df.loc[mask, :]
                cut_low = -2.5
                ch = None
                tgt = df_s.loc[:, param + '_label'] > cut_low

            elif r in ['none']:
                mask = (df.loc[:, 'severity'] == r)
                df_s = df.loc[mask, :]
                cut_low = -1.64
                cut_high = 10
                ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

            elif r in ['mild']:
                mask = (df.loc[:, 'severity'] == r)
                df_s = df.loc[mask, :]
                cut_low = -2.5
                cut_high = -2.0
                ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

            elif r in ['moderate']:
                mask = (df.loc[:, 'severity'] == r)
                df_s = df.loc[mask, :]
                cut_low = -4.0
                cut_high = -2.5
                ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

            elif r in ['severe']:
                mask = (df.loc[:, 'severity'] == r)
                df_s = df.loc[mask, :]
                cut_low = -10.0
                cut_high = -4.0
                ch = cut_high*np.ones_like(df_s.loc[:, param].to_numpy())
                tgt = (df_s.loc[:, param + '_label'] > cut_low) * (df_s.loc[:, param + '_label'] < cut_high)

            else:
                raise Exception('miss-specified')

            if df_s.shape[0] <= 1:
                for metric, k in zip(['AUC', 'F1', 'Precision', 'Recall'], ['auc', 'f1', 'prec', 'rec']):
                    df_tgt.loc[r + ' ' + param.upper() + ' (n=' + str(df_s.shape[0]) + ')', metric] = 'NA'
                continue

            res = bootstrap_confidence_intervals_roc_auc(estimates=df_s.loc[:, param].to_numpy(),
                                                         targets=tgt.to_numpy(),
                                                         threshold=cut_low*np.ones_like(df_s.loc[:, param].to_numpy()),
                                                         threshold_up=ch, n_iter=n_iter)

            for metric, k in zip(['AUC', 'F1', 'Precision', 'Recall'], ['auc', 'f1', 'prec', 'rec']):
                if np.isnan(res.get(k + '_mean')):
                    df_tgt.loc[r + ' ' + param.upper() + ' (n=' + str(df_s.shape[0]) + ')', metric] = 'NA'
                else:
                    df_tgt.loc[r + ' ' + param.upper() + ' (n=' + str(df_s.shape[0]) + ')', metric] = (
                                str(np.round(res.get(k + '_mean'), 3))
                                + ' (' + str(np.round(res.get(k + '_low'), 3)) + '-'
                                + str(np.round(res.get(k + '_high'), 3)) + ')')

    df_tgt.to_excel(path + '/model_performance_by_pathology.xlsx')
    df_tgt.to_csv(path + '/model_performance_by_pathology.csv')
