import pandas as pd
import numpy as np


def pivot_table(df: pd.DataFrame, col_name: str, date_col: bool = False) -> pd.DataFrame:
    if date_col:
        df[col_name] = pd.to_datetime(df.loc[:, col_name], format='%Y-%m-%d')
    df = df.pivot(columns=['visit_number_unified'], values=[col_name])
    df.columns = df.columns.droplevel(0)
    return df


def extract_spirometry_data(df: pd.DataFrame, crf_no: str = '18', crf_letter: str = 'b') -> pd.DataFrame:
    """

    :param df: full TBS1 data frame
    :param crf_no: number of spirometry crf (18 or 19)
    :param crf_letter: ident letter of crf entry (b or f)
    :return: filtered df
    """

    test_type = 'pre' if crf_no == '18' else 'post'

    cols = ['visit_number_unified', 'site_id',
            crf_no + '_visit_date',
            crf_no + '_q_3a_spr' + crf_letter + '_spiro_' + test_type + '_test',
            crf_no + '_q_4a_spr' + crf_letter + '_time_point',
            crf_no + '_q_5a_spr' + crf_letter + '_outcome',
            crf_no + '_q_7a_spr' + crf_letter + '_vital_capacity',
            crf_no + '_q_8a_spr' + crf_letter + '_expiratory_volume',
            crf_no + '_q_9a_spr' + crf_letter + '_peak',
            ]
    df[crf_no + '_visit_date'] = pd.to_datetime(df.loc[:, crf_no + '_visit_date'], format='%Y-%m-%d')
    return df.loc[:, cols]


def augment(df: pd.DataFrame, crf_no: str = '18', crf_letter: str = 'b') -> pd.DataFrame:
    # compute fev1/fvc ratio
    df.insert(len(df.columns), crf_no + "_q_8X_spr" + crf_letter + "_ratio_fev1_fvc",
              df.loc[:, crf_no + '_q_8a_spr' + crf_letter + '_expiratory_volume'] /
              df.loc[:, crf_no + '_q_7a_spr' + crf_letter + '_vital_capacity'])
    return df


def drop_df_na(df: pd.DataFrame, crf_no: str = '18', crf_letter: str = 'b') -> pd.DataFrame:
    # df.dropna(subset=[crf_no + '_q_7a_spr' + crf_letter + '_vital_capacity'], inplace=True)
    df.dropna(subset=[crf_no + '_visit_date'], inplace=True)
    # df.dropna(axis=1, how='all', inplace=True)
    return df


def preprocess_spirometry_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - extracts spirometry data for pre and post dilation from the dataset
    - augments by fev1/fvc ratio
    - filters for satisfactory results
    - adds day 14 values to baseline
    :param df: full tbs1 dataset
    :return: processed spirometry dataset
    """
    df_spiro_pre = extract_spirometry_data(df, crf_no='18', crf_letter='b')
    df_spiro_pre = augment(df_spiro_pre, crf_no='18', crf_letter='b')
    df_spiro_pre = drop_df_na(df_spiro_pre, crf_no='18', crf_letter='b')

    df_spiro_post = extract_spirometry_data(df, crf_no='19', crf_letter='f')
    df_spiro_post = augment(df_spiro_post, crf_no='19', crf_letter='f')
    df_spiro_post = drop_df_na(df_spiro_post, crf_no='19', crf_letter='f')

    # df_spiro.loc[:, 'visit_number_unified'].value_counts()

    return df_spiro_pre.copy()

