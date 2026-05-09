import copy
import pandas as pd


def extract_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    df_demo = df.dropna(subset=['02_q_3a_dem_sex_at_birth'])
    missing_height_idxs = df_demo.loc[df_demo.loc[:, '03_q_3a_exm_height'].isna(), :].index
    heights = df.loc[missing_height_idxs, :].dropna(subset=['03_q_3a_exm_height'])
    df_demo.loc[missing_height_idxs, '03_q_3a_exm_height'] = heights.loc[heights.loc[:, 'visit_number_unified'] == 'Baseline', '03_q_3a_exm_height']

    age = pd.to_datetime(df.loc[:, '02_visit_date'].dropna(), format='mixed') - df.loc[:, '02_q_2adate_dem_date_of_birth_numeric'].dropna()
    df_demo.insert(len(df_demo.columns), 'age_years_at_baseline', age.dt.days/365.25)

    df_demo_relapse = df_demo.copy()
    df_demo_relapse.index = [pid[:-1] + '2' for pid in list(df_demo_relapse.index)]
    df_demo = pd.concat([df_demo, df_demo_relapse], axis=0)

    df_demo = df_demo[~df_demo.index.duplicated(keep='first')]

    return df_demo.copy()


def merge_demographics(df: pd.DataFrame, df_demographic: pd.DataFrame, idx: int):
    df = pd.merge(left=df, right=df_demographic.loc[:, ['02_q_3a_dem_sex_at_birth', '02_visit_date']],
                  left_index=True, right_index=True)

    df.insert(len(df.columns), 'days_in_study',
              pd.to_timedelta(df.loc[:, str(idx) + '_visit_date'] - df.loc[:, '02_visit_date'], 'Days'))

    return df
