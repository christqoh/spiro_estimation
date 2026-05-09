import pandas as pd
import numpy as np


def benchmark_runs(data, bests_10, bests_20, bests_30, args, path):
    df = pd.DataFrame()
    df_means = pd.DataFrame()

    feature = ['FEV1', 'FVC']
    for f in feature:
        # base = f + '_ensemble_indep_comb'
        base = f + '_bootstrap'
        df.loc[f + ': CXR raw [liters]', 'rmse'] = (
                            str(np.round(data.get(args.rmse_raw).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.rmse_raw).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.rmse_raw).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [liters]', 'rmse'] = (
                            str(np.round(data.get(args.rmse_l).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.rmse_l).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.rmse_l).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [z-score]', 'rmse'] =(
                            str(np.round(data.get(args.rmse_z).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.rmse_z).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.rmse_z).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR & PEF [z-score]', 'rmse'] = (
                            str(np.round(data.get(args.rmse_z_pef).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.rmse_z_pef).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.rmse_z_pef).get(base).get('auc_high'), 3)) + ')')

        df_means.loc[f + ': CXR raw [liters]', 'rmse'] = data.get(args.rmse_raw).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [liters]', 'rmse'] = data.get(args.rmse_l).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [z-score]', 'rmse'] = data.get(args.rmse_z).get(base).get('auc_mean')
        df_means.loc[f + ': CXR & PEF [z-score]', 'rmse'] = data.get(args.rmse_z_pef).get(base).get('auc_mean')

        df.loc[f + ': CXR raw [liters]', 'huber'] = (
                            str(np.round(data.get(args.huber_raw).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.huber_raw).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.huber_raw).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [liters]', 'huber'] = (
                            str(np.round(data.get(args.huber_l).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.huber_l).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.huber_l).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [z-score]', 'huber'] =(
                            str(np.round(data.get(args.huber_z).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.huber_z).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.huber_z).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR & PEF [z-score]', 'huber'] = (
                            str(np.round(data.get(args.huber_z_pef).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.huber_z_pef).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.huber_z_pef).get(base).get('auc_high'), 3)) + ')')

        df_means.loc[f + ': CXR raw [liters]', 'huber'] = data.get(args.huber_raw).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [liters]', 'huber'] = data.get(args.huber_l).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [z-score]', 'huber'] = data.get(args.huber_z).get(base).get('auc_mean')
        df_means.loc[f + ': CXR & PEF [z-score]', 'huber'] = data.get(args.huber_z_pef).get(base).get('auc_mean')

        df.loc[f + ': CXR raw [liters]', 'neg. log-likelihood'] = (
                            str(np.round(data.get(args.nll_raw).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.nll_raw).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.nll_raw).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [liters]', 'neg. log-likelihood'] = (
                            str(np.round(data.get(args.nll_l).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.nll_l).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.nll_l).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR [z-score]', 'neg. log-likelihood'] =(
                            str(np.round(data.get(args.nll_z).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.nll_z).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.nll_z).get(base).get('auc_high'), 3)) + ')')
        df.loc[f + ': CXR & PEF [z-score]', 'neg. log-likelihood'] = (
                            str(np.round(data.get(args.nll_z_pef).get(base).get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.nll_z_pef).get(base).get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.nll_z_pef).get(base).get('auc_high'), 3)) + ')')

        df_means.loc[f + ': CXR raw [liters]', 'neg. log-likelihood'] = data.get(args.nll_raw).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [liters]', 'neg. log-likelihood'] = data.get(args.nll_l).get(base).get('auc_mean')
        df_means.loc[f + ': CXR [z-score]', 'neg. log-likelihood'] = data.get(args.nll_z).get(base).get('auc_mean')
        df_means.loc[f + ': CXR & PEF [z-score]', 'neg. log-likelihood'] = data.get(args.nll_z_pef).get(base).get('auc_mean')

        df.loc[f + ': CXR & PEF [weighted z-score]', 'neg. log-likelihood'] = (
                            str(np.round(data.get(args.nll_z_pef).get(f + '_bootstrap_weighted').get('auc_mean'), 3)) + ' ('
                            + str(np.round(data.get(args.nll_z_pef).get(f + '_bootstrap_weighted').get('auc_low'), 3)) + ', '
                            + str(np.round(data.get(args.nll_z_pef).get(f + '_bootstrap_weighted').get('auc_high'), 3)) + ')')

        df.loc[f + ': CXR & PEF [up to 10% uncertain z-score]', 'neg. log-likelihood'] = (
                            str(np.round(bests_10.get(f.lower()).get('mean'), 3)) + ' ('
                            + str(np.round(bests_10.get(f.lower()).get('low'), 3)) + ', '
                            + str(np.round(bests_10.get(f.lower()).get('high'), 3)) + ')')
        df.loc[f + ': CXR & PEF [up to 20% uncertain z-score]', 'neg. log-likelihood'] = (
                            str(np.round(bests_20.get(f.lower()).get('mean'), 3)) + ' ('
                            + str(np.round(bests_20.get(f.lower()).get('low'), 3)) + ', '
                            + str(np.round(bests_20.get(f.lower()).get('high'), 3)) + ')')
        df.loc[f + ': CXR & PEF [up to 30% uncertain z-score]', 'neg. log-likelihood'] = (
                            str(np.round(bests_30.get(f.lower()).get('mean'), 3)) + ' ('
                            + str(np.round(bests_30.get(f.lower()).get('low'), 3)) + ', '
                            + str(np.round(bests_30.get(f.lower()).get('high'), 3)) + ')')

        df_means.loc[f + ': CXR & PEF [up to 10% uncertain z-score]', 'neg. log-likelihood'] = bests_10.get(f.lower()).get('mean')
        df_means.loc[f + ': CXR & PEF [up to 20% uncertain z-score]', 'neg. log-likelihood'] = bests_20.get(f.lower()).get('mean')
        df_means.loc[f + ': CXR & PEF [up to 30% uncertain z-score]', 'neg. log-likelihood'] = bests_30.get(f.lower()).get('mean')

        #t_stat, p_value = stats.ttest_ind(bests.get(f.lower()).get('aucs'), data.get(args.nll_z_pef).get(base).get('aucs'))
        #df.loc[f + ': CXR & PEF p-val [up to 10% uncertain z-score]', 'neg. log-likelihood'] = np.round(p_value, 5)

        #t_stat, p_value = stats.ttest_ind(bests.get(f.lower()).get('aucs'), data.get(args.huber_z_pef).get(base).get('aucs'))
        #df.loc[f + ': CXR & PEF p-val [up to 10% uncertain z-score]', 'huber'] = np.round(p_value, 5)

        #t_stat, p_value = stats.ttest_ind(bests.get(f.lower()).get('aucs'), data.get(args.rmse_z_pef).get(base).get('aucs'))
        #df.loc[f + ': CXR & PEF p-val [up to 10% uncertain z-score]', 'rmse'] = np.round(p_value, 5)

    df.to_excel(path + '/table_benchmark_runs.xlsx')
    df.to_csv(path + '/table_benchmark_runs.csv')
    df_means.to_csv(path + '/table_benchmark_runs.csv')

    pass
