from copy import deepcopy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='serif')
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r", "g"], N=256)

from scipy.stats import norm

from sklearn.metrics import auc, roc_curve
from evaluation.utils import compute_confidence_interval, compute_metrics


def add_probabilistic_voting(data, run: str, tolerance: float = 0.1, n_iter: int = 100):
    means = data.get(run).get('estimate_mean_weighted')
    stds = data.get(run).get('estimate_std_weighted')
    target = data.get(run).get('label_tgt')
    target_cont = data.get(run).get('target_cont')
    pids = data.get(run).get('pid')
    site = np.array([int(x[0]) for x in pids])

    lower_threshold = -2.5 - tolerance
    upper_threshold = -2.5 + tolerance

    res = dict(tolerance=tolerance)
    features = ['FEV1', 'FVC']

    for idx, f in zip(range(len(features)), features):
        rng = np.random.default_rng(1111)
        probs, estimates, targets, targets_continuous, site_indicators = [], [], [], [], []

        for i in range(n_iter):
            indices = rng.choice(means.shape[0], size=means.shape[0], replace=True)

            m = means[indices, idx]
            s = stds[indices, idx]
            t = target[indices, idx]
            tc = target_cont[indices, idx]
            si = site[indices]

            p_negative = norm.cdf(lower_threshold, loc=m.squeeze(), scale=s.squeeze())
            p_positive = 1 - norm.cdf(upper_threshold, loc=m.squeeze(), scale=s.squeeze())
            p_uncertain = norm.cdf(upper_threshold, loc=m.squeeze(), scale=s.squeeze()) - p_negative
            probabilities = np.stack([p_negative, p_uncertain, p_positive], axis=1)  # Shape: (n_samples, 3)
            probs.append(probabilities)
            estimates.append(m)
            targets.append(t)
            targets_continuous.append(tc)
            site_indicators.append(si)

        res[f + '_cat_probabilities'] = probs
        res[f + '_cat_estimates'] = estimates
        res[f + '_cat_targets'] = targets
        res[f + '_cat_targets_cont'] = targets_continuous
        res[f + '_cat_sites'] = site_indicators

    if 'certainty' in data[run].keys():
        data[run]['certainty'][tolerance] = res
    else:
        k = str(tolerance)
        data[run]['certainty'] = {k: res}

    return data.copy()


def generate_sens_spec_overview_table(data, path, args):
    certainty_dict = data.get(args.nll_z_pef).get('certainty')
    prob_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    df_result = pd.DataFrame()
    df_result_summary = pd.DataFrame()

    for feature in ['FEV1', 'FVC']:
        for tolerance in certainty_dict.keys():
            estimates = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_estimates')
            probs = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_probabilities')
            targets = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_targets')

            for thresh in prob_thresholds:
                rejection_share, aucs = [], []
                sens, spec, fprs = [], [], []

                for est, prob, tgt in zip(estimates, probs, targets):
                    prob = deepcopy(prob)
                    prob[prob >= thresh] = 1.0
                    prob[prob < thresh] = 0.0

                    p = (prob[:, 0] + prob[:, 2]).astype(bool)  # ignore uncertain clas
                    rejection_share.append(100 - ((p.sum() * 100) / prob.shape[0]))

                    unique, counts = np.unique(tgt[p], return_counts=True)
                    class_weights = {cls: 1.0 / count for cls, count in zip(unique, counts)}

                    # Create sample weights
                    sample_weights = np.array([class_weights[label] for label in tgt[p]])
                    sample_weights *= len(tgt[p]) / sample_weights.sum()

                    fpr, tpr, thresholds = roc_curve(tgt[p], est[p], pos_label=True, sample_weight=sample_weights)
                    aucs.append(auc(fpr, tpr))

                    youden_index = tpr - fpr
                    optimal_idx = np.argmax(youden_index)
                    optimal_threshold = thresholds[optimal_idx]

                    binary = est[p] > optimal_threshold
                    sensitivity, specificity, fpr = compute_metrics(tgt[p], binary)
                    sens.append(sensitivity)
                    spec.append(specificity)
                    fprs.append(fpr)

                # compute ci
                sens_mean = np.nanmean(sens)
                sens_low, sens_high , _= compute_confidence_interval(value_list=sens, n=len(estimates))

                spec_mean = np.nanmean(spec)
                spec_low, spec_high, _ = compute_confidence_interval(value_list=spec, n=len(estimates))

                fprs_mean = np.nanmean(fprs)
                fprs_low, fprs_high, _ = compute_confidence_interval(value_list=fprs, n=len(estimates))

                count_mean = np.mean(rejection_share)
                count_low, count_up, _ = compute_confidence_interval(value_list=rejection_share, n=len(estimates))

                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Sensitivity'] = np.round(sens_mean, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Sensitivity [99% CI low]'] = np.round(sens_low, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Sensitivity [99% CI hig]'] = np.round(sens_high, 3)
                df_result_summary.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh),
                'Sensitivity'] = str(np.round(sens_mean, 3)) + ' [' + str(np.round(sens_low, 3)) + ', ' + str(np.round(sens_high, 3)) + ']'

                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Specificity'] = np.round(spec_mean, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Specificity [99% CI low]'] = np.round(spec_low, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Specificity [99% CI high]'] = np.round(spec_high, 3)
                df_result_summary.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh),
                'Specificity'] = str(np.round(spec_mean, 3)) + ' [' + str(np.round(spec_low, 3)) + ', ' + str(np.round(spec_high, 3)) + ']'

                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'FPR'] = np.round(fprs_mean, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'FPR [99% CI low]'] = np.round(fprs_low, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'FPR [99% CI high]'] = np.round(fprs_high, 3)
                df_result_summary.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh),
                'FPR'] = str(np.round(fprs_mean, 3)) + ' [' + str(np.round(fprs_low, 3)) + ', ' + str(np.round(fprs_high, 3)) + ']'

                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Count'] = np.round(count_mean, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Count [99% CI low]'] = np.round(count_low, 3)
                df_result.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh), 'Count [99% CI high]'] = np.round(count_up, 3)
                df_result_summary.loc[feature + ' - ' + str(tolerance) + ' - ' + str(thresh),
                'Count'] = str(np.round(count_mean, 3)) + ' [' + str(np.round(count_low, 3)) + ', ' + str(np.round(count_up, 3)) + ']'

    df_result.to_excel(path + '/sens_spec_thresholding.xlsx')
    df_result_summary.to_excel(path + '/sens_spec_thresholding_condensed.xlsx')


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=90.0, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True, length=0.1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=90.0, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=0.0, ha="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None,
                     data_q1=None, data_q3=None, se = None,
                     n=None,
                     valfmt="{x:.3f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)


    data_se = np.round(se, 5)
    offset = 0.05

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            # t ='\n' + rf"[{data_q1[i, j]}, {data_q3[i, j]}]"
            main_text = f"{data[i, j]:.3f}"
            sub_text = f"(±{data_se[i, j]:.4f})"
            n_text = 'uncertain [%]:\n' + str(n[i, j])

            text_n = im.axes.text(j, i - 0.3 + offset, n_text, fontsize=9, ha='center', va='center', color='white')

            # Place the main text with larger font
            text_main = im.axes.text(j, i + offset, main_text, fontsize=13, ha='center', va='center', color='white')

            # Place the sub text with smaller font below the main text
            text_sub = im.axes.text(j, i + 0.2 + offset, sub_text, fontsize=9, ha='center', va='center', color='white')

            #t = rf"$\mathbf{{{data[i, j]}}}$" + "\n" + rf"\text{{\fontsize{8}{10}\selectfont [{data_q1[i, j]}, {data_q3[i, j]}]}}"
            #text = im.axes.text(j, i, t, **kw)
            # text_2 = im.axes.text(j, i, t, fontisze=8, **kw)
            texts.append(text_main)

    return texts


def generate_figure(data, path, args):
    certainty_dict = data.get(args.nll_z_pef).get('certainty')
    prob_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    fs = 12  # fontsize

    labels = ['FEV1', 'FVC']
    bests_10 = dict(fev1=dict(mean=0.0, low=0.0, high=0.0), fvc=dict(mean=0.0, low=0.0, high=0.0))
    bests_20 = dict(fev1=dict(mean=0.0, low=0.0, high=0.0), fvc=dict(mean=0.0, low=0.0, high=0.0))
    bests_30 = dict(fev1=dict(mean=0.0, low=0.0, high=0.0), fvc=dict(mean=0.0, low=0.0, high=0.0))
    labels_long = ['FEV1', 'FVC']

    fig, ax = plt.subplots(1, 2, figsize=(14, 9)) #, gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(top=0.8, bottom=0.25)

    for idx, feature, head, b in zip(range(len(labels)), labels, labels_long, bests_10.keys()):
        df_auc = pd.DataFrame()
        df_auc_99_lower = pd.DataFrame()
        df_auc_99_upper = pd.DataFrame()
        df_auc_se = pd.DataFrame()
        df_n = pd.DataFrame()

        for tolerance in certainty_dict.keys():
            estimates = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_estimates')
            probs = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_probabilities')
            targets = data.get(args.nll_z_pef).get('certainty').get(tolerance).get(feature + '_cat_targets')

            for thresh in prob_thresholds:
                rejection_share, aucs = [], []

                for est, prob, tgt in zip(estimates, probs, targets):
                    prob = deepcopy(prob)
                    prob[prob >= thresh] = 1.0
                    prob[prob < thresh] = 0.0

                    p = (prob[:, 0] + prob[:, 2]).astype(bool)  # ignore uncertain clas
                    rejection_share.append(100 - ((p.sum() * 100) / prob.shape[0]))

                    unique, counts = np.unique(tgt[p], return_counts=True)
                    class_weights = {cls: 1.0 / count for cls, count in zip(unique, counts)}

                    # Create sample weights
                    sample_weights = np.array([class_weights[label] for label in tgt[p]])
                    sample_weights *= len(tgt[p]) / sample_weights.sum()

                    fpr, tpr, thresholds = roc_curve(tgt[p], est[p], pos_label=True, sample_weight=sample_weights)
                    aucs.append(auc(fpr, tpr))

                # compute ci
                auc_mean = np.mean(aucs)
                auc_low, auc_up, auc_standard_error = compute_confidence_interval(value_list=aucs, n=len(estimates))

                count_mean = np.mean(rejection_share)
                count_low, count_up, share_standard_error = compute_confidence_interval(value_list=rejection_share, n=len(estimates))

                if count_mean < 10 and auc_mean > bests_10[b]['mean']:
                    bests_10[b]['mean'] = auc_mean
                    bests_10[b]['low'] = auc_low
                    bests_10[b]['high'] = auc_up
                    bests_10[b]['aucs'] = aucs
                if count_mean < 20 and auc_mean > bests_20[b]['mean']:
                    bests_20[b]['mean'] = auc_mean
                    bests_20[b]['low'] = auc_low
                    bests_20[b]['high'] = auc_up
                    bests_20[b]['aucs'] = aucs
                if count_mean < 30 and auc_mean > bests_30[b]['mean']:
                    bests_30[b]['mean'] = auc_mean
                    bests_30[b]['low'] = auc_low
                    bests_30[b]['high'] = auc_up
                    bests_30[b]['aucs'] = aucs

                df_auc.loc[tolerance, thresh] = np.round(auc_mean, 5)
                df_auc_99_lower.loc[tolerance, thresh] = np.round(auc_low, 5)
                df_auc_99_upper.loc[tolerance, thresh] = np.round(auc_up, 5)
                df_auc_se.loc[tolerance, thresh] = np.round(auc_standard_error, 5)
                df_n.loc[tolerance, thresh] = f"{count_mean:.2f} (±{share_standard_error:.2f})"

        vmin = df_auc.to_numpy().min() * 0.95
        vmax = df_auc.to_numpy().max() * 1.0
        kwargs = {'vmin': vmin,
                  'vmax': vmax,
                  # 'norm': mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                  }
        cm = 'Blues'

        im = heatmap(df_auc.to_numpy(), ax=ax[idx], cmap=cm,
                     row_labels=list(df_auc.index), col_labels=[str(int(x))+' %' for x in list(df_auc.columns*100)],
                     cbarlabel='', **kwargs)

        ax[idx].set_title(head, fontsize=fs+4)
        ax[idx].set_ylabel('Bandwidth either side of threshold [z-scores]', fontsize=fs)
        ax[idx].set_xlabel('Minimum required share of CDF outside corridor', fontsize=fs)

        texts = annotate_heatmap(im=im, data_q1=df_auc_99_lower.to_numpy(), data_q3=df_auc_99_upper.to_numpy(),
                                 se=df_auc_se.to_numpy(),
                                 n=df_n.to_numpy(), valfmt="{x:.3f}")

    plt.tight_layout()
    plt.savefig(path + '/probabilistic_evaluation.png', format='png', dpi=600, bbox_inches='tight')
    plt.savefig(path + '/probabilistic_evaluation.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(path + '/probabilistic_evaluation.eps', format='eps', bbox_inches='tight')

    plt.close()

    return bests_10, bests_20, bests_30
