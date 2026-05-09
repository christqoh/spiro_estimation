import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
from evaluation.utils import combine_fold_estimates


def plot_bland_altman(predictions, label_tgt, feature_labels, save_dir: str,
                      predictions_std=None, exclude_extreme_values: bool = False, sample: bool = False):

    mu, mu_w, sig_w = combine_fold_estimates(predictions, predictions_std)
    l = label_tgt
    if sample:
        no_samples = 100000
        res = []
        for i in range(no_samples):
            res.append(np.random.normal(loc=mu_w, scale=sig_w, size=(103, 2)))
        mu_w = np.concatenate(res, axis=0)
        l = np.tile(l, (no_samples, 1))
        print(np.corrcoef(mu_w[:, 0], l[:, 0])[0][1])
        print(np.corrcoef(mu_w[:, 1], l[:, 1])[0][1])

    # Regression plot configurations
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical space

    col_labels = ['Mean', 'Mean', 'Mean']
    x_limits = [[-5.5, 2], [-5.5, 2], [1.0, 6.0]]
    y_limits = [[-5.0, 5.0], [-5.0, 5.0], [-3.5, 3.5]]
    label_pos = [1.9, 1.9, 5.9]
    subplot_titles = [[r"$\bf{A}$" + '  FEV1 z-score estimated from CXR, PEFR', r"$\bf{C}$" + '  FEV1 z-score estimated from CXR', r"$\bf{E}$ "+ '  FEV1 z-score estimated from CXR'],
                      [r"$\bf{B}$" + '  FVC z-score estimated from CXR, PEFR', r"$\bf{D}$" + '  FVC z-score estimated from CXR', r"$\bf{F}$" + '  FVC z-score estimated from CXR']]
    res = dict()

    for param, row in zip(feature_labels, range(len(feature_labels))):
        if mu_w is not None:
            prediction = mu_w[:, row]
            variance = sig_w[:, row]**2
        else:
            prediction = mu[:, row]
            variance = None

        target = l[:, row]

        if exclude_extreme_values:
            q = 2.5
            lower_threshold = np.percentile(target, q)
            upper_threshold = np.percentile(target, 100 - q)
            mask = (target > lower_threshold) & (target < upper_threshold)

            prediction = prediction[mask]
            target = target[mask]
            variance = variance[mask] if variance is not None else None

        mean_measurements = (prediction + target) / 2
        differences = prediction - target
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)

        res[param + '_mean_diff'] = mean_diff
        res[param + '_std_diff'] = std_diff

        ax[row].scatter(mean_measurements, differences, color='tab:red', edgecolors=None, s=3, alpha=0.7)
        ax[row].axhline(mean_diff, color='gray', linestyle='--', linewidth=0.8, label="Mean")
        #ax[row].axhline(mean_diff + 2.58 * std_diff, color='gray', linestyle='--', linewidth=0.8)
        #ax[row].axhline(mean_diff - 2.58 * std_diff, color='gray', linestyle='--', linewidth=0.8)

        ax[row].axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', linewidth=0.8)
        ax[row].axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', linewidth=0.8)

        ax[row].set_ylabel('Difference')

        ax[row].set_ylim([-5.5, 5.5])
        ax[row].set_xlim([-5, 2])

        ax[row].text(label_pos[0], mean_diff, 'Mean:\n' + str(np.round(mean_diff, 2)), color='black', fontsize=8,
                          verticalalignment='center', horizontalalignment='right')
        #ax[row].text(label_pos[0], mean_diff - 2.58 * std_diff, '-2.58 SD:\n' + str(np.round(mean_diff - 2.58 * std_diff, 2)), color='black', fontsize=8,
        #                  verticalalignment='center', horizontalalignment='right')
        #ax[row].text(label_pos[0], mean_diff + 2.58 * std_diff, '2.58 SD:\n' + str(np.round(mean_diff + 2.58 * std_diff, 2)), color='black', fontsize=8,
       #                   verticalalignment='center', horizontalalignment='right')

        ax[row].text(label_pos[0], mean_diff - 1.96 * std_diff, '-1.96 SD:\n' + str(np.round(mean_diff - 1.96 * std_diff, 2)), color='black', fontsize=8,
                          verticalalignment='center', horizontalalignment='right')
        ax[row].text(label_pos[0], mean_diff + 1.96 * std_diff, '1.96 SD:\n' + str(np.round(mean_diff + 1.96 * std_diff, 2)), color='black', fontsize=8,
                          verticalalignment='center', horizontalalignment='right')

        ax[row].set_title(subplot_titles[row][0], loc='left')

        ax[row].set_xlabel(col_labels[0])

    plt.savefig(save_dir + '/bland_altman.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(save_dir + '/bland_altman.eps', format='eps', bbox_inches='tight')
    plt.close()

    return res
