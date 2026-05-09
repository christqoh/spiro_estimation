from copy import deepcopy
from typing import List

import warnings
import numpy as np

from scipy.stats import norm

from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import UndefinedMetricWarning


# Ignore sklearn warnings caused by undefined precision/recall
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def compute_metrics(target, estimate):
    """
    Compute basic binary classification metrics.

    Parameters
    ----------
    target : np.ndarray
        Ground-truth binary labels.
    estimate : np.ndarray
        Binary predictions.

    Returns
    -------
    sensitivity : float
        True positive rate.
    specificity : float
        True negative rate.
    fpr : float
        False positive rate.
    """

    # Compute confusion matrix values:
    # tn = true negatives
    # fp = false positives
    # fn = false negatives
    # tp = true positives
    tn, fp, fn, tp = confusion_matrix(target, estimate).ravel()

    # Diagnostic classification metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    return sensitivity, specificity, fpr


def compute_auc(target, estimate):
    """
    Compute weighted ROC-AUC score.

    Class balancing is performed using sklearn's
    compute_sample_weight() utility.

    Parameters
    ----------
    target : np.ndarray
        Binary ground-truth labels.
    estimate : np.ndarray
        Continuous prediction scores.

    Returns
    -------
    float
        ROC-AUC score.
    """

    # Balance positive/negative samples
    sample_weights = compute_sample_weight('balanced', target)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(
        target,
        estimate,
        pos_label=True,
        sample_weight=sample_weights
    )

    return auc(fpr, tpr)


def compute_confidence_interval(value_list: List, n: int, z: float = 2.575829):
    """
    Compute confidence interval assuming normality.

    Default:
        z = 2.575829  -> 99% CI

    Common alternatives:
        z = 1.96      -> 95% CI

    Parameters
    ----------
    value_list : List
        List or array of values.
    n : int
        Number of samples / bootstrap iterations.
    z : float
        Z-score defining confidence interval width.

    Returns
    -------
    lower_bound : float
    upper_bound : float
    se : float
        Standard error.
    """

    # Mean and standard deviation while ignoring NaNs
    mean = np.nanmean(value_list)
    std = np.nanstd(value_list)

    # Adjust sample count for NaN values
    n_corrected = n - np.isnan(value_list).sum()

    # Standard error
    se = std / np.sqrt(n_corrected)

    # Confidence interval bounds
    lower_bound = mean - z * se
    upper_bound = mean + z * se

    return lower_bound, upper_bound, se


def combine_fold_estimates(estimates: List, stds: None):
    """
    Combine predictions across folds.

    If predictive uncertainty (standard deviations) is available,
    a variance-weighted ensemble estimate is additionally computed.

    Parameters
    ----------
    estimates : List
        Fold-wise prediction means.
    stds : List or None
        Fold-wise prediction standard deviations.

    Returns
    -------
    mu_basis : np.ndarray
        Simple mean across folds.
    mu_weighted : np.ndarray or None
        Variance-weighted mean.
    sig_weighted : np.ndarray or None
        Weighted predictive uncertainty.
    """

    # Concatenate fold predictions along ensemble dimension
    means = np.concatenate(deepcopy(estimates), axis=2)

    # Simple ensemble mean
    mu_basis = means.mean(axis=-1)

    if stds is not None:

        # Convert standard deviations to variances
        var = np.concatenate(deepcopy(stds), axis=2) ** 2

        # Small constant for numerical stability
        stability = 1e-6

        # Inverse variance weighting
        weighting = 1 / (var + stability)

        # Weighted ensemble mean
        mu_weighted = (
            np.nansum(means * weighting, axis=2)
            / np.nansum(weighting, axis=2)
        )

        # Weighted variance and standard deviation
        var_weighted = 1 / np.nansum(weighting, axis=2)
        sig_weighted = np.sqrt(var_weighted)

    else:
        mu_weighted = None
        sig_weighted = None

    return mu_basis, mu_weighted, sig_weighted


def bootstrap_confidence_intervals_roc_auc(
        estimates: np.ndarray,
        targets: np.ndarray,
        threshold: np.ndarray,
        threshold_up: np.ndarray = None,
        n_iter: int = 1000,
):
    """
    Bootstrap ROC-AUC and classification metrics.

    Parameters
    ----------
    estimates : np.ndarray
        Continuous prediction scores.
    targets : np.ndarray
        Binary ground-truth labels.
    threshold : np.ndarray
        Lower classification threshold.
    threshold_up : np.ndarray, optional
        Optional upper threshold for uncertainty regions.
    n_iter : int
        Number of bootstrap iterations.

    Returns
    -------
    dict
        Dictionary containing metric summaries and ROC statistics.
    """

    rng = np.random.default_rng(1111)

    # Standardized FPR axis for interpolation
    false_positive_rate = np.linspace(0, 1, 100)

    # Containers for bootstrap results
    true_positive_rates = []

    aucs = []
    precs = []
    recs = []
    f1s = []

    sens = []
    spec = []
    fprs = []

    for i in range(n_iter):

        # Bootstrap sample indices
        indices = rng.choice(
            estimates.shape[0],
            size=estimates.shape[0],
            replace=True
        )

        # Resampled data
        p = estimates[indices]
        tgt = targets[indices]
        th = threshold[indices]

        # Compute balanced sample weights
        sample_weights = compute_sample_weight('balanced', tgt)

        # Compute ROC curve
        fpr, tpr, roc_thresholds = roc_curve(
            tgt,
            p,
            pos_label=True,
            sample_weight=sample_weights
        )

        aucs.append(auc(fpr, tpr))

        # Ensure ROC starts at (0,0)
        if fpr[0] > 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)

        # Ensure ROC ends at (1,1)
        if fpr[-1] < 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        # Interpolate ROC curve onto common FPR grid
        true_positive_rates.append(
            np.interp(false_positive_rate, fpr, tpr)[:, None]
        )

        # Binary classification using thresholds
        if threshold_up is not None:
            binary = (p > th) * (p < threshold_up[indices])
        else:
            binary = p > th

        # Precision / Recall / F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            tgt,
            binary,
            average='weighted',
            sample_weight=sample_weights,
            zero_division=np.nan
        )

        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)

        # Youden index (currently not used further)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = roc_thresholds[optimal_idx]

        # Optional sensitivity/specificity evaluation
        # sensitivity, specificity, fpr = compute_metrics(
        #     tgt,
        #     p > optimal_threshold
        # )
        #
        # sens.append(sensitivity)
        # spec.append(specificity)
        # fprs.append(fpr)

    # ------------------------------------------------------------------
    # Aggregate bootstrap statistics
    # ------------------------------------------------------------------

    auc_mean = np.nanmean(aucs)
    auc_low, auc_high, _ = compute_confidence_interval(
        value_list=aucs,
        n=n_iter
    )

    f1_mean = np.nanmean(f1s)
    f1_low, f1_high, _ = compute_confidence_interval(
        value_list=f1s,
        n=n_iter
    )

    prec_mean = np.nanmean(precs)
    prec_low, prec_high, _ = compute_confidence_interval(
        value_list=precs,
        n=n_iter
    )

    rec_mean = np.nanmean(recs)
    rec_low, rec_high, _ = compute_confidence_interval(
        value_list=recs,
        n=n_iter
    )

    sens_mean = np.nanmean(sens)
    sens_low, sens_high, _ = compute_confidence_interval(
        value_list=sens,
        n=n_iter
    )

    spec_mean = np.nanmean(spec)
    spec_low, spec_high, _ = compute_confidence_interval(
        value_list=spec,
        n=n_iter
    )

    fprs_mean = np.nanmean(fprs)
    fprs_low, fprs_high, _ = compute_confidence_interval(
        value_list=fprs,
        n=n_iter
    )

    # Mean ROC curve
    tprs = np.concatenate(true_positive_rates, axis=-1)
    tprs_mean = tprs.mean(axis=-1)

    tprs_low, tprs_high, _ = compute_confidence_interval(
        value_list=tprs_mean,
        n=n_iter
    )

    # Collect results
    results = dict(
        auc_mean=auc_mean,
        auc_low=auc_low,
        auc_high=auc_high,
        aucs=aucs,

        f1_mean=f1_mean,
        f1_low=f1_low,
        f1_high=f1_high,

        prec_mean=prec_mean,
        prec_low=prec_low,
        prec_high=prec_high,

        rec_mean=rec_mean,
        rec_low=rec_low,
        rec_high=rec_high,

        sens_mean=sens_mean,
        sens_low=sens_low,
        sens_high=sens_high,

        spec_mean=spec_mean,
        spec_low=spec_low,
        spec_high=spec_high,

        fprs_mean=fprs_mean,
        fprs_low=fprs_low,
        fprs_high=fprs_high,

        false_positive_rate=false_positive_rate,

        true_positive_rate_mean=tprs_mean,
        tprs_low=tprs_low,
        tprs_high=tprs_high,
        tprs=tprs,
    )

    return results


def combine_fold_votes_aucs(
        estimates: List,
        tgt: np.ndarray,
        feature: int,
        percs: float = 0.5,
):
    """
    Compute ROC statistics independently for each fold and
    combine them afterwards.

    Parameters
    ----------
    estimates : List
        Fold-wise predictions.
    tgt : np.ndarray
        Ground-truth labels.
    feature : int
        Feature index (e.g. FEV1/FVC).
    percs : float
        Percentile threshold for confidence bounds.

    Returns
    -------
    dict
        ROC and AUC statistics.
    """

    aucs = []
    true_positive_rates = []

    false_positive_rate = np.linspace(0, 1, 100)
    n_iter = len(estimates)

    for est in estimates:

        # Current fold predictions
        e = est[:, feature]
        t = tgt[:, feature]

        # Balanced sample weighting
        unique, counts = np.unique(t, return_counts=True)
        class_weights = {
            cls: 1.0 / count
            for cls, count in zip(unique, counts)
        }

        sample_weights = np.array(
            [class_weights[label] for label in t]
        )

        sample_weights *= len(t) / sample_weights.sum()

        # ROC computation
        fpr, tpr, _ = roc_curve(
            t,
            e,
            pos_label=True,
            sample_weight=sample_weights
        )

        aucs.append(auc(fpr, tpr))

        # Ensure full ROC range
        if fpr[0] > 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)

        if fpr[-1] < 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        # Interpolate ROC
        true_positive_rates.append(
            np.interp(false_positive_rate, fpr, tpr)[:, None]
        )

    # Aggregate AUC statistics
    auc_mean = np.mean(aucs)

    # Percentile-based confidence interval
    auc_low, auc_high = np.percentile(
        aucs,
        [percs, 100 - percs]
    )

    # Aggregate TPR curves
    tprs = np.concatenate(true_positive_rates, axis=-1)
    tprs_mean = tprs.mean(axis=-1)

    tprs_low, tprs_high, _ = compute_confidence_interval(
        tprs_mean,
        n_iter
    )

    results = dict(
        auc_mean=auc_mean,
        auc_low=auc_low,
        auc_high=auc_high,

        false_positive_rate=false_positive_rate,

        true_positive_rate_mean=tprs_mean,
        tprs_low=tprs_low,
        tprs_high=tprs_high,
        tprs=tprs,
    )

    return results


def compute_roc_curve(estimate, target, feature: int):
    """
    Compute interpolated ROC curve for a single feature.

    Parameters
    ----------
    estimate : np.ndarray
        Prediction scores.
    target : np.ndarray
        Binary targets.
    feature : int
        Feature index.

    Returns
    -------
    estimate_feature : np.ndarray
        Prediction scores for selected feature.
    fpr_mean : np.ndarray
        Standardized FPR axis.
    tpr : np.ndarray
        Interpolated TPR values.
    auc_score : float
        ROC-AUC score.
    """

    # Standardized ROC interpolation axis
    fpr_mean = np.linspace(0, 1, 100)

    # Compute class balancing weights
    unique, counts = np.unique(target, return_counts=True)

    class_weights = {
        cls: 1.0 / count
        for cls, count in zip(unique, counts)
    }

    sample_weights = np.array(
        [class_weights[label] for label in target[:, feature]]
    )

    sample_weights *= len(target) / sample_weights.sum()

    # ROC computation
    fpr, tpr, thresholds = roc_curve(
        target[:, feature],
        estimate[:, feature].squeeze(),
        pos_label=True,
        sample_weight=sample_weights
    )

    # Ensure complete ROC range
    if fpr[0] > 0:
        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)

    if fpr[-1] < 1:
        fpr = np.append(fpr, 1)
        tpr = np.append(tpr, 1)

    # Interpolate onto common FPR grid
    tpr = np.interp(fpr_mean, fpr, tpr)

    return estimate[:, feature], fpr_mean, tpr, auc(fpr_mean, tpr)


def estimates_to_probs(
        mean,
        scale,
        lower_threshold: float,
        upper_threshold: float,
        prob_threshold: float,
):
    """
    Convert Gaussian prediction estimates into probability classes.

    Classes:
        - negative
        - uncertain
        - positive

    Parameters
    ----------
    mean : np.ndarray
        Predicted means.
    scale : np.ndarray
        Predicted standard deviations.
    lower_threshold : float
        Lower decision boundary.
    upper_threshold : float
        Upper decision boundary.
    prob_threshold : float
        Minimum confidence threshold.

    Returns
    -------
    np.ndarray
        Boolean mask identifying confident predictions.
    """

    # Probability of being below lower threshold
    p_negative = norm.cdf(
        lower_threshold,
        loc=mean,
        scale=scale
    )

    # Probability of being above upper threshold
    p_positive = 1 - norm.cdf(
        upper_threshold,
        loc=mean,
        scale=scale
    )

    # Intermediate uncertainty region
    p_uncertain = (
        norm.cdf(upper_threshold, loc=mean, scale=scale)
        - p_negative
    )

    # Stack into class probability tensor
    probabilities = np.stack(
        [p_negative, p_uncertain, p_positive],
        axis=1
    )

    # Keep only sufficiently confident predictions
    mask = np.logical_or(
        probabilities[:, 0] > prob_threshold,
        probabilities[:, 2] > prob_threshold
    )

    return mask