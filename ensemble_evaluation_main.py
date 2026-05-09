"""
Evaluation pipeline for pulmonary function prediction experiments.

This script:
    1. Loads prediction outputs from multiple trained models/folds
    2. Aggregates ensemble predictions
    3. Computes ROC-based metrics and bootstrap confidence intervals
    4. Generates Bland–Altman analyses
    5. Creates benchmark and performance summary tables
    6. Evaluates probabilistic voting strategies

The script assumes that model outputs are stored in the expected
directory structure generated during training.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataloader.tbs1_link import load_cohort_data

from evaluation.figure_supp_12_bland_altman_plot import plot_bland_altman
from evaluation.invert_gli_norming.main import invert_z_scores
from evaluation.figure_3_heatmap import (
    add_probabilistic_voting,
    generate_figure,
    generate_sens_spec_overview_table,
)
from evaluation.table_3_benchmark import benchmark_runs
from evaluation.table_performance import generate_performance_tables
from evaluation.utils import (
    bootstrap_confidence_intervals_roc_auc,
    combine_fold_estimates,
    combine_fold_votes_aucs,
    compute_roc_curve,
)


def load_runs(path, args, training_stage: str = "best_val_loss"):
    """
    Load predictions and metadata for all configured runs.

    Parameters
    ----------
    path : str
        Root directory containing experiment folders.
    args : argparse.Namespace
        Parsed command-line arguments containing run identifiers.
    training_stage : str
        Which checkpoint outputs to load
        (e.g. 'last' or 'best_val_loss').

    Returns
    -------
    dict
        Dictionary containing loaded predictions, targets,
        uncertainty estimates, and ROC statistics.
    """

    runs = {}

    # Iterate through all configured experiment runs
    for key, run in vars(args).items():

        # Skip non-run parser arguments
        if key in ["no_folds", "n_iter"]:
            continue

        try:
            log_dir = path + run

            # Store fold-wise predictions
            label_predicted_list = []

            # Store uncertainty estimates if available
            std_list = []

            # ---------------------------------------------------------
            # Load outputs for each cross-validation fold
            # ---------------------------------------------------------
            for f in range(args.no_folds):

                version_dir = log_dir + "/version_" + str(f)

                # Ground-truth continuous targets
                label_tgt = torch.load(
                    version_dir + "/labels_test_tgt.pt",
                    map_location="cpu",
                    weights_only=False,
                ).numpy()

                # Binary thresholded targets
                label_norm_tgt = torch.load(
                    version_dir + "/labels_test_tgt_norm_binary.pt",
                    map_location="cpu",
                    weights_only=False,
                ).numpy()

                # Patient IDs
                label_pid = torch.load(
                    os.path.join(path, args.nll_z_pef)
                    + "/version_"
                    + str(f)
                    + "/labels_test_pids.pt",
                    map_location="cpu",
                    weights_only=False,
                )

                # Predicted means
                label_pred = torch.load(
                    version_dir
                    + "/test_estimates_"
                    + training_stage
                    + "_mean.pt",
                    map_location="cpu",
                    weights_only=False,
                ).numpy()

                # Visit identifiers
                label_visit = torch.load(
                    version_dir + "/labels_test_visits.pt",
                    map_location="cpu",
                    weights_only=False,
                )

                # Add fold predictions
                label_predicted_list.append(label_pred[:, :, None])

                # -----------------------------------------------------
                # Load uncertainty estimates if available
                # -----------------------------------------------------
                try:
                    label_std = torch.load(
                        version_dir
                        + "/test_estimates_"
                        + training_stage
                        + "_std.pt",
                        map_location="cpu",
                        weights_only=False,
                    ).numpy()

                    std_list.append(label_std[:, :, None])

                except:
                    # Some models may not provide uncertainty estimates
                    std_list = None

                # -----------------------------------------------------
                # Load GLI predictions for threshold conversion
                # -----------------------------------------------------
                try:
                    gli_pred = torch.load(
                        version_dir + "/labels_test_gli_predicted.pt",
                        map_location="cpu",
                        weights_only=False,
                    ).numpy()

                    # Convert continuous labels into abnormality labels
                    # using 70% of predicted GLI values
                    if key in [
                        "nll_l",
                        "nll_raw",
                        "rmse_l",
                        "rmse_raw",
                        "huber_l",
                        "huber_raw",
                    ]:
                        label_norm_tgt = label_tgt > (gli_pred * 0.7)

                except:
                    gli_pred = None

            # ---------------------------------------------------------
            # Store loaded run information
            # ---------------------------------------------------------
            runs[run] = dict(
                label_pred=label_predicted_list,
                label_tgt=label_norm_tgt,
                label_std=std_list,
                target_cont=label_tgt,
                visit=label_visit,
                pid=label_pid,
                gli_pred=gli_pred,
            )

        except:
            # Skip runs that cannot be loaded
            pass

        # -------------------------------------------------------------
        # Compute weighted ROC statistics
        # -------------------------------------------------------------
        try:

            # FEV1 ROC metrics
            preds, fpr_mean, tpr, auc = compute_roc_curve(
                label_predicted_list,
                std_list,
                label_norm_tgt,
                feature=0,
            )

            runs[run]["FEV1_pred_mean_weighted"] = preds
            runs[run]["FEV1_fpr_mean_var_weighted"] = fpr_mean
            runs[run]["FEV1_tpr_var_weighted"] = tpr
            runs[run]["FEV1_auc_var_weighted"] = auc

            # FVC ROC metrics
            preds, fpr_mean, tpr, auc = compute_roc_curve(
                label_predicted_list,
                std_list,
                label_norm_tgt,
                feature=1,
            )

            runs[run]["FVC_pred_mean_weighted"] = preds
            runs[run]["FVC_fpr_mean_var_weighted"] = fpr_mean
            runs[run]["FVC_tpr_var_weighted"] = tpr
            runs[run]["FVC_auc_var_weighted"] = auc

        except:
            # Continue even if ROC computation fails
            pass

    return runs


def prep_data_frame(data, args):
    """
    Prepare and harmonize the evaluation dataframe.

    This function:
        - Extracts prediction targets
        - Merges cohort metadata
        - Standardizes visit naming
        - Converts date columns
        - Repairs missing follow-up dates where possible

    Parameters
    ----------
    data : dict
        Loaded run data.
    args : argparse.Namespace
        Runtime configuration.

    Returns
    -------
    pandas.DataFrame
        Harmonized dataframe for downstream analysis.
    """

    # -------------------------------------------------------------
    # Extract predictions and metadata
    # -------------------------------------------------------------
    pids = data.get(args.nll_z_pef).get("pid")
    visits = data.get(args.nll_z_pef).get("visit")

    fev1_z = data.get(args.nll_z_pef).get("target_cont")[:, 0]
    fvc_z = data.get(args.nll_z_pef).get("target_cont")[:, 1]

    # Create prediction dataframe
    df_t = pd.DataFrame(
        {
            "person_id_complete": pids,
            "visit_number_unified": visits,
            "fev1_z": fev1_z,
            "fvc_z": fvc_z,
        }
    )

    # Standardize visit naming
    df_t.replace(
        {
            "D0": "Baseline",
            "M6": "Mth 06",
            "M24": "Mth 24",
            "M30": "Mth 30",
            "M48": "Mth 48",
            "M36": "Mth 36",
            "M42": "Mth 42",
        },
        inplace=True,
    )

    # -------------------------------------------------------------
    # Load and filter cohort data
    # -------------------------------------------------------------
    df_tbs = load_cohort_data()

    df_tbs = df_tbs.loc[df_tbs.index.isin(np.unique(pids))]
    df_tbs = df_tbs.copy()

    df_tbs.reset_index(inplace=True)

    # Merge prediction dataframe with cohort metadata
    df_tbs = df_t.merge(
        df_tbs,
        on=["person_id_complete", "visit_number_unified"],
        how="outer",
    )

    # Keep only required columns
    df_tbs = df_tbs.loc[
        :,
        [
            "person_id_complete",
            "visit_number_unified",
            "15_q_6a_cxr_result",
            "15_q_8a_cxr_cavities",
            "15_q_9d1_cxr_d_calcifications_status",
            "15_q_9g1_cxr_g_infiltration_status",
            "15_q_9i1_cxr_i_lobar_volume_loss_colapse_bronchiectasis_status",
            "02_visit_date",
            "02_q_2adate_dem_date_of_birth_numeric",
            "02_q_3a_dem_sex_at_birth",
            "03_q_3a_exm_height",
            "18_visit_date",
            "fev1_z",
            "fvc_z",
            "site_id",
            "hiv_positive",
        ],
    ]

    # -------------------------------------------------------------
    # Convert date columns
    # -------------------------------------------------------------
    df_tbs["02_visit_date"] = pd.to_datetime(
        df_tbs["02_visit_date"],
        format="%Y-%m-%d",
    )

    df_tbs["02_q_2adate_dem_date_of_birth_numeric"] = pd.to_datetime(
        df_tbs["02_q_2adate_dem_date_of_birth_numeric"],
        format="%Y-%m-%d",
    )

    df_tbs["18_visit_date"] = pd.to_datetime(
        df_tbs["18_visit_date"].str.slice(0, 10),
        format="%Y-%m-%d",
    )

    df_tbs.set_index("person_id_complete", inplace=True)

    # -------------------------------------------------------------
    # Repair missing visit dates where possible
    # -------------------------------------------------------------
    mask = (
        df_tbs["18_visit_date"].isna()
        * df_tbs["fev1_z"].notna()
    )

    if mask.sum() > 0:

        df_search = df_tbs.loc[mask, ["visit_number_unified"]]

        # Convert visit labels into approximate time deltas
        df_search.replace(
            {
                "Mth 24": pd.to_timedelta(2 * 365.25, "D"),
                "Mth 30": pd.to_timedelta(2.5 * 365.25, "D"),
                "Mth 36": pd.to_timedelta(3 * 365.25, "D"),
                "Mth 42": pd.to_timedelta(3.5 * 365.25, "D"),
                "Mth 48": pd.to_timedelta(4 * 365.25, "D"),
                "Mth 6": pd.to_timedelta(2 * 30.5, "D"),
            },
            inplace=True,
        )

        df_search.sort_index(inplace=True)

        df_fix = df_tbs.loc[
            df_tbs.index.isin(df_search.index),
            ["02_visit_date"],
        ].dropna()

        df_fix.sort_index(inplace=True)

        # Estimate missing follow-up dates
        dfx = pd.DataFrame(df_fix.values + df_search.values)
        dfx.index = df_fix.index

        df_tbs.loc[mask, "18_visit_date"] = dfx.values

    df_tbs.reset_index(inplace=True)

    return df_tbs.copy()


# =================================================================
# Main evaluation pipeline
# =================================================================
if __name__ == "__main__":

    seed = 3333

    # Root directory containing experiment outputs
    path = os.getcwd() + "/../your/path/here"

    # -------------------------------------------------------------
    # Parse experiment identifiers
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-nll_z_pef", "--nll_z_pef", type=str)
    parser.add_argument("-nll_z", "--nll_z", type=str)
    parser.add_argument("-nll_l", "--nll_l", type=str)
    parser.add_argument("-nll_raw", "--nll_raw", type=str)

    parser.add_argument("-rmse_z_pef", "--rmse_z_pef", type=str)
    parser.add_argument("-rmse_z", "--rmse_z", type=str)
    parser.add_argument("-rmse_l", "--rmse_l", type=str)
    parser.add_argument("-rmse_raw", "--rmse_raw", type=str)

    parser.add_argument("-huber_z_pef", "--huber_z_pef", type=str)
    parser.add_argument("-huber_z", "--huber_z", type=str)
    parser.add_argument("-huber_l", "--huber_l", type=str)
    parser.add_argument("-huber_raw", "--huber_raw", type=str)

    parser.add_argument(
        "-f",
        "--no_folds",
        type=int,
        default=9,
    )

    parser.add_argument(
        "-n",
        "--n_iter",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Evaluate both checkpoint selection strategies
    # -------------------------------------------------------------
    for training_stage in ["last", "best_val_loss"]:

        print(f"Evaluating training stage: {training_stage}")

        # Load predictions and metadata
        data = load_runs(path, args, training_stage)

        run = args.nll_z_pef

        log_dir = (
            path
            + run
            + "/"
            + training_stage
            + "_"
            + str(args.n_iter)
        )

        # Create output directory if necessary
        Path(log_dir).mkdir(
            exist_ok=True,
            parents=True,
        )

        # ---------------------------------------------------------
        # Generate Bland–Altman plots
        # ---------------------------------------------------------
        label_estimated_mean = data.get(run).get("label_pred")
        label_estimated_std = data.get(run).get("label_std")
        label_tgt_cont = data.get(run).get("target_cont")

        plot_bland_altman(
            label_estimated_mean,
            label_tgt_cont,
            predictions_std=label_estimated_std,
            feature_labels=["FEV1", "FVC"],
            save_dir=log_dir,
        )

        # Store summary statistics
        df_bland_altman = pd.DataFrame()
        df = pd.DataFrame()

        # ---------------------------------------------------------
        # Aggregate fold estimates
        # ---------------------------------------------------------
        for k in data.keys():

            mu, mu_w, sig_w = combine_fold_estimates(
                data.get(k).get("label_pred"),
                data.get(k).get("label_std"),
            )

            mean_measurements = (
                mu + data.get(k).get("target_cont")
            ) / 2

            differences = (
                mu - data.get(k).get("target_cont")
            )

            data[k]["estimate_mean"] = mu

            # Use uncertainty-weighted ensemble estimates if available
            if mu_w is not None:

                data[k]["estimate_mean_weighted"] = mu_w
                data[k]["estimate_std_weighted"] = sig_w

                mean_measurements = (
                    mu_w + data.get(k).get("target_cont")
                ) / 2

                differences = (
                    mu_w - data.get(k).get("target_cont")
                )

            # -----------------------------------------------------
            # Compute Bland–Altman summary statistics
            # -----------------------------------------------------
            features = ["FEV1", "FVC"]

            for f, idx in zip(features, range(len(features))):

                mean_diff = np.mean(differences[:, idx])
                std_diff = np.std(differences[:, idx])

                # Assign readable labels for result tables
                if k == args.rmse_raw:
                    row = f + ": CXR raw [liters]"
                    col = "rmse"

                elif k == args.rmse_l:
                    row = f + ": CXR [liters]"
                    col = "rmse"

                elif k == args.rmse_z:
                    row = f + ": CXR [z-score]"
                    col = "rmse"

                elif k == args.rmse_z_pef:
                    row = f + ": CXR & PEF [z-score]"
                    col = "rmse"

                elif k == args.huber_raw:
                    row = f + ": CXR raw [liters]"
                    col = "huber"

                elif k == args.huber_l:
                    row = f + ": CXR [liters]"
                    col = "huber"

                elif k == args.huber_z:
                    row = f + ": CXR [z-score]"
                    col = "huber"

                elif k == args.huber_z_pef:
                    row = f + ": CXR & PEF [z-score]"
                    col = "huber"

                elif k == args.nll_raw:
                    row = f + ": CXR raw [liters]"
                    col = "neg. log-likelihood"

                elif k == args.nll_l:
                    row = f + ": CXR [liters]"
                    col = "neg. log-likelihood"

                elif k == args.nll_z:
                    row = f + ": CXR [z-score]"
                    col = "neg. log-likelihood"

                elif k == args.nll_z_pef:
                    row = f + ": CXR & PEF [z-score]"
                    col = "neg. log-likelihood"

                else:
                    raise Exception()

                df.loc[row, col] = (
                    str(np.round(mean_diff, 3))
                    + " ± "
                    + str(np.round(1.96 * std_diff, 3))
                )

            df.sort_index(inplace=True)

            df.to_csv(log_dir + "/bland_altman_mean_diff.csv")

        # ---------------------------------------------------------
        # Bootstrap confidence intervals
        # ---------------------------------------------------------
        df_sens_specs = pd.DataFrame()

        features = ["FEV1", "FVC"]

        for k in tqdm(
            data.keys(),
            desc="preparing results",
            leave=False,
        ):

            for idx, f in zip(range(len(features)), features):

                # Threshold definition depends on output space
                if k in [
                    args.nll_l,
                    args.rmse_l,
                    args.huber_l,
                    args.nll_raw,
                    args.rmse_raw,
                    args.huber_raw,
                ]:

                    # For liters: threshold = 70% GLI
                    threshold = (
                        0.7 * data.get(k).get("gli_pred")[:, idx]
                    )

                else:
                    # For z-scores: threshold = -2.5
                    threshold = -2.5 * np.ones_like(
                        data.get(k).get("estimate_mean")[:, idx]
                    )

                # -------------------------------------------------
                # Bootstrap ROC metrics
                # -------------------------------------------------
                e = data.get(k).get("estimate_mean")[:, idx]

                b = bootstrap_confidence_intervals_roc_auc(
                    estimates=e,
                    targets=data.get(k).get("label_tgt")[:, idx],
                    threshold=threshold,
                    n_iter=args.n_iter,
                )

                data[k][f + "_bootstrap"] = b

                # Weighted ensemble bootstrap evaluation
                if k in [args.nll_z_pef]:

                    e = data.get(k).get(
                        "estimate_mean_weighted"
                    )[:, idx]

                    b = bootstrap_confidence_intervals_roc_auc(
                        estimates=e,
                        targets=data.get(k).get("label_tgt")[:, idx],
                        threshold=threshold,
                        n_iter=args.n_iter,
                    )

                    data[k][f + "_bootstrap_weighted"] = b

                # -------------------------------------------------
                # Store sensitivity / specificity summaries
                # -------------------------------------------------
                if k in [args.nll_z_pef, args.nll_z]:

                    sens_mean = np.round(
                        data.get(k)
                        .get(f + "_bootstrap")
                        .get("sens_mean"),
                        3,
                    )

                    sens_low = np.round(
                        data.get(k)
                        .get(f + "_bootstrap")
                        .get("sens_low"),
                        3,
                    )

                    sens_high = np.round(
                        data.get(k)
                        .get(f + "_bootstrap")
                        .get("sens_high"),
                        3,
                    )

                    df_sens_specs.loc[
                        k + " " + f,
                        "Sensitivity",
                    ] = (
                        str(sens_mean)
                        + " ["
                        + str(sens_low)
                        + ", "
                        + str(sens_high)
                        + "]"
                    )

                # -------------------------------------------------
                # Combine fold-wise ensemble AUCs
                # -------------------------------------------------
                b = combine_fold_votes_aucs(
                    data.get(k).get("label_pred"),
                    data.get(k).get("label_tgt"),
                    feature=idx,
                )

                data[k][f + "_ensemble_indep_comb"] = b

        # ---------------------------------------------------------
        # Prepare metadata dataframe
        # ---------------------------------------------------------
        df_tbs = prep_data_frame(data, args)

        df_test_proc = invert_z_scores(df_tbs, "other")

        df_tbs = df_tbs.merge(
            df_test_proc,
            how="left",
            left_on=list(df_tbs.columns),
            right_on=list(df_tbs.columns),
        )

        df_tbs.set_index(
            "person_id_complete",
            inplace=True,
        )

        # =========================================================
        # Final evaluation
        # =========================================================

        # ---------------------------------------------------------
        # Probabilistic voting evaluation
        # ---------------------------------------------------------
        for tolerance in tqdm(
            [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
            desc="bootstrapping cutoffs",
        ):

            data = add_probabilistic_voting(
                data,
                args.nll_z_pef,
                tolerance,
                n_iter=args.n_iter,
            )

        generate_sens_spec_overview_table(
            data,
            log_dir,
            args,
        )

        bests_10, bests_20, bests_30 = generate_figure(
            data,
            log_dir,
            args,
        )

        # ---------------------------------------------------------
        # Benchmark evaluation
        # ---------------------------------------------------------
        benchmark_runs(
            data,
            bests_10,
            bests_20,
            bests_30,
            args,
            log_dir,
        )

        print("benchmarked runs")

        # ---------------------------------------------------------
        # Generate final performance tables
        # ---------------------------------------------------------
        generate_performance_tables(
            args.nll_z_pef,
            data.copy(),
            log_dir,
            df_tbs.copy(),
            n_iter=args.n_iter,
        )

        print("generated performance tables")

    print("done.")