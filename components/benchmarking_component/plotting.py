"""
Plotting functions for benchmarking step.
Author: Edoardo Draetta
Date: 2025/06/12
"""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import missingno as msno
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from translations import translate


def plot_completeness(
    patient_completeness: pd.Series,
    exam_completeness: pd.Series,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    output_dir: Optional[str],  # noqa: UP045
    prefix: Optional[str],  # noqa: UP045
):
    """Create and log feature that defines completeness.

    Args:
        patient_completeness (pd.Series): Series of patient-wise completeness.
        exam_completeness (pd.Series): Series of exam-wise completeness.
        patient_wise_threshold (float): Threshold to draw for patient-wise completeness.
        exam_wise_threshold (float): Threshold to draw for exam-wise completeness.
        output_dir (str, optional): Save path for the image.
        prefix (str, optional): Prefix for the save path.
    """
    # Set up modern plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("Set2")  # Professional, colorblind-friendly palette

    # Create single figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Completeness Analysis", fontsize=20, fontweight="bold", y=0.95)

    # Define consistent colors
    primary_color = "#2E86AB"  # Professional blue
    secondary_color = "#A23B72"  # Complementary magenta
    threshold_color = "#F18F01"  # Warning orange
    accent_color = "#C73E1D"  # Deep red

    # Sort data
    patient_completeness_sorted = patient_completeness.sort_values(ascending=False)
    exam_completeness_sorted = exam_completeness.sort_values(ascending=False)

    # Top left: Patient-wise completeness line plot
    axes[0, 0].plot(patient_completeness_sorted.values, linewidth=2.5, color=primary_color, alpha=0.8)
    axes[0, 0].fill_between(range(len(patient_completeness_sorted)), patient_completeness_sorted.values, alpha=0.3, color=primary_color)
    axes[0, 0].axhline(
        y=patient_wise_threshold * 100,
        color=threshold_color,
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {patient_wise_threshold * 100}%",
    )
    axes[0, 0].set_xlabel("Patient Index (Ranked)", fontsize=12, fontweight="semibold")
    axes[0, 0].set_ylabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[0, 0].set_title("Patient-wise Completeness Profile", fontsize=14, fontweight="bold", pad=20)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # Top right: Patient completeness histogram
    n, bins, patches = axes[0, 1].hist(patient_completeness, bins=40, color=primary_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    # Color bars based on threshold
    for _, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):  # noqa: B905
        if bin_val < patient_wise_threshold * 100:
            patch.set_facecolor(accent_color)
            patch.set_alpha(0.8)

    axes[0, 1].axvline(
        x=patient_wise_threshold * 100,
        color=threshold_color,
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {patient_wise_threshold * 100}%",
    )
    axes[0, 1].set_xlabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[0, 1].set_ylabel("Number of Patients", fontsize=12, fontweight="semibold")
    axes[0, 1].set_title("Patient Completeness Distribution", fontsize=14, fontweight="bold", pad=20)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # Bottom left: Exam-wise completeness line plot
    axes[1, 0].plot(exam_completeness_sorted.values, linewidth=2.5, color=secondary_color, alpha=0.8)
    axes[1, 0].fill_between(range(len(exam_completeness_sorted)), exam_completeness_sorted.values, alpha=0.3, color=secondary_color)
    axes[1, 0].axhline(
        y=exam_wise_threshold * 100, color=threshold_color, linestyle="--", linewidth=2, label=f"Threshold: {exam_wise_threshold * 100}%"
    )
    axes[1, 0].set_xlabel("Lab Exam Index (Ranked)", fontsize=12, fontweight="semibold")
    axes[1, 0].set_ylabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[1, 0].set_title("Exam-wise Completeness Profile", fontsize=14, fontweight="bold", pad=20)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # Bottom right: Exam completeness histogram
    n, bins, patches = axes[1, 1].hist(exam_completeness, bins=40, color=secondary_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    # Color bars based on threshold
    for _, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):  # noqa: B905
        if bin_val < exam_wise_threshold * 100:
            patch.set_facecolor(accent_color)
            patch.set_alpha(0.8)

    axes[1, 1].axvline(
        x=exam_wise_threshold * 100, color=threshold_color, linestyle="--", linewidth=2, label=f"Threshold: {exam_wise_threshold * 100}%"
    )
    axes[1, 1].set_xlabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[1, 1].set_ylabel("Number of Lab Exams", fontsize=12, fontweight="semibold")
    axes[1, 1].set_title("Exam Completeness Distribution", fontsize=14, fontweight="bold", pad=20)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}_completeness.png")
    else:
        path = f"{prefix}_completeness.png"

    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.show()


# ---------------------------------------------------------------------------------
# ORDER-AWARE COMPLETENESS FILTERING
# ---------------------------------------------------------------------------------


def apply_completeness_filters(
    df: pd.DataFrame,
    passthrough_cols: list,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    drop_order: str = "features_first",
) -> dict:
    """Apply patient-wise and exam-wise completeness filters in a given order.

    The two orders are NOT equivalent: whichever axis you filter first changes the
    denominator the second filter is judged against. Dropping sparse exams first
    ("features_first") means a patient is no longer penalised for missingness in
    columns that were going to be dropped anyway, which tends to retain more
    patients. Dropping sparse patients first ("patients_first") is the reverse —
    exam completeness is computed only over the patients that already survived.

    Args:
        df: Input dataframe, including passthrough columns, before any completeness
            filtering.
        passthrough_cols: Columns exempt from exam-wise filtering (e.g. Sesso, Age
            at Diagnosis, the target column).
        patient_wise_threshold: Minimum row completeness fraction to keep a patient.
        exam_wise_threshold: Minimum column completeness fraction to keep an exam.
        drop_order: "features_first" (drop sparse exams, then sparse patients — the
            default) or "patients_first" (drop sparse patients, then sparse exams).

    Returns:
        dict with keys:
            "df": filtered dataframe.
            "patient_completeness": pd.Series actually used to decide the patient
                cut (computed over whatever columns were present at that stage).
            "exam_completeness": pd.Series actually used to decide the exam cut
                (computed over whatever patients were present at that stage).
            "drop_order": the order that was applied.
            "n_patients_start", "n_patients_after_step1", "n_patients_final": int.
            "n_features_start", "n_features_after_step1", "n_features_final": int.
    """
    if drop_order not in ("features_first", "patients_first"):
        raise ValueError(
            f"Unknown drop_order: {drop_order!r}. Use 'features_first' or 'patients_first'."
        )

    n_patients_start = len(df)
    filterable_columns_start = df.columns.difference(passthrough_cols)
    n_features_start = len(filterable_columns_start)

    if drop_order == "features_first":
        # Step 1: drop sparse exams, judged over ALL patients
        exam_completeness = df[filterable_columns_start].notna().mean(axis=0)
        exams_to_keep = exam_completeness[exam_completeness >= exam_wise_threshold].index
        final_columns = list(exams_to_keep) + [c for c in passthrough_cols if c in df.columns]
        df = df.loc[:, final_columns]

        n_patients_after_step1 = len(df)  # patients untouched by this step
        n_features_after_step1 = len(exams_to_keep)

        # Step 2: drop sparse patients, judged over the surviving exams only
        patient_completeness = df.notna().mean(axis=1)
        patients_to_keep = patient_completeness[patient_completeness >= patient_wise_threshold].index
        df = df.loc[patients_to_keep]

        n_patients_final = len(df)
        n_features_final = n_features_after_step1

    else:  # patients_first
        # Step 1: drop sparse patients, judged over ALL exams
        patient_completeness = df.notna().mean(axis=1)
        patients_to_keep = patient_completeness[patient_completeness >= patient_wise_threshold].index
        df = df.loc[patients_to_keep]

        n_patients_after_step1 = len(patients_to_keep)
        n_features_after_step1 = n_features_start  # features untouched by this step

        # Step 2: drop sparse exams, judged over the surviving patients only
        filterable_columns = df.columns.difference(passthrough_cols)
        exam_completeness = df[filterable_columns].notna().mean(axis=0)
        exams_to_keep = exam_completeness[exam_completeness >= exam_wise_threshold].index
        final_columns = list(exams_to_keep) + [c for c in passthrough_cols if c in df.columns]
        df = df.loc[:, final_columns]

        n_features_final = len(exams_to_keep)
        n_patients_final = n_patients_after_step1

    return {
        "df": df,
        "patient_completeness": patient_completeness,
        "exam_completeness": exam_completeness,
        "drop_order": drop_order,
        "n_patients_start": n_patients_start,
        "n_patients_after_step1": n_patients_after_step1,
        "n_patients_final": n_patients_final,
        "n_features_start": n_features_start,
        "n_features_after_step1": n_features_after_step1,
        "n_features_final": n_features_final,
    }


def plot_drop_order_comparison(
    df: pd.DataFrame,
    passthrough_cols: list,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    output_dir: str,
    prefix: str,
) -> dict:
    """Compare patient/feature retention: dropping features first vs. patients first.

    Runs both completeness-filtering orders on the same input dataframe (via
    apply_completeness_filters) and plots a side-by-side grouped bar chart showing
    how many patients and features survive at each stage, for each order. This is
    diagnostic only — it does not affect the dataframe used downstream.

    Args:
        df: Input dataframe, including passthrough columns, before any completeness
            filtering.
        passthrough_cols: Columns exempt from exam-wise filtering.
        patient_wise_threshold: Minimum row completeness fraction to keep a patient.
        exam_wise_threshold: Minimum column completeness fraction to keep an exam.
        output_dir: Directory to save the plot.
        prefix: Filename prefix (target name).

    Returns:
        dict of the two apply_completeness_filters() results, keyed by drop_order —
        handy for logging final patient/feature counts under each order.
    """
    results = {
        order: apply_completeness_filters(
            df=df,
            passthrough_cols=passthrough_cols,
            patient_wise_threshold=patient_wise_threshold,
            exam_wise_threshold=exam_wise_threshold,
            drop_order=order,
        )
        for order in ("features_first", "patients_first")
    }

    order_labels = {
        "features_first": "Features \u2192 Patients",
        "patients_first": "Patients \u2192 Features",
    }
    stage_labels = ["Start", "After step 1", "After step 2 (final)"]
    palette = {"features_first": "#4C72B0", "patients_first": "#DD8452"}

    fig, (ax_patients, ax_features) = plt.subplots(1, 2, figsize=(14, 6))

    width = 0.35
    x = np.arange(len(stage_labels))

    for i, order in enumerate(("features_first", "patients_first")):
        r = results[order]
        patient_counts = [r["n_patients_start"], r["n_patients_after_step1"], r["n_patients_final"]]
        feature_counts = [r["n_features_start"], r["n_features_after_step1"], r["n_features_final"]]
        offset = (i - 0.5) * width
        ax_patients.bar(x + offset, patient_counts, width, label=order_labels[order], color=palette[order], alpha=0.85)
        ax_features.bar(x + offset, feature_counts, width, label=order_labels[order], color=palette[order], alpha=0.85)

        # Annotate bar values
        for xi, val in zip(x + offset, patient_counts):  # noqa: B905
            ax_patients.text(xi, val, str(val), ha="center", va="bottom", fontsize=9)
        for xi, val in zip(x + offset, feature_counts):  # noqa: B905
            ax_features.text(xi, val, str(val), ha="center", va="bottom", fontsize=9)

    for ax, ylabel, title in [
        (ax_patients, "N Patients", "Patients Retained by Stage"),
        (ax_features, "N Features", "Features Retained by Stage"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Drop Order")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Drop-Order Comparison \u2014 {prefix}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(output_dir, f"{prefix}_drop_order_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()

    logging.info(
        f"[drop order comparison] {prefix}: "
        f"features_first final = {results['features_first']['n_patients_final']} patients / "
        f"{results['features_first']['n_features_final']} features | "
        f"patients_first final = {results['patients_first']['n_patients_final']} patients / "
        f"{results['patients_first']['n_features_final']} features \u2192 {save_path}"
    )

    return results


# ---------------------------------------------------------------------------------
# NESTED FEATURE-SELECTION STABILITY
# ---------------------------------------------------------------------------------


def plot_feature_selection_stability(
    selection_counts: dict,
    total_folds: int,
    output_dir: str,
    prefix: str,
    p_threshold: float = 0.05,
) -> None:
    """Bar chart of how often each feature was selected (p < threshold) across CV folds.

    This is the stability diagnostic for the nested significant-feature selector
    (see SignificantFeatureSelector in benchmark.py): selection is re-run on each
    fold's training portion only, so a feature that clears the threshold in most
    or all folds is a much stronger candidate to describe as "significant" than
    one that only clears it occasionally by chance. Every feature that survived
    the completeness filter is shown, including those never selected, so the
    full spectrum of stability is visible in one plot (mirrors make_forest_plot
    in statistics.py, which likewise shows every feature rather than just the
    significant ones).

    Args:
        selection_counts: feature name -> number of folds it was selected in.
        total_folds: Denominator (n_folds * n_repeats) for the proportion.
        output_dir: Directory to save the plot.
        prefix: Filename prefix (target name).
        p_threshold: Significance threshold used for selection (shown in title/axis).
    """
    if not selection_counts or total_folds == 0:
        logging.warning(f"[feature selection stability] No data for {prefix}, skipping plot.")
        return

    df = pd.DataFrame([
        {"Feature": translate(feat), "Count": count, "Proportion": count / total_folds}
        for feat, count in selection_counts.items()
    ]).sort_values("Proportion", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.32)))

    colors = [
        "#2E86AB" if p >= 0.8 else "#4C72B0" if p >= 0.5 else "#B0B0B0"
        for p in df["Proportion"]
    ]
    bars = ax.barh(df["Feature"], df["Proportion"] * 100, color=colors, alpha=0.9)

    for bar, count in zip(bars, df["Count"]):  # noqa: B905
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{count}/{total_folds}", va="center", fontsize=8,
        )

    ax.set_xlim(0, 108)
    ax.set_xlabel(f"Selected in % of CV folds (p < {p_threshold})", fontsize=12, fontweight="semibold")
    ax.set_title(f"Feature Selection Stability \u2014 {prefix}", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{prefix}_feature_selection_stability.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"[feature selection stability] Saved \u2192 {save_path}")


# ---------------------------------------------------------------------------------
# FEATURE COUNTS ACROSS TARGETS
# ---------------------------------------------------------------------------------


def plot_feature_counts_by_target(
    feature_counts: dict,
    output_dir: str,
) -> None:
    """Bar chart of feature counts (full vs significant) per remission target.

    Cross-target summary companion to plot_feature_set_comparison, which shows
    model performance for one target at a time -- this answers "how many
    features is each target actually being modelled with" in one glance across
    all three targets. The same numbers are also logged to mlflow per-target as
    "{target}_n_features_full" / "{target}_n_features_significant" (at the
    parent run level, so they're visible without drilling into a nested run).

    Args:
        feature_counts: target_name -> {"full": int, "sig": int}.
        output_dir: Directory to save the plot.
    """
    if not feature_counts:
        logging.warning("[feature counts by target] No data, skipping plot.")
        return

    label_map = {
        "IDAA1c_12m":    "12m",
        "IDAA1c_12_24m": "12\u201324m",
        "IDAA1c_12_36m": "12\u201336m",
    }
    targets = list(feature_counts.keys())
    target_labels = [label_map.get(t, t) for t in targets]

    full_counts = [feature_counts[t]["full"] for t in targets]
    sig_counts  = [feature_counts[t]["sig"] for t in targets]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_full = ax.bar(x - width / 2, full_counts, width, label="Full", color="#4C72B0", alpha=0.9)
    bars_sig  = ax.bar(x + width / 2, sig_counts,  width, label="Significant", color="#DD8452", alpha=0.9)

    for bars in (bars_full, bars_sig):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                ha="center", va="bottom", fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_xlabel("Remission Target", fontsize=12, fontweight="semibold")
    ax.set_ylabel("N Features", fontsize=12, fontweight="semibold")
    ax.set_title("Feature Count by Target \u2014 Full vs Significant", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Feature Set")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_counts_by_target.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"[feature counts by target] Saved \u2192 {save_path}")