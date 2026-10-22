"""
Explanation of File:
File that is used to create a statistical table using a univariate logistic regression and a table similar to the plos1 literature

Author: Muhammad Salaar Arslan
Date: 10/06/2026
Email: arslan.muhammad@hsr.it
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
from sklearn.preprocessing import StandardScaler
from translations import translate, translate_list


def main(args: argparse.Namespace):
    """
    Runs all of the functions and creates the graphs for model evaluation
    """
    data_dir = args.data_dir
    output_dir = args.output_dir
    min_age = args.min_age
    max_age = args.max_age
    patient_wise_threshold = args.patient_wise_threshold
    exam_wise_threshold = args.exam_wise_threshold

    remission_cols = ["IDAA1c_12m", "IDAA1c_12_24m", "IDAA1c_12_36m"]

    with mlflow.start_run():

        for target_name in remission_cols:

            # Load data
            df = load_data(data_dir, target_name=target_name)

            # Drop the same columns as benchmark.py before completeness filtering
            # so patient counts are consistent between the two scripts
            cols_to_drop = [
                "Year of Diagnosis",
                "IDAA1c_min_12m",
                "IDAA1c_min_12_24m",
                "IDAA1c_min_12_36m",
            ]
            existing = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=existing)

            # Apply the same completeness filters as benchmark.py, in the same
            # order (drop_order must match what benchmark.py is run with)
            df = apply_completeness_filter(
                df=df,
                target_name=target_name,
                patient_wise_threshold=patient_wise_threshold,
                exam_wise_threshold=exam_wise_threshold,
                drop_order=args.drop_order,
            )

            statistics_table = run_univariate_logistic(df, target_variable=target_name, min_age=min_age, max_age=max_age)

            # Make the forest plot
            make_forest_plot(statistics_table, output_dir, prefix=target_name)
            make_significant_forest_plot(statistics_table, output_dir, prefix=target_name, p_threshold=args.sig_p_threshold)
            # NOTE: this is a full-cohort, descriptive significance table -- it is
            # NOT used by benchmark.py's model-training pipeline anymore. That
            # pipeline now selects features INSIDE the CV loop (and once on
            # X_train only for the test-set numbers), via SignificantFeatureSelector,
            # so it never uses the test set to decide which features to keep. This
            # file is kept as a separate, whole-cohort exploratory artifact for the
            # paper's descriptive stats -- see benchmark.py's own
            # "{target}_significant_features_train_only.csv" for the feature list
            # actually used in modelling.
            save_significant_features(statistics_table, output_dir, prefix=target_name, p_threshold=args.sig_p_threshold)

            # Save results for this target
            output_path = os.path.join(output_dir, f"{target_name}_logreg_statistics.csv")
            pd.DataFrame(statistics_table).to_csv(output_path, index=False)

            plos_one_statistics = table1_statistics(df, target_name=target_name, min_age=min_age, max_age=max_age)

            output_path = os.path.join(output_dir, f"{target_name}_table1_stats.csv")
            pd.DataFrame(plos_one_statistics).to_csv(output_path, index=False)

            extended_table = table1_extended(
                df,
                target_name=target_name,
                min_age=min_age,
                max_age=max_age,
                logreg_results=statistics_table,
            )

            output_path = os.path.join(output_dir, f"{target_name}_table1_extended.csv")
            extended_table.to_csv(output_path, index=False)
            mlflow.log_artifact(output_path)


def load_data(data_dir: str, target_name: str) -> pd.DataFrame:
    """
    Loading the data from Azure Machine Learning

    Args:
        data_dir (str): input path for the dataset
        target_name (str): name of the target variable

    Returns:
        pd.DataFrame: returns the main dataframe
    """
    path_data = os.path.join(data_dir, f"{target_name}_data.csv")
    df = pd.read_csv(path_data)
    return df


def apply_completeness_filter(
    df: pd.DataFrame,
    target_name: str,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    drop_order: str = "features_first",
) -> pd.DataFrame:
    """Apply the same patient-wise and exam-wise completeness filters used in
    benchmark.py, IN THE SAME ORDER, so both scripts describe/model the same cohort.

    The two orders are NOT equivalent -- whichever axis is filtered first changes
    the denominator the second filter is judged against. Dropping sparse exams
    first ("features_first", the default here and in benchmark.py) means a
    patient isn't penalised for missingness in columns that get dropped anyway,
    which tends to retain more patients than dropping patients first. See
    apply_completeness_filters in benchmark.py's plotting.py for the full
    rationale and a diagnostic comparison plot of both orders.

    This is a duplicate of that same logic (statistics.py and benchmark.py are
    separate AML components with their own code folders, same reason
    translations.py is duplicated rather than imported across them) -- keep the
    two in sync if either changes.

    Args:
        df:                      Raw feature dataframe including the target column.
        target_name:             Name of the target column.
        patient_wise_threshold:  Minimum row completeness fraction to keep a patient.
        exam_wise_threshold:     Minimum column completeness fraction to keep an exam.
        drop_order:              "features_first" (default) or "patients_first" --
                                  must match whatever benchmark.py is run with, or
                                  this script's Table 1 / forest plots will describe
                                  a different cohort than the one actually modelled.

    Returns:
        Filtered dataframe with the same patients and features benchmark.py's
        "full features" run would produce for the same thresholds/order.
    """
    if drop_order not in ("features_first", "patients_first"):
        raise ValueError(
            f"Unknown drop_order: {drop_order!r}. Use 'features_first' or 'patients_first'."
        )

    # Columns exempt from exam-wise filtering
    passthrough_cols = ["Sesso", "Age at Diagnosis", target_name]

    if drop_order == "features_first":
        # Step 1: drop sparse exams, judged over ALL patients
        filterable_columns = df.columns.difference(passthrough_cols)
        exam_completeness = df[filterable_columns].notna().mean(axis=0)
        exams_to_keep = exam_completeness[exam_completeness >= exam_wise_threshold].index
        final_columns = list(exams_to_keep) + [col for col in passthrough_cols if col in df.columns]
        df = df.loc[:, final_columns]

        # Step 2: drop sparse patients, judged over the surviving exams only
        patient_completeness = df.notna().mean(axis=1)
        patients_to_keep = patient_completeness[patient_completeness >= patient_wise_threshold].index
        df = df.loc[patients_to_keep]

    else:  # patients_first
        # Step 1: drop sparse patients, judged over ALL exams
        patient_completeness = df.notna().mean(axis=1)
        patients_to_keep = patient_completeness[patient_completeness >= patient_wise_threshold].index
        df = df.loc[patients_to_keep]

        # Step 2: drop sparse exams, judged over the surviving patients only
        filterable_columns = df.columns.difference(passthrough_cols)
        exam_completeness = df[filterable_columns].notna().mean(axis=0)
        exams_to_keep = exam_completeness[exam_completeness >= exam_wise_threshold].index
        final_columns = list(exams_to_keep) + [col for col in passthrough_cols if col in df.columns]
        df = df.loc[:, final_columns]

    return df


def _convert_to_bool(val) -> bool:
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip().lower()
        if val in ["true", "m"]:
            return True
        elif val in ["false", "f"]:
            return False
    return val


def _convert_objects_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        df[col] = df[col].apply(_convert_to_bool).astype("bool")
    return df


def _preprocess_predictors(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X


def run_univariate_logistic(
    df: pd.DataFrame, target_variable: str, min_age: int, max_age: int
) -> pd.DataFrame:
    """Run univariate logistic regression for each feature against the target.

    Args:
        df (pd.DataFrame): main dataframe
        target_variable (str): target column name
        min_age (int): minimum patient age filter
        max_age (int): maximum patient age filter

    Returns:
        pd.DataFrame: table containing the univariate logistic regression statistics.
    """
    results = []

    # Filter by age
    if min_age is not None:
        df = df[df["Age at Diagnosis"] >= min_age].copy()
    if max_age is not None:
        df = df[df["Age at Diagnosis"] < max_age].copy()

    # Guard against empty dataframe after filtering
    if len(df) == 0:
        logging.warning(
            f"[run_univariate_logistic] No patients remain after age filtering "
            f"(min_age={min_age}, max_age={max_age}). Returning empty results."
        )
        return pd.DataFrame(columns=["Variable", "Odds Ratio", "95% CI Lower", "95% CI Upper", "P-value"])

    # Preprocess features
    X = _convert_objects_to_bool(df)
    X = X.drop(columns=[target_variable])

    X_numeric = X.select_dtypes(include=["int", "float"])
    X_non_numeric = X.drop(columns=X_numeric.columns)

    scaler = StandardScaler()
    X_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(X_numeric),
        columns=X_numeric.columns,
        index=X_numeric.index,
    )

    X_scaled = pd.concat([X_numeric_scaled, X_non_numeric], axis=1)
    X_scaled = _preprocess_predictors(X_scaled)

    # Get target from original dataframe
    y = df[target_variable].astype(int)

    mask = y.notna() & np.isfinite(y)
    y = y[mask]

    # Align X to y
    X = X_scaled.loc[y.index]

    for col in X.columns:
        predictor = X[col]
        temp_df = pd.DataFrame({"const": 1, col: predictor, "y": y})

        temp_df = temp_df.dropna()
        temp_df = temp_df[np.isfinite(temp_df[col])]

        try:
            model = sm.Logit(temp_df["y"], temp_df[["const", col]]).fit(disp=0)
            odds_ratio = np.exp(model.params[col])
            conf = model.conf_int().loc[col]
            ci_lower, ci_upper = np.exp(conf[0]), np.exp(conf[1])
            p_value = model.pvalues[col]

            results.append({
                "Variable": col,
                "Odds Ratio": odds_ratio,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "P-value": p_value,
            })

        except Exception as e:
            logging.error(f"Could not fit model for {col}: {e}")

    return pd.DataFrame(results)


def _get_p_value(
    group1: pd.Series,
    group2: pd.Series,
    is_continuous: bool = True,
    use_ttest: bool = True,
):
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    if is_continuous:
        if use_ttest:
            _, p = ttest_ind(group1_clean, group2_clean, equal_var=False)
        else:
            _, p = mannwhitneyu(group1_clean, group2_clean, alternative="two-sided")
    else:
        combined = pd.Series(list(group1_clean) + list(group2_clean))
        labels = pd.Series([0] * len(group1_clean) + [1] * len(group2_clean))
        contingency_table = pd.crosstab(combined, labels)

        if contingency_table.shape == (2, 2):
            _, p = fisher_exact(contingency_table)
        else:
            _, p, _, _ = chi2_contingency(contingency_table)

    return round(p, 4)


def make_forest_plot(df: pd.DataFrame, output_dir: str, prefix: str):
    """Make a forest plot for the univariate logistic regression.

    Args:
        df (pd.DataFrame): Logistic regression results.
        prefix (str): Prefix file name.
        output_dir (str): Output directory.
    """
    if df.empty:
        logging.warning(f"[make_forest_plot] Empty results for {prefix}, skipping plot.")
        return

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(12, 10))

    sorted_logreg = df.sort_values("Odds Ratio", ascending=True)

    y_pos = np.arange(len(sorted_logreg))
    odds_ratios = sorted_logreg["Odds Ratio"]
    ci_lower = sorted_logreg["95% CI Lower"]
    ci_upper = sorted_logreg["95% CI Upper"]
    p_values = sorted_logreg["P-value"]

    colors = ["red" if p < 0.01 else "orange" if p < 0.05 else "lightblue" for p in p_values]

    ax.scatter(odds_ratios, y_pos, c=colors, s=60, alpha=0.8, zorder=3)
    ax.errorbar(
        odds_ratios,
        y_pos,
        xerr=[odds_ratios - ci_lower, ci_upper - odds_ratios],
        fmt="none",
        color="gray",
        alpha=0.6,
        capsize=3,
    )

    ax.axvline(x=1, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_yticks(y_pos)

    label_colors = ["red" if p < 0.05 else "black" for p in p_values]
    # labels = [var[:30] + "..." if len(var) > 30 else var for var in sorted_logreg["Variable"]]
    labels = [
        translate(var)[:35] + "..." if len(translate(var)) > 35 else translate(var)
        for var in sorted_logreg["Variable"]
    ]

    for i, (label, color) in enumerate(zip(labels, label_colors)):  # noqa: B905
        ax.text(
            -0.02, y_pos[i], label, ha="right", va="center",
            fontsize=8, color=color, transform=ax.get_yaxis_transform(),
        )

    ax.set_yticklabels([])
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
    ax.set_title("Forest Plot: Odds Ratios for Remission Predictors", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        plt.scatter([], [], c="red", s=60, label="p < 0.01"),
        plt.scatter([], [], c="orange", s=60, label="0.01 ≤ p < 0.05"),
        plt.scatter([], [], c="lightblue", s=60, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}_forest.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(path)
    plt.close()

def make_significant_forest_plot(df: pd.DataFrame, output_dir: str, prefix: str, p_threshold: float = 0.05) -> None:
    """Forest plot restricted to statistically significant features only.

    Args:
        df:           Logistic regression results from run_univariate_logistic.
        output_dir:   Directory to save the plot.
        prefix:       Filename prefix (target name).
        p_threshold:  Significance threshold — only features with p < this are shown.
                      Defaults to 0.05.
    """
    significant = df[df["P-value"] < p_threshold].copy()

    if significant.empty:
        logging.warning(
            f"[make_significant_forest_plot] No significant features "
            f"(p < {p_threshold}) found for {prefix}. Skipping plot."
        )
        return

    significant = significant.sort_values("Odds Ratio", ascending=True)

    y_pos = np.arange(len(significant))
    odds_ratios = significant["Odds Ratio"]
    ci_lower    = significant["95% CI Lower"]
    ci_upper    = significant["95% CI Upper"]
    p_values    = significant["P-value"]

    # Two significance tiers within the significant subset
    colors = ["red" if p < 0.01 else "orange" for p in p_values]

    fig, ax = plt.subplots(figsize=(12, max(4, len(significant) * 0.45)))

    ax.scatter(odds_ratios, y_pos, c=colors, s=70, alpha=0.9, zorder=3)
    ax.errorbar(
        odds_ratios,
        y_pos,
        xerr=[odds_ratios - ci_lower, ci_upper - odds_ratios],
        fmt="none",
        color="gray",
        alpha=0.6,
        capsize=4,
    )

    ax.axvline(x=1, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_yticks(y_pos)

    labels = [
        translate(var)[:35] + "..." if len(translate(var)) > 35 else translate(var)
        for var in significant["Variable"]
    ]

    for i, label in enumerate(labels):
        ax.text(
            -0.02, y_pos[i], label,
            ha="right", va="center", fontsize=9,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_yticklabels([])
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
    ax.set_title(
        f"Significant Predictors (p < {p_threshold}) — {prefix}",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [
        plt.scatter([], [], c="red",    s=60, label="p < 0.01"),
        plt.scatter([], [], c="orange", s=60, label="0.01 ≤ p < 0.05"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    path = os.path.join(output_dir, f"{prefix}_significant_forest.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(path)
    plt.close()
    logging.info(f"[significant forest plot] {len(significant)} features plotted → {path}")

def save_significant_features(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str,
    p_threshold: float = 0.05,
) -> None:
    """Save the names of statistically significant features to a CSV.

    This is a full-cohort, descriptive artifact (paired with the forest plots
    and Table 1) -- it is NOT the feature list benchmark.py's model-training
    pipeline uses. That pipeline selects features nested inside its own CV loop
    via SignificantFeatureSelector and writes its own
    "{target}_significant_features_train_only.csv"; the two files are expected
    to differ, since this one is computed on the whole cohort.

    Args:
        df:           Logistic regression results from run_univariate_logistic.
        output_dir:   Directory to save the file.
        prefix:       Filename prefix (target name).
        p_threshold:  Significance threshold. Defaults to 0.05.
    """
    significant = df[df["P-value"] < p_threshold][["Variable", "Odds Ratio", "P-value"]].copy()
    significant = significant.sort_values("P-value", ascending=True).reset_index(drop=True)

    if significant.empty:
        logging.warning(
            f"[save_significant_features] No significant features "
            f"(p < {p_threshold}) for {prefix}. File will be empty."
        )

    path = os.path.join(output_dir, f"{prefix}_significant_features_full_cohort.csv")
    significant.to_csv(path, index=False)
    mlflow.log_artifact(path)
    logging.info(
        f"[save_significant_features] {len(significant)} significant features "
        f"(full-cohort, descriptive only) saved → {path}"
    )

def table1_statistics(
    df: pd.DataFrame,
    target_name: str,
    min_age: int,
    max_age: int,
) -> pd.DataFrame:
    """Create a statistics table in the style of PLoS ONE literature.

    Args:
        df (pd.DataFrame): Feature dataframe including the target column.
        target_name (str): Remission target column name.
        min_age (int): Minimum age filter (inclusive).
        max_age (int): Maximum age filter (exclusive).

    Returns:
        pd.DataFrame: Table with one row per feature.
    """
    # Preferred features — only those that survived completeness filtering are kept
    preferred_features = [
        "Age at Diagnosis",
        "Sesso",
        "num_positive_diabetic_aa",
        "Sg-EMOGLOBINA GLICATA (HbA1c)",
    ]

    df = _convert_objects_to_bool(df)
    if "Sesso" in df.columns:
        df["Sesso"] = df["Sesso"].astype(bool)

    # Filter to only features that survived the completeness filter
    table1_features = [f for f in preferred_features if f in df.columns]
    table1_features_with_target = table1_features + [target_name]
    table1_features_with_target = [f for f in table1_features_with_target if f in df.columns]

    if not table1_features:
        logging.warning(
            f"[table1_statistics] No table1 features remain after completeness "
            f"filtering for {target_name}."
        )
        return pd.DataFrame()

    table1_df = df[table1_features_with_target].copy()

    # Age filter
    if min_age is not None and max_age is not None:
        table1_df = table1_df[
            (table1_df["Age at Diagnosis"] >= min_age) & (table1_df["Age at Diagnosis"] < max_age)
        ].copy()
    elif min_age is not None:
        table1_df = table1_df[table1_df["Age at Diagnosis"] >= min_age].copy()
    elif max_age is not None:
        table1_df = table1_df[table1_df["Age at Diagnosis"] < max_age].copy()

    if len(table1_df) == 0:
        logging.warning(
            f"[table1_statistics] No patients remain after age filtering for {target_name}. "
            f"Returning empty table."
        )
        return pd.DataFrame()

    remitters = table1_df[table1_df[target_name] == True]
    nonremitters = table1_df[table1_df[target_name] == False]
    len_remitters = len(remitters)
    len_nonremitters = len(nonremitters)

    # ── Age ──────────────────────────────────────────────────────────────────
    remitters_age = (
        f"{np.round(remitters['Age at Diagnosis'].mean(), 1)} +- "
        f"{np.round(remitters['Age at Diagnosis'].std(), 1)}"
    )
    nonremitters_age = (
        f"{np.round(nonremitters['Age at Diagnosis'].mean(), 1)} +- "
        f"{np.round(nonremitters['Age at Diagnosis'].std(), 1)}"
    )
    p_age = _get_p_value(remitters["Age at Diagnosis"], nonremitters["Age at Diagnosis"])

    # ── Sesso ─────────────────────────────────────────────────────────────────
    if "Sesso" in remitters.columns:
        males_remitters = int(remitters["Sesso"].sum())
        females_remitters = len_remitters - males_remitters
        males_nonremitters = int(nonremitters["Sesso"].sum())
        females_nonremitters = len_nonremitters - males_nonremitters
        sex_remitters = (
            f"{males_remitters} ({np.round(males_remitters / len_remitters * 100, 1)} %)/"
            f"{females_remitters} ({np.round(females_remitters / len_remitters * 100, 1)} %)"
        )
        sex_nonremitters = (
            f"{males_nonremitters} ({np.round(males_nonremitters / len_nonremitters * 100, 1)} %)/"
            f"{females_nonremitters} ({np.round(females_nonremitters / len_nonremitters * 100, 1)} %)"
        )
        p_sex = _get_p_value(remitters["Sesso"], nonremitters["Sesso"], is_continuous=False)
    else:
        sex_remitters = "N/A"
        sex_nonremitters = "N/A"
        p_sex = np.nan

    # ── Antibodies ────────────────────────────────────────────────────────────
    if "num_positive_diabetic_aa" in remitters.columns:
        two_plus_r = len(remitters[remitters["num_positive_diabetic_aa"] > 2])
        two_plus_nr = len(nonremitters[nonremitters["num_positive_diabetic_aa"] > 2])
        remitters_2_antibodies = (
            f"{two_plus_r}/{len_remitters} "
            f"({np.round(two_plus_r / len_remitters * 100, 1)}%)"
        )
        nonremitters_2_antibodies = (
            f"{two_plus_nr}/{len_nonremitters} "
            f"({np.round(two_plus_nr / len_nonremitters * 100, 1)}%)"
        )
        p_aa = _get_p_value(
            remitters["num_positive_diabetic_aa"],
            nonremitters["num_positive_diabetic_aa"],
            use_ttest=False,
        )
    else:
        remitters_2_antibodies = "N/A"
        nonremitters_2_antibodies = "N/A"
        p_aa = np.nan

    # ── HbA1c ─────────────────────────────────────────────────────────────────
    if "Sg-EMOGLOBINA GLICATA (HbA1c)" in remitters.columns:
        hba1c_value_remitters = (
            f"{np.round(remitters['Sg-EMOGLOBINA GLICATA (HbA1c)'].mean(), 1)} +- "
            f"{np.round(remitters['Sg-EMOGLOBINA GLICATA (HbA1c)'].std(), 1)}"
        )
        hba1c_value_nonremitters = (
            f"{np.round(nonremitters['Sg-EMOGLOBINA GLICATA (HbA1c)'].mean(), 1)} +- "
            f"{np.round(nonremitters['Sg-EMOGLOBINA GLICATA (HbA1c)'].std(), 1)}"
        )
        p_hba1c = _get_p_value(
            remitters["Sg-EMOGLOBINA GLICATA (HbA1c)"],
            nonremitters["Sg-EMOGLOBINA GLICATA (HbA1c)"],
        )
    else:
        hba1c_value_remitters = "N/A"
        hba1c_value_nonremitters = "N/A"
        p_hba1c = np.nan

    # ── Build table dynamically from surviving features ───────────────────────
    # Maps column name → (display label, remitters value, nonremitters value, p-value)
    feature_display = {
        "Age at Diagnosis": (
            "Age (yr)",
            remitters_age,
            nonremitters_age,
            p_age,
        ),
        "Sesso": (
            "Sesso (male/female)",
            sex_remitters,
            sex_nonremitters,
            p_sex,
        ),
        "num_positive_diabetic_aa": (
            "Proportion with >2 antibodies",
            remitters_2_antibodies,
            nonremitters_2_antibodies,
            p_aa,
        ),
        "Sg-EMOGLOBINA GLICATA (HbA1c)": (
            "HbA1c at Diagnosis",
            hba1c_value_remitters,
            hba1c_value_nonremitters,
            p_hba1c,
        ),
    }

    # Only include rows whose underlying column survived the completeness filter
    surviving = [f for f in feature_display if f in table1_df.columns]

    table1_dict = {
        "Parameter": [feature_display[f][0] for f in surviving],
        f"Remitters n={len_remitters}": [feature_display[f][1] for f in surviving],
        f"Non-Remitters n={len_nonremitters}": [feature_display[f][2] for f in surviving],
        "Missing Remitters": [
            f"{remitters[f].isna().sum()} ({round(remitters[f].isna().sum() / len_remitters * 100, 1)}%)"
            if f in remitters.columns else "N/A"
            for f in surviving
        ],
        "Missing Non-Remitters": [
            f"{nonremitters[f].isna().sum()} ({round(nonremitters[f].isna().sum() / len_nonremitters * 100, 1)}%)"
            if f in nonremitters.columns else "N/A"
            for f in surviving
        ],
        "p-values": [feature_display[f][3] for f in surviving],
    }

    return pd.DataFrame(table1_dict)


def table1_extended(
    df: pd.DataFrame,
    target_name: str,
    min_age: int,
    max_age: int,
    logreg_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extended Table 1 in the style of the reference paper.

    For each feature reports:
      - Overall:            mean ± std  (continuous)  or  n (%)  (binary)
      - Non-Remitters:      same
      - Remitters:          same
      - Available Overall:  n/N (%)  — data density visible to reviewer
      - Available Non-Rem:  n/N (%)
      - Available Rem:      n/N (%)
      - OR (95% CI):        from univariate logistic regression
      - p-value:            from univariate logistic regression (or direct test)

    Args:
        df:             Feature dataframe including the target column.
        target_name:    Name of the binary remission target column (0/1 or bool).
        min_age:        Minimum age filter (inclusive).
        max_age:        Maximum age filter (exclusive).
        logreg_results: DataFrame from run_univariate_logistic with columns
                        Variable, Odds Ratio, 95% CI Lower, 95% CI Upper, P-value.

    Returns:
        pd.DataFrame with one row per feature.
    """
    non_feature_cols = {
        target_name, "CPI", "Year of Diagnosis",
        "IDAA1c_min_12m", "IDAA1c_min_12_24m", "IDAA1c_min_12_36m",
    }
    table1_features = [col for col in df.columns if col not in non_feature_cols]

    df = _convert_objects_to_bool(df.copy())
    if "Sesso" in df.columns:
        df["Sesso"] = df["Sesso"].astype(bool)

    # Age filter
    if min_age is not None and max_age is not None:
        df = df[(df["Age at Diagnosis"] >= min_age) & (df["Age at Diagnosis"] < max_age)].copy()
    elif min_age is not None:
        df = df[df["Age at Diagnosis"] >= min_age].copy()
    elif max_age is not None:
        df = df[df["Age at Diagnosis"] < max_age].copy()

    if len(df) == 0:
        logging.warning(
            f"[table1_extended] No patients remain after filtering for {target_name}. "
            f"Returning empty table."
        )
        return pd.DataFrame()

    remitters = df[df[target_name] == True]
    nonremitters = df[df[target_name] == False]
    n_total = len(df)
    n_rem = len(remitters)
    n_nonrem = len(nonremitters)

    # Build OR and p-value lookup from logistic regression results
    or_lookup: dict[str, str] = {}
    p_lookup: dict[str, str] = {}
    if logreg_results is not None and not logreg_results.empty:
        for _, row in logreg_results.iterrows():
            var = row["Variable"]
            or_lookup[var] = (
                f"{row['Odds Ratio']:.2f} "
                f"({row['95% CI Lower']:.2f}–{row['95% CI Upper']:.2f})"
            )
            p_lookup[var] = (
                f"{row['P-value']:.3f}" if row["P-value"] >= 0.001 else "<0.001"
            )

    def _availability(series: pd.Series, n_denom: int) -> str:
        n_avail = int(series.notna().sum())
        pct = round(n_avail / n_denom * 100, 1) if n_denom > 0 else 0.0
        return f"{n_avail}/{n_denom} ({pct}%)"

    def _summarise(series: pd.Series, is_binary: bool) -> str:
        clean = series.dropna()
        if len(clean) == 0:
            return "N/A"
        if is_binary:
            n_true = int(clean.sum())
            pct = round(n_true / len(clean) * 100, 1)
            return f"{n_true} ({pct}%)"
        return f"{round(clean.mean(), 1)} ± {round(clean.std(), 1)}"

    rows = []
    for feat in table1_features:
        if feat not in df.columns:
            continue

        is_binary = feat == "Sesso" or (
            df[feat].dropna().isin([0, 1, True, False]).all()
            and df[feat].nunique() <= 2
        )

        p_value_str = p_lookup.get(feat)
        if p_value_str is None:
            try:
                p = _get_p_value(
                    remitters[feat],
                    nonremitters[feat],
                    is_continuous=not is_binary,
                )
                p_value_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
            except Exception:
                p_value_str = "—"

        rows.append({
            "Variable": feat,
            f"Overall (n={n_total})": _summarise(df[feat], is_binary),
            f"Non-Remitters (n={n_nonrem})": _summarise(nonremitters[feat], is_binary),
            f"Remitters (n={n_rem})": _summarise(remitters[feat], is_binary),
            "Available Overall": _availability(df[feat], n_total),
            "Available Non-Remitters": _availability(nonremitters[feat], n_nonrem),
            "Available Remitters": _availability(remitters[feat], n_rem),
            "OR (95% CI)": or_lookup.get(feat, "—"),
            "p-value": p_value_str,
        })

    result_df = pd.DataFrame(rows)
    result_df["Variable"] = result_df["Variable"].map(translate)
    return result_df


def parse_args() -> argparse.Namespace:
    """
    Set up argument parser

    Returns:
        Namespace: returning the arguments from the command line
    """
    parser = argparse.ArgumentParser(description="Module for benchmarking data against different models")

    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", help="Path to output the results", type=str)
    parser.add_argument("--min_age", type=int, help="Selecting the minimum age for the Plos One Dataset")
    parser.add_argument("--max_age", type=int, help="Selecting the maximum age for the Plos One Dataset")
    parser.add_argument("--patient_wise_threshold", type=float, default=0.0, help="Patient-wise completeness threshold.")
    parser.add_argument("--exam_wise_threshold", type=float, default=0.0, help="Exam-wise completeness threshold.")
    parser.add_argument(
        "--drop_order",
        type=str,
        default="features_first",
        choices=["features_first", "patients_first"],
        help=(
            "Order to apply completeness filters -- must match the value passed "
            "to benchmark.py so both scripts describe/model the same cohort. "
            "'features_first' (default) drops sparse exams before sparse patients."
        ),
    )
    parser.add_argument(
        "--sig_p_threshold",
        type=float,
        default=0.05,
        help="Significance threshold for the full-cohort forest plot / significant-features CSV.",
    )

    args = parser.parse_args()

    return args


def setup_logging():
    """
    Configures logging to output to both the console and a log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("development.log", mode="w"),
        ],
    )


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)