"""
Compares baseline characteristics between patients who have a computable IDAA1c
value and those who don't, on the final 2-18 cohort -- for each of the three
remission windows. This is a missingness/selection-bias check (is IDAA1c
availability related to anything else about the patient?), not an outcome
comparison, so there's no OR/CI -- just group means/proportions and a p-value
per field.

Runs as its own parallel pipeline step, depending only on clean_step and
target_step's outputs (see pipeline.yml's idaa1c_availability_step, alongside
diagnose_attrition_step).
"""

import argparse
import logging
import os

import mlflow
import numpy as np
import pandas as pd
import scipy.stats as stats
from openpyxl.styles import Font
from scipy.stats import fisher_exact

from features import engineer_features
from translations import VARIABLE_TRANSLATION, translate
from utils import clean_and_encode_exams, get_features_for_cohort_at_timepoint

REMISSION_COLS = ["IDAA1c_12m", "IDAA1c_12_24m", "IDAA1c_12_36m"]

# Binary/proportion fields -- used to decide whether to apply proportion
# formatting rather than mean/SD or median/IQR.
CATEGORICAL_FIELD_NAMES = {
    "Sesso",
    "ANTICORPI ANTI IA2",
    "ANTICORPI ANTI INSULINA",
    "ANTICORPI ANTI ZNT8",
    "S-ANTICORPI ANTI GAD",
    "AC. Celiachia",
    "U-ALBUMINA",
    "URINE  - U-Aspetto",
    "URINE  - U-Colore",
    "URINE  - U-Leucociti",
    "URINE  - U-Nitriti",
}


def main(args: argparse.Namespace):
    """Load clean_step/target_step outputs, build the full feature table, and run
    the availability comparison for each remission window."""

    with mlflow.start_run():
        cohort      = pd.read_csv(os.path.join(args.clean_dir, "cohort.csv"))
        esami       = pd.read_csv(os.path.join(args.clean_dir, "esami.csv"))
        insulin     = pd.read_csv(os.path.join(args.clean_dir, "insulin.csv"))
        lab         = pd.read_csv(os.path.join(args.clean_dir, "lab.csv"))
        all_targets = pd.read_csv(os.path.join(args.target_dir, "all_targets.csv"))

        mlflow.log_metric("cohort_n", cohort["CPI"].nunique())
        logging.info(f"Final 2-18 cohort: {cohort['CPI'].nunique()} patients")

        full_features = build_full_feature_table(cohort, esami, insulin, lab)
        mlflow.log_metric("full_feature_table_n_columns", full_features.shape[1])

        for remission_col in REMISSION_COLS:
            if remission_col not in all_targets.columns:
                logging.warning(f"{remission_col} not found in all_targets.csv -- skipping.")
                continue

            run_comparison_for_target(
                remission_col=remission_col,
                full_features=full_features,
                all_targets=all_targets,
                output_dir=args.output_dir,
            )


def build_full_feature_table(
    cohort: pd.DataFrame, esami: pd.DataFrame, insulin: pd.DataFrame, lab: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct the full wide feature table for every patient in the final
    cohort -- mirrors features.py's collect_features() exactly, up to (but not
    including) the per-target loop, so patients without a computable target are
    kept rather than dropped. Reuses utils.py/features.py directly so this can't
    silently drift from what the AML pipeline actually does.
    """
    cohort = cohort[["CPI", "Sesso", "T0", "Age at Diagnosis", "Year of Diagnosis"]].copy()

    lab_clean = clean_and_encode_exams(lab.copy())
    wide = get_features_for_cohort_at_timepoint(
        cohort_dataframe=cohort, target_date_col="T0", patient_index="CPI",
        feature_dataframe=lab_clean, feature_names_to_find=lab_clean["Method_text"].unique(),
        feature_name_col="Method_text", feature_value_col="Numeric_value", feature_date_col="Date",
        offset_days=60, tolerance_days=365, feature_availability_threshold=0.1,
    )

    insulin_as_exam = pd.DataFrame({
        "CPI":   insulin["CPI"],
        "Date":  pd.to_datetime(insulin["Date"]),
        "Exam":  "Insulin",
        "Value": insulin["Insulin (u/24h)"],
        "Measure": "u/24h", "RangeMin": "", "RangeMax": "",
    })
    meteda = pd.concat([esami, insulin_as_exam], ignore_index=True)
    wide = get_features_for_cohort_at_timepoint(
        cohort_dataframe=wide, target_date_col="T0", patient_index="CPI",
        feature_dataframe=meteda, feature_names_to_find=meteda["Exam"].unique(),
        feature_name_col="Exam", feature_value_col="Value", feature_date_col="Date",
        offset_days=60, tolerance_days=365, feature_availability_threshold=0.1,
    )
    return engineer_features(wide)


def run_comparison_for_target(
    remission_col: str,
    full_features: pd.DataFrame,
    all_targets: pd.DataFrame,
    output_dir: str,
):
    """Build the has/doesn't-have group for one remission window, run the
    comparison across ALL feature columns, and save + mlflow-log the result."""

    # Left join -- patients absent from all_targets.csv entirely, or present but
    # with NaN specifically in remission_col, both correctly land as
    # "doesn't have <remission_col>".
    df_analysis = full_features.merge(
        all_targets[["CPI", remission_col]], on="CPI", how="left"
    )
    has_col = f"has_{remission_col}"
    df_analysis[has_col] = df_analysis[remission_col].notna().astype(int)

    n_has     = int(df_analysis[has_col].sum())
    n_not_has = int(len(df_analysis) - n_has)
    mlflow.log_metric(f"{remission_col}_n_has",     n_has)
    mlflow.log_metric(f"{remission_col}_n_not_has", n_not_has)
    logging.info(f"{remission_col}: {n_has} have it, {n_not_has} don't")

    # Exclude bookkeeping / target columns -- everything else is a feature.
    non_feature_cols = {
        "CPI", has_col, remission_col, "T0",
        "Year of Diagnosis",
        "IDAA1c_min_12m", "IDAA1c_min_12_24m", "IDAA1c_min_12_36m",
    }

    table = summarize_group_table(
        df=df_analysis,
        group_col=has_col,
        group_values=[0, 1],
        group_labels=[f"No {remission_col}", f"Has {remission_col}"],
        non_feature_cols=non_feature_cols,
    )

    csv_path = os.path.join(output_dir, f"{remission_col}_availability_comparison.csv")
    table.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    xlsx_path = os.path.join(output_dir, f"{remission_col}_availability_comparison.xlsx")
    export_styled_excel(table, xlsx_path)
    mlflow.log_artifact(xlsx_path)
    logging.info(f"[{remission_col}] availability table saved → {csv_path}")


def summarize_group_table(
    df: pd.DataFrame,
    group_col: str,
    group_values: list,
    non_feature_cols: set,
    group_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Two-group comparison table matching table1_extended's column layout.

    Output columns per row:
        Variable | Overall (n=N) | {label_a} (n=n_a) | {label_b} (n=n_b) |
        Available Overall | Available {label_a} | Available {label_b} | p-value

    No OR/CI -- this is a missingness/selection-bias check, not an outcome model.

    Continuous variables use a Shapiro-Wilk normality gate:
        - Both groups normal → mean ± SD + Welch t-test
        - Otherwise          → median (IQR) + Mann-Whitney U
    Binary variables use Fisher's exact test (2×2) or chi-square (larger tables).

    Args:
        df:               Full merged dataframe (features + group_col).
        group_col:        Column containing the two group indicators.
        group_values:     Exactly two values: [value_a, value_b].
        non_feature_cols: Set of column names to skip (bookkeeping / targets).
        group_labels:     Display labels for value_a and value_b respectively.

    Returns:
        pd.DataFrame with one row per feature, translated variable names.
    """
    if len(group_values) != 2:
        raise ValueError("group_values must contain exactly two values.")

    value_a, value_b = group_values
    label_a, label_b = group_labels if group_labels else (str(value_a), str(value_b))

    df = df[df[group_col].isin([value_a, value_b])].copy()
    n_total = len(df)
    n_a     = int((df[group_col] == value_a).sum())
    n_b     = int((df[group_col] == value_b).sum())

    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _availability(series: pd.Series, n_denom: int) -> str:
        n_avail = int(series.notna().sum())
        pct = round(n_avail / n_denom * 100, 1) if n_denom > 0 else 0.0
        return f"{n_avail}/{n_denom} ({pct}%)"

    def _fmt_continuous(series: pd.Series, use_mean: bool) -> str:
        clean = series.dropna()
        if len(clean) == 0:
            return "N/A"
        if use_mean:
            return f"{clean.mean():.2f} \u00b1 {clean.std():.2f}"
        return (
            f"{clean.median():.2f} "
            f"({clean.quantile(0.25):.2f}\u2013{clean.quantile(0.75):.2f})"
        )

    def _fmt_binary(series: pd.Series) -> str:
        clean = series.dropna()
        if len(clean) == 0:
            return "N/A"
        n_true = int(clean.astype(int).sum())
        return f"{n_true} ({round(n_true / len(clean) * 100, 1)}%)"

    def _fmt_p(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "\u2014"  # em dash
        value = float(value)
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    # ── per-feature rows ──────────────────────────────────────────────────────
    rows = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue

        col_all = df[feat]
        col_a   = df.loc[df[group_col] == value_a, feat]
        col_b   = df.loc[df[group_col] == value_b, feat]

        is_binary = feat in CATEGORICAL_FIELD_NAMES or (
            df[feat].dropna().isin([0, 1, True, False]).all()
            and df[feat].nunique() <= 2
        )

        if is_binary:
            overall_str = _fmt_binary(col_all)
            a_str       = _fmt_binary(col_a)
            b_str       = _fmt_binary(col_b)
            try:
                tab = pd.crosstab(
                    df.loc[df[feat].notna(), feat].astype(int),
                    df.loc[df[feat].notna(), group_col],
                )
                if tab.shape == (2, 2):
                    _, p = fisher_exact(tab)
                else:
                    _, p, _, _ = stats.chi2_contingency(tab)
            except Exception:
                p = np.nan

        else:
            g1 = col_a.dropna().astype(float)
            g2 = col_b.dropna().astype(float)

            # Shapiro-Wilk normality gate (requires ≥ 3 observations per group)
            if len(g1) >= 3 and len(g2) >= 3:
                _, p1 = stats.shapiro(g1)
                _, p2 = stats.shapiro(g2)
            else:
                p1 = p2 = 0.0  # default to non-parametric when too few obs

            use_mean = (p1 > 0.05 and p2 > 0.05)

            overall_str = _fmt_continuous(col_all, use_mean)
            a_str       = _fmt_continuous(col_a,   use_mean)
            b_str       = _fmt_continuous(col_b,   use_mean)

            if len(g1) >= 2 and len(g2) >= 2:
                if use_mean:
                    _, p = stats.ttest_ind(g1, g2)
                else:
                    _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            else:
                p = np.nan

        rows.append({
            "Variable":               feat,
            f"Overall (n={n_total})": overall_str,
            f"{label_a} (n={n_a})":   a_str,
            f"{label_b} (n={n_b})":   b_str,
            "Available Overall":      _availability(col_all, n_total),
            f"Available {label_a}":   _availability(col_a,   n_a),
            f"Available {label_b}":   _availability(col_b,   n_b),
            "p-value":                _fmt_p(p),
        })

    result = pd.DataFrame(rows)
    result["Variable"] = result["Variable"].map(translate)
    return result


def export_styled_excel(table: pd.DataFrame, path: str):
    """Manuscript-style export: bold header row, bold rows where p < 0.05,
    auto-sized columns."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="Table")

        ws = writer.sheets["Table"]
        for col in ws.columns:
            max_length = max(
                (len(str(cell.value)) for cell in col if cell.value is not None),
                default=0,
            )
            ws.column_dimensions[col[0].column_letter].width = min(max(max_length + 2, 14), 30)

        for cell in ws[1]:
            cell.font = Font(bold=True)

        p_col_idx = table.columns.get_loc("p-value")
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            p_cell = row[p_col_idx]
            try:
                raw = (
                    str(p_cell.value)
                    .replace("<", "")
                    .replace(">", "")
                    .replace("\u2014", "")
                    .strip()
                )
                if raw and float(raw) < 0.05:
                    for cell in row:
                        cell.font = Font(bold=True)
            except (ValueError, TypeError):
                pass


def parse_args() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline characteristics by IDAA1c availability."
    )
    parser.add_argument(
        "--clean_dir", type=str,
        help="Path to clean_step's output (cohort/esami/insulin/lab.csv).",
    )
    parser.add_argument(
        "--target_dir", type=str,
        help="Path to target_step's output (all_targets.csv).",
    )
    parser.add_argument("--output_dir", type=str, help="Output path.")
    return parser.parse_args()


def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)