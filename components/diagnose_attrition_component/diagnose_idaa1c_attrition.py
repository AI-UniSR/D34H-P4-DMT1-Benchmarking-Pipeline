"""
diagnose_idaa1c_attrition.py

STANDALONE DIAGNOSTIC -- not part of the AML pipeline, does not modify or
replace clean.py / target.py / features.py / benchmark.py. Read-only against
clean_step's output.

Explains WHY the cohort shrinks from the clean.py cohort (3952 patients) down
to the IDAA1c_12m-computable cohort (978, per features.py's
"IDAA1c_12m_n_patients" metric), by classifying every single patient into ONE
mutually-exclusive reason.

It re-derives target.py's prepare_data() exactly -- same HbA1c extraction
(esami + lab, with mmol/mol -> % conversion), same 36-month cutoff window, same
merge_asof (direction="backward", same tolerance) for weight and insulin dose,
same TDDI/IDAA1c formula -- but keeps the intermediate steps visible instead of
collapsing straight to the final per-target boolean, so we can see exactly
which step drops each patient.

IMPORTANT: IDAA1c needs THREE things to be computable for a given patient in
the 9-16 month window: (1) an HbA1c value in that window, (2) a weight (Peso)
matched within `tolerance` days backward of that HbA1c reading, AND (3) an
insulin dose matched within `tolerance` days backward of that same reading
(TDDI = Insulin / Peso). Missing weight is a real, code-driven failure mode
distinct from missing insulin dose -- both make TDDI (and hence IDAA1c) NaN.

VALIDATION CHECKPOINT: after the age (>=2, <18) and year-of-diagnosis (>=2008)
filters below, the "idaa1c_computable" count should land close to your
features.py "IDAA1c_12m_n_patients" metric (978 in your run). If it doesn't,
that's a sign this script's reconstruction has drifted from the real pipeline
somewhere -- flag it and we'll find the discrepancy, don't trust the breakdown
until it matches.

Usage:
    python diagnose_idaa1c_attrition.py \
        --clean_dir /path/to/clean_step/output \
        --tolerance 10 \
        --min_age 2 --max_age 18 \
        --min_year 2008 \
        --output_dir ./attrition_output \
        --n_patients_after_completeness_filters 716 \
        --n_patients_final 634

The last two arguments are optional -- if given, they're appended as the final
two bars of the combined waterfall chart (patient-wise/exam-wise completeness
filtering and the age-18 upper bound, both applied downstream in
benchmark.py's preprocess()) so the chart tells the whole 3952 -> final story
in one picture, even though this script only *computes* the target-computability
portion itself.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

TARGET_WINDOWS = {
    "IDAA1c_12m":    (9, 16),
    "IDAA1c_12_24m": (9, 25),
    "IDAA1c_12_36m": (9, 37),
}

# Mutually-exclusive reasons, in priority order (first match wins)
REASON_ORDER = [
    "no_hba1c_ever",
    "hba1c_outside_36mo_window",
    "no_hba1c_in_target_window",
    "no_weight_matched",
    "no_insulin_matched",
    "idaa1c_formula_nan_other",
    "idaa1c_computable",
]

REASON_LABELS = {
    "no_hba1c_ever":              "No HbA1c ever recorded",
    "hba1c_outside_36mo_window":  "HbA1c exists, but not within 36mo of diagnosis",
    "no_hba1c_in_target_window":  "HbA1c within 36mo, but not in the 9\u201316mo window",
    "no_weight_matched":          "HbA1c in window, but no weight matched (tolerance)",
    "no_insulin_matched":         "HbA1c + weight in window, but no insulin dose matched",
    "idaa1c_formula_nan_other":   "HbA1c + weight + insulin present, but IDAA1c still NaN (unexpected)",
    "idaa1c_computable":          "IDAA1c_12m computable",
}


def load_clean_data(clean_dir: str) -> dict:
    """Load clean.py's four output CSVs -- identical to target.py's load_data()."""
    filenames = ["cohort.csv", "esami.csv", "insulin.csv", "lab.csv"]
    dataframes = {}
    for f in filenames:
        path = os.path.join(clean_dir, f)
        loaded = pd.read_csv(path)
        # Force a consistent CPI dtype -- an empty dataframe (e.g. no galileo
        # HbA1c rows at all) can otherwise get inferred with a different
        # string dtype than a populated one, which merge_asof rejects later.
        if "CPI" in loaded.columns:
            loaded["CPI"] = loaded["CPI"].astype(str)
        dataframes[f.split(".")[0]] = loaded
        logging.info(f"Loaded {f}: {len(loaded)} rows")
    return dataframes


def extract_hba1c(dataframes: dict) -> pd.DataFrame:
    """Exact replica of the HbA1c extraction (esami + lab, mmol/mol -> %) in
    target.py's prepare_data(), BEFORE any date windowing. Columns: CPI, Date, HbA1c.
    """
    esami_df = dataframes["esami"].copy()
    glic = "Emoglob.Glicata HbA1c"

    esami_mmol = esami_df[(esami_df["Exam"] == glic) & (esami_df["Measure"] == "mmol/mol")].copy()
    esami_mmol["Value_percentage"] = (esami_mmol["Value"].astype(float) * 0.09148) + 2.152
    esami_mmol = esami_mmol.rename(columns={"Value_percentage": "HbA1c"})

    esami_percentage = esami_df[(esami_df["Exam"] == glic) & (esami_df["Measure"] == "%")].copy()
    esami_percentage = esami_percentage.rename(columns={"Value": "HbA1c"})

    esami = pd.concat([esami_percentage, esami_mmol])[["CPI", "Date", "HbA1c"]]

    lab = dataframes["lab"].copy()
    lab_mmol = lab[(lab["Method_text"] == "Sg-EMOGLOBINA GLICATA (HbA1c)") & (lab["Measure"] == "mmol/mol")].copy()
    lab_mmol["Value_percentage"] = (lab_mmol["Value"].astype(float) * 0.09148) + 2.152
    lab_mmol = lab_mmol.rename(columns={"Value_percentage": "HbA1c"})

    lab_percentage = lab[(lab["Method_text"] == "Sg-EMOGLOBINA GLICATA (HbA1c)") & (lab["Measure"] == "%")].copy()
    lab_percentage = lab_percentage.rename(columns={"Value": "HbA1c"})

    lab = pd.concat([lab_percentage, lab_mmol])[["CPI", "Date", "HbA1c"]]

    esami_lab = pd.concat([esami, lab])
    esami_lab["HbA1c"] = pd.to_numeric(esami_lab["HbA1c"]).round(2)
    esami_lab["Date"] = pd.to_datetime(esami_lab["Date"])

    return esami_lab


def extract_weight(dataframes: dict) -> pd.DataFrame:
    """Exact replica of the Peso extraction in target.py's prepare_data(),
    before any date windowing. Columns: CPI, Date, Peso.
    """
    esami_df = dataframes["esami"].copy()
    peso = esami_df[esami_df["Exam"] == "Peso"].copy()
    peso = peso.rename(columns={"Value": "Peso"})[["CPI", "Date", "Peso"]]
    peso["Date"] = pd.to_datetime(peso["Date"])
    return peso


def build_matched_frame(
    dataframes: dict,
    esami_lab_raw: pd.DataFrame,
    peso_raw: pd.DataFrame,
    tolerance: int,
) -> tuple:
    """Reproduce target.py's 36-month cutoff + merge_asof matching exactly,
    but return the intermediate frames too so we can classify per-patient
    failure points instead of only the final collapsed table.

    Returns:
        (df, cohort) where df has one row per HbA1c reading within 36mo of T0,
        with the nearest-prior (within `tolerance` days) Peso and Insulin dose
        merged in, plus TDDI/IDAA1c computed exactly as target.py does, plus a
        MonthsSinceDiagnosis column.
    """
    insulin = dataframes["insulin"].copy()
    insulin = insulin[["CPI", "Date", "Insulin (u/24h)", "Rapid Acting Insulin", "Insulin Pump"]]
    insulin["Date"] = pd.to_datetime(insulin["Date"])

    cohort = dataframes["cohort"].copy()
    cohort["T0"] = pd.to_datetime(cohort["T0"])

    esami_lab = esami_lab_raw.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")
    peso = peso_raw.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")
    insulin = insulin.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")

    esami_lab["T0"] = pd.to_datetime(esami_lab["T0"])
    peso["T0"] = pd.to_datetime(peso["T0"])
    insulin["T0"] = pd.to_datetime(insulin["T0"])

    cutoff = pd.Timedelta(days=30 * 36)
    esami_lab_36mo = esami_lab[(esami_lab["T0"] <= esami_lab["Date"]) & (esami_lab["Date"] <= esami_lab["T0"] + cutoff)]
    peso_36mo = peso[(peso["T0"] <= peso["Date"]) & (peso["Date"] <= peso["T0"] + cutoff)]
    insulin_36mo = insulin[(insulin["T0"] <= insulin["Date"]) & (insulin["Date"] <= insulin["T0"] + cutoff)]

    esami_lab_36mo = esami_lab_36mo.sort_values(["Date", "CPI"])
    peso_36mo = peso_36mo.sort_values(["Date", "CPI"])
    insulin_36mo = insulin_36mo.sort_values(["Date", "CPI"])

    days = pd.Timedelta(days=tolerance) if tolerance is not None else None

    df = pd.merge_asof(
        esami_lab_36mo, peso_36mo, on="Date", by="CPI", direction="backward",
        suffixes=("", "_peso"), tolerance=days,
    )
    df = pd.merge_asof(
        df.sort_values(["Date"]), insulin_36mo, on="Date", by="CPI", direction="backward",
        suffixes=("", "_insulin"), tolerance=days,
    )

    df["TDDI"] = df["Insulin (u/24h)"] / df["Peso"]
    df["IDAA1c"] = df["HbA1c"] + 4 * df["TDDI"]
    df = df.drop(columns=["T0_peso", "T0_insulin"]).sort_values(["CPI", "T0"])
    df["MonthsSinceDiagnosis"] = (df["Date"] - df["T0"]) / pd.Timedelta(days=30)

    return df, cohort, esami_lab_36mo


def classify_patients(
    cohort: pd.DataFrame,
    esami_lab_raw: pd.DataFrame,
    esami_lab_36mo: pd.DataFrame,
    df: pd.DataFrame,
    window: tuple,
) -> pd.DataFrame:
    """Classify every patient in the clean cohort into exactly one reason
    (see REASON_ORDER / REASON_LABELS at the top of this file).
    """
    min_m, max_m = window

    all_cpi = cohort["CPI"].unique()
    has_hba1c_ever      = set(esami_lab_raw["CPI"].unique())
    has_hba1c_in_36mo    = set(esami_lab_36mo["CPI"].unique())

    window_rows = df[(df["MonthsSinceDiagnosis"] >= min_m) & (df["MonthsSinceDiagnosis"] < max_m)]
    has_hba1c_in_window = set(window_rows["CPI"].unique())

    weight_ok = window_rows.dropna(subset=["Peso"])
    has_weight_in_window = set(weight_ok["CPI"].unique())

    insulin_ok = weight_ok.dropna(subset=["Insulin (u/24h)"])
    has_insulin_in_window = set(insulin_ok["CPI"].unique())

    idaa1c_ok = insulin_ok.dropna(subset=["IDAA1c"])
    has_idaa1c_in_window = set(idaa1c_ok["CPI"].unique())

    def reason(cpi) -> str:
        if cpi not in has_hba1c_ever:
            return "no_hba1c_ever"
        if cpi not in has_hba1c_in_36mo:
            return "hba1c_outside_36mo_window"
        if cpi not in has_hba1c_in_window:
            return "no_hba1c_in_target_window"
        if cpi not in has_weight_in_window:
            return "no_weight_matched"
        if cpi not in has_insulin_in_window:
            return "no_insulin_matched"
        if cpi not in has_idaa1c_in_window:
            return "idaa1c_formula_nan_other"
        return "idaa1c_computable"

    out = pd.DataFrame({"CPI": all_cpi})
    out["reason"] = out["CPI"].apply(reason)
    out = out.merge(
        cohort[["CPI", "Age at Diagnosis", "Year of Diagnosis"]], on="CPI", how="left"
    )

    # Non-exclusive supplementary flags, so "weight vs insulin" can be
    # cross-tabbed independently of the priority order above (a patient can
    # be missing BOTH -- the reason column above only reports whichever this
    # script checks first, weight before insulin)
    out["_has_weight_in_window"]  = out["CPI"].isin(has_weight_in_window)
    out["_has_insulin_in_window"] = out["CPI"].isin(
        set(window_rows.dropna(subset=["Insulin (u/24h)"])["CPI"].unique())
    )
    out["_has_hba1c_in_window"] = out["CPI"].isin(has_hba1c_in_window)

    return out


def classify_all_windows(
    cohort: pd.DataFrame,
    esami_lab_raw: pd.DataFrame,
    esami_lab_36mo: pd.DataFrame,
    df: pd.DataFrame,
    windows: dict,
    min_age: int,
) -> tuple:
    """Classify every patient against EACH of the three target windows (12m,
    12-24m, 12-36m), and compute the union: is IDAA1c computable in AT LEAST
    ONE window.

    This is what target.py's compute_target() actually filters on --
    `result[result.IDAA1c_12m.notna() | result.IDAA1c_12_24m.notna() | ...]`
    -- followed by clean_function()'s age>=min_age filter (no upper bound, no
    year filter -- those come later, in benchmark.py / features.py). This is
    DIFFERENT from the single-window (IDAA1c_12m-only) breakdown used
    elsewhere in this script, and is the number that should be compared
    against target.py's "total_patients_all_methods_older_than_2" metric.

    Returns:
        (per_patient_df, n_computable_any_after_age_filter)
    """
    per_patient = pd.DataFrame({"CPI": cohort["CPI"].astype(str).unique()})
    per_patient = per_patient.merge(
        cohort[["CPI", "Age at Diagnosis", "Year of Diagnosis"]], on="CPI", how="left"
    )

    computable_any = pd.Series(False, index=per_patient.index)
    for window_name, window in windows.items():
        classified = classify_patients(cohort, esami_lab_raw, esami_lab_36mo, df, window)
        is_computable = classified.set_index("CPI")["reason"] == "idaa1c_computable"
        col = f"computable_{window_name}"
        per_patient[col] = per_patient["CPI"].map(is_computable).fillna(False)
        computable_any = computable_any | per_patient[col]

    per_patient["computable_any_window"] = computable_any

    # Mirrors target.py's clean_function exactly: age >= min_age, no upper
    # bound, no year filter.
    age_ok = per_patient["Age at Diagnosis"] >= min_age
    n_computable_any_after_age_filter = int((per_patient["computable_any_window"] & age_ok).sum())

    return per_patient, n_computable_any_after_age_filter


def apply_downstream_filters(
    classified: pd.DataFrame,
    min_age: int,
    max_age: int,
    min_year: int,
) -> pd.DataFrame:
    """Add age/year exclusion reasons on top of the target-computability
    reasons, for patients who were "idaa1c_computable" but get excluded by
    features.py's year>=min_year filter or target.py's clean_function's
    age>=min_age filter (both applied to the SAME 1414-scale population in
    the real pipeline). The upper age bound (max_age, i.e. <18) is NOT
    applied until benchmark.py's preprocess() -- much later, after the
    completeness filters -- so it is intentionally left out of this
    reclassification and only shown as a separate, already-known bar in the
    combined waterfall chart.

    Returns the same dataframe with an updated "final_reason" column: still
    "idaa1c_computable" only for patients who clear age>=min_age AND
    year>=min_year on top of having a computable IDAA1c_12m.
    """
    out = classified.copy()
    out["final_reason"] = out["reason"]

    computable_mask = out["reason"] == "idaa1c_computable"

    age_fail = computable_mask & (out["Age at Diagnosis"] < min_age)
    out.loc[age_fail, "final_reason"] = f"age_below_{min_age}"

    year_fail = computable_mask & ~age_fail & (out["Year of Diagnosis"] < min_year)
    out.loc[year_fail, "final_reason"] = f"year_before_{min_year}"

    return out


def summarize(classified: pd.DataFrame, reason_col: str) -> pd.DataFrame:
    """Counts and percentages per reason, ordered for readability."""
    counts = classified[reason_col].value_counts()
    total = len(classified)
    summary = pd.DataFrame({
        "reason": counts.index,
        "n_patients": counts.values,
        "pct_of_total": (counts.values / total * 100).round(1),
    })
    return summary.sort_values("n_patients", ascending=False).reset_index(drop=True)


def plot_reason_breakdown(summary: pd.DataFrame, output_dir: str, title: str, filename: str) -> None:
    """Horizontal bar chart of patient counts per mutually-exclusive reason."""
    df = summary.copy()
    df["label"] = df["reason"].map(REASON_LABELS).fillna(df["reason"])
    df = df.sort_values("n_patients", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    colors = ["#55A868" if r == "idaa1c_computable" else "#C44E52" for r in df["reason"]]
    bars = ax.barh(df["label"], df["n_patients"], color=colors, alpha=0.9)

    for bar, n, pct in zip(bars, df["n_patients"], df["pct_of_total"]):  # noqa: B905
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f"  {n} ({pct}%)", va="center", fontsize=10)

    ax.set_xlabel("N Patients", fontsize=12, fontweight="semibold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"Saved reason breakdown \u2192 {save_path}")


def plot_full_waterfall(
    stages: list,
    output_dir: str,
    filename: str = "idaa1c_12m_full_waterfall.png",
) -> None:
    """Simple attrition waterfall across ALL stages -- from the clean cohort
    down to whatever the last provided stage is.

    Args:
        stages: ordered list of (label, n_patients) tuples.
    """
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(max(10, len(stages) * 1.6), 6))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.9)

    for bar, v in zip(bars, values):  # noqa: B905
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)

    for i in range(1, len(values)):
        dropped = values[i - 1] - values[i]
        if dropped > 0:
            mid_x = (bars[i - 1].get_x() + bars[i].get_x() + bars[i].get_width()) / 2
            ax.annotate(
                f"\u2212{dropped}", xy=(mid_x, max(values[i - 1], values[i]) * 0.5),
                ha="center", fontsize=9, color="#C44E52",
            )

    ax.set_ylabel("N Patients", fontsize=12, fontweight="semibold")
    ax.set_title("Cohort Attrition \u2014 IDAA1c_12m Pipeline", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"Saved full waterfall \u2192 {save_path}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "tolerance": args.tolerance,
            "min_age": args.min_age,
            "max_age": args.max_age,
            "min_year": args.min_year,
        })

        logging.info(f"Loading clean.py output from {args.clean_dir}")
        dataframes = load_clean_data(args.clean_dir)

        n_clean_cohort = dataframes["cohort"]["CPI"].nunique()
        logging.info(f"Clean cohort: {n_clean_cohort} unique CPI")
        mlflow.log_metric("n_clean_cohort", n_clean_cohort)

        esami_lab_raw = extract_hba1c(dataframes)
        peso_raw = extract_weight(dataframes)

        df, cohort, esami_lab_36mo = build_matched_frame(
            dataframes, esami_lab_raw, peso_raw, tolerance=args.tolerance
        )

        window = TARGET_WINDOWS["IDAA1c_12m"]
        classified = classify_patients(cohort, esami_lab_raw, esami_lab_36mo, df, window)

        # ── Reason breakdown BEFORE age/year filters ──────────────────────
        summary_pre = summarize(classified, "reason")
        pre_path = os.path.join(args.output_dir, "idaa1c_12m_reason_breakdown.csv")
        summary_pre.to_csv(pre_path, index=False)
        mlflow.log_artifact(pre_path)
        plot_reason_breakdown(
            summary_pre, args.output_dir,
            title="Why is IDAA1c_12m Not Computable? (clean cohort, before age/year filters)",
            filename="idaa1c_12m_reason_breakdown.png",
        )
        logging.info("\n" + summary_pre.to_string(index=False))

        for _, row in summary_pre.iterrows():
            mlflow.log_metric(f"n_{row['reason']}", int(row["n_patients"]))

        # ── Weight vs insulin cross-tab, for patients WITH HbA1c in window ─
        with_hba1c = classified[classified["_has_hba1c_in_window"]]
        crosstab = pd.crosstab(
            with_hba1c["_has_weight_in_window"].map({True: "Weight matched", False: "Weight NOT matched"}),
            with_hba1c["_has_insulin_in_window"].map({True: "Insulin matched", False: "Insulin NOT matched"}),
        )
        crosstab_path = os.path.join(args.output_dir, "idaa1c_12m_weight_vs_insulin_crosstab.csv")
        crosstab.to_csv(crosstab_path)
        mlflow.log_artifact(crosstab_path)
        logging.info(
            f"\nOf {len(with_hba1c)} patients with HbA1c in the 9-16mo window "
            f"(weight x insulin matching, independent of priority order):\n{crosstab}"
        )

        # ── Age / year filters on top of computability ─────────────────────
        final = apply_downstream_filters(
            classified, min_age=args.min_age, max_age=args.max_age, min_year=args.min_year
        )
        summary_final = summarize(final, "final_reason")
        final_path = os.path.join(args.output_dir, "idaa1c_12m_reason_breakdown_with_age_year.csv")
        summary_final.to_csv(final_path, index=False)
        mlflow.log_artifact(final_path)
        plot_reason_breakdown(
            summary_final, args.output_dir,
            title="Why is IDAA1c_12m Not Computable? (+ age/year filters)",
            filename="idaa1c_12m_reason_breakdown_with_age_year.png",
        )
        logging.info("\n" + summary_final.to_string(index=False))

        # NOTE: most of these repeat summary_pre's numbers unchanged (age/year
        # filters only touch patients who were "idaa1c_computable" going in) --
        # logged under a "final_" prefix so both epochs are visible as
        # separate metric cards, including age_below_X / year_before_Y which
        # summary_pre never had in the first place.
        for _, row in summary_final.iterrows():
            mlflow.log_metric(f"n_final_{row['reason']}", int(row["n_patients"]))

        n_computable_final = int((final["final_reason"] == "idaa1c_computable").sum())
        mlflow.log_metric("n_idaa1c_12m_computable_final", n_computable_final)
        logging.info(
            f"\nVALIDATION CHECKPOINT (features.py): {n_computable_final} patients "
            f"are IDAA1c_12m-computable after age>={args.min_age} and "
            f"year>={args.min_year} filters. Compare this to your features.py "
            f"'IDAA1c_12m_n_patients' metric -- they should be close. If they "
            f"don't match, this reconstruction has drifted from the real "
            f"pipeline somewhere and the breakdown above shouldn't be trusted "
            f"yet."
        )

        # ── Union across all 3 windows + age-only filter, for comparison
        # against target.py's own "total_patients_all_methods_older_than_2"
        # metric -- NOT the same population as n_idaa1c_12m_computable_final
        # above, which is single-window and also has the year filter applied.
        _, n_computable_any_after_age = classify_all_windows(
            cohort, esami_lab_raw, esami_lab_36mo, df,
            windows=TARGET_WINDOWS, min_age=args.min_age,
        )
        mlflow.log_metric("n_computable_any_window_after_age_filter", n_computable_any_after_age)
        logging.info(
            f"\nVALIDATION CHECKPOINT (target.py): {n_computable_any_after_age} "
            f"patients are computable in AT LEAST ONE of the three windows "
            f"(12m / 12-24m / 12-36m) AND age>={args.min_age}, with NO year "
            f"filter (year filtering only starts at features.py, after "
            f"target_step). Compare this to target.py's "
            f"'total_patients_all_methods_older_than_2' metric directly -- "
            f"this is the comparable number, NOT n_idaa1c_12m_computable_final "
            f"above, which is single-window and year-filtered."
        )
        if args.n_patients_after_target_step is not None:
            diff = n_computable_any_after_age - args.n_patients_after_target_step
            mlflow.log_metric("n_patients_after_target_step_reported", args.n_patients_after_target_step)
            mlflow.log_metric("target_step_validation_diff", diff)
            logging.info(
                f"Reported (target.py): {args.n_patients_after_target_step} | "
                f"Computed: {n_computable_any_after_age} | Difference: {diff}"
            )

        # ── Full waterfall, splicing in already-known downstream numbers ──
        stages = [
            ("Clean cohort\n(clean.py)", n_clean_cohort),
            ("IDAA1c_12m\ncomputable", (classified["reason"] == "idaa1c_computable").sum()),
            (f"+ age\u2265{args.min_age}\n+ year\u2265{args.min_year}", n_computable_final),
        ]
        if args.n_patients_after_completeness_filters is not None:
            stages.append(("+ completeness\nfilters", args.n_patients_after_completeness_filters))
        if args.n_patients_final is not None:
            stages.append((f"+ age<{args.max_age}\n(final)", args.n_patients_final))

        plot_full_waterfall(stages, args.output_dir)

        logging.info(f"\nAll outputs saved to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why IDAA1c_12m is not computable for excluded patients."
    )
    parser.add_argument("--clean_dir", type=str, required=True, help="Path to clean_step output (cohort.csv, esami.csv, insulin.csv, lab.csv)")
    parser.add_argument("--tolerance", type=int, default=10, help="merge_asof tolerance in days (must match target.py's --tolerance)")
    parser.add_argument("--min_age", type=int, default=2, help="Minimum age at diagnosis (must match target.py's clean_function)")
    parser.add_argument("--max_age", type=int, default=18, help="Upper age bound, applied later in benchmark.py -- shown as a reference bar only")
    parser.add_argument("--min_year", type=int, default=2008, help="Minimum diagnosis year (must match features.py's min_exam_year)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save CSVs and plots")
    parser.add_argument("--n_patients_after_completeness_filters", type=int, default=None, help="Optional: already-known post-completeness-filter count, appended as a reference bar")
    parser.add_argument("--n_patients_final", type=int, default=None, help="Optional: already-known final count (after benchmark.py's age<max_age filter), appended as a reference bar")
    parser.add_argument("--n_patients_after_target_step", type=int, default=None, help="Optional: target.py's 'total_patients_all_methods_older_than_2' metric, for an explicit computed-vs-reported diff (the computed union-across-windows number is always logged to mlflow regardless)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    main(args)