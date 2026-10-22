"""
This module cleans meteda, laboratory obsevations, and clinical report observations
for downstream steps.
Author: Edoardo Draetta
Date: 2025-05-23
"""

import argparse
import logging
import os
import warnings

import mlflow
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(args: argparse.Namespace):
    """Load the meteda and galileo data, apply necessary cleaning steps, then save."""

    meteda_path = args.meteda_path
    lab_path = args.lab_path
    output_dir = args.output_dir

    with mlflow.start_run():
        mlflow.log_params({
            "min_age": args.min_age,
            "max_age": args.max_age,
            "min_year": args.min_year,
        })

        # Load data into dataframes dictionary
        raw_dataframes = load_data(meteda_path=meteda_path, lab_path=lab_path)

        # Log unique CPIs for each loaded dataframe
        log_data(raw_dataframes, "RAW")

        # Clean dataframes
        clean_dataframes = clean_data(
            raw_dataframes,
            min_age=args.min_age,
            max_age=args.max_age,
            min_year=args.min_year,
        )

        # Log unique CPIs for each cleaned dataframe
        log_data(clean_dataframes, "CLEAN")

        # Save cleaned dataframes to output directory
        save_data(clean_dataframes, output_dir)


def load_data(meteda_path: str, lab_path: str) -> dict[str, pd.DataFrame]:
    """Loads multiple datasets from different sources and returns them as a dictionary of pandas DataFrames.

    Args:
    - meteda_path (str): The directory path where Meteda CSV files are located.
    - lab_path (str): The file path to the Galileo laboratory data in Parquet format.

    Returns:
    - dict(str, pd.DataFrame): A dictionary where keys are dataset names and values are the corresponding pandas
    DataFrames.
    """

    # 1. Load meteda files
    meteda_filenames = [
        "Diagnosi.csv",
        "Anagrafica.csv",
        "Esami.csv",
        "Terapia.csv",
        "AntiDiabetica.csv",
        "TerapiaMicroinfusori.csv",
    ]

    dataframes = {file.split(".")[0]: pd.read_csv(os.path.join(meteda_path, file)) for file in meteda_filenames}

    # 2. Load galileo laboratory data
    dataframes["lab"] = pd.read_parquet(lab_path)

    return dataframes


def clean_data(
    dataframes: dict[str, pd.DataFrame],
    min_age: int = 2,
    max_age: int = 18,
    min_year: int = 2008,
) -> dict[str, pd.DataFrame]:
    """Cleans fundamental dataframes for downstream analysis: cohort, exams, insulin, lab.

    The cohort is built via five ordered filters, each logging a patient count
    to mlflow so the attrition at each step is auditable:
        1. Has a known T1DM diagnosis -- i.e., a diagnosis date on record at
           all. Diagnosi.csv is itself a T1DM registry (the merge below
           already gates on presence there), so "known" here specifically
           means T0 isn't blank. This is distinct from step 4: a genuinely
           missing date (this step) is a different data-quality issue than a
           present-but-placeholder date (step 4).
        2. Diagnosed on/after min_year (lab data is missing before this year)
        3. Deduplicated by CPI
        4. Valid (non-placeholder) diagnosis date and date of birth -- meteda
           uses "1900-01-01" as a placeholder for "unknown"
        5. Age at diagnosis within [min_age, max_age)

    This is the single place the full age range is enforced -- target.py,
    features.py, and benchmark.py each carry a defensive re-check of this same
    range further downstream (see their docstrings), but none of them should
    ever actually remove a patient once this function is correct.

    Args:
        dataframes (dict[str, pd.DataFrame]): Dataframes to be cleaned.
        min_age (int): Minimum age at diagnosis to keep (inclusive).
        max_age (int): Maximum age at diagnosis to keep (exclusive).
        min_year (int): Minimum diagnosis year to keep (inclusive).

    Returns:
        dict[str, pd.DataFrame]: Cleaned dataframes.
    """

    cleaned_dataframes = {}

    # -- METEDA COHORT --
    diagnosi = dataframes["Diagnosi"].copy()
    anagrafica = dataframes["Anagrafica"].copy()

    cohort = pd.merge(diagnosi, anagrafica, on="CPI", how="inner")
    mlflow.log_metric("cohort_n_after_merge", cohort["CPI"].nunique())

    # Convert dates once, early, so every subsequent filter below operates on
    # real datetimes rather than raw strings. errors="coerce" turns any blank
    # or unparseable value into NaT instead of raising -- that failure mode is
    # exactly what step 1 below is designed to catch and count, rather than
    # having a single bad row crash the whole clean_step with no visibility
    # into how many patients were affected.
    cohort["T0"] = pd.to_datetime(cohort["DataDiagnosi"], errors="coerce")
    cohort["DDN"] = pd.to_datetime(cohort["DDN"], dayfirst=True, errors="coerce")
    cohort.drop(columns=["DataDiagnosi"], inplace=True)

    # ── Step 1: keep only patients with a known T1DM diagnosis ────────────
    # "Known" = has a diagnosis date on record at all (not blank/unparseable).
    cohort = cohort[cohort["T0"].notna()]
    mlflow.log_metric("cohort_n_after_diagnosis_date_filter", cohort["CPI"].nunique())

    # ── Step 2: drop patients diagnosed before min_year ────────────────────
    # (Galileo lab data is missing before 2008/2009, so anything earlier has
    # no usable lab features regardless of everything else.)
    cohort = cohort[cohort["T0"].dt.year >= min_year]
    mlflow.log_metric("cohort_n_after_year_filter", cohort["CPI"].nunique())

    # ── Step 3: drop duplicated patient records ────────────────────────────
    cohort.drop_duplicates(subset="CPI", inplace=True)
    mlflow.log_metric("cohort_n_after_dedup", cohort["CPI"].nunique())

    # ── Step 4: drop invalid (placeholder) timestamps ──────────────────────
    # meteda uses 1900-01-01 as a placeholder for "unknown" on both the
    # diagnosis date and date of birth -- both feed into Age at Diagnosis
    # below, so both need to be valid, not just T0 (the original check here
    # only validated T0; DDN was unchecked, which could silently produce a
    # nonsensical Age at Diagnosis rather than an explicit exclusion). DDN's
    # own missing/blank values (distinct from the 1900-01-01 placeholder) are
    # also caught here, via notna() -- T0 already can't be missing by this
    # point, since step 1 above already required it.
    cohort = cohort[
        (cohort["T0"] != pd.Timestamp("1900-01-01"))
        & cohort["DDN"].notna()
        & (cohort["DDN"] != pd.Timestamp("1900-01-01"))
    ]
    mlflow.log_metric("cohort_n_after_timestamp_filter", cohort["CPI"].nunique())

    # Derived fields, needed for step 5
    cohort["Age at Diagnosis"] = round((cohort["T0"] - cohort["DDN"]).dt.days / 365.25, 2)
    cohort["Year of Diagnosis"] = cohort["T0"].dt.year

    # ── Step 5: drop patients outside the study age range ──────────────────
    cohort = cohort[(cohort["Age at Diagnosis"] >= min_age) & (cohort["Age at Diagnosis"] < max_age)]
    mlflow.log_metric("cohort_n_after_age_filter", cohort["CPI"].nunique())

    # store the result
    cleaned_dataframes["cohort"] = cohort[["CPI", "T0", "Sesso", "Age at Diagnosis", "Year of Diagnosis"]]

    # -- METEDA EXAMS --

    cleaned_exams = dataframes["Esami"].copy()

    rename_map = {"Data": "Date", "Valore": "Value", "Descrizione": "Exam", "UM": "Measure"}

    # Rename
    cleaned_exams = cleaned_exams.rename(columns=rename_map)

    # Clean value column
    cleaned_exams["Value"] = _replace_comma_with_period(cleaned_exams["Value"])
    cleaned_exams.dropna(subset=["Value"], inplace=True)

    # Drop null dates
    cleaned_exams = cleaned_exams[cleaned_exams["Date"] != "1900-01-01"]

    # Uppercase CPI
    cleaned_exams["CPI"] = cleaned_exams["CPI"].str.upper()

    # Clean exam names
    cleaned_exams["Exam"] = cleaned_exams["Exam"].str.strip()

    # Select columns
    cleaned_exams = cleaned_exams[["CPI", "Date", "Exam", "Value", "Measure", "RangeMin", "RangeMax"]]

    # Store the results
    cleaned_dataframes["esami"] = cleaned_exams

    # -- METEDA INSULIN --
    terapia = dataframes["Terapia"].copy()
    infusori = dataframes["TerapiaMicroinfusori"].copy()
    antidiabetica = dataframes["AntiDiabetica"].copy()

    # Converting the commas into period for the antidiabetica dataframe
    antidiabetica["Unita"] = _replace_comma_with_period(antidiabetica["Unita"])

    # Converting to datetime
    terapia["Data"] = pd.to_datetime(terapia["Data"])

    # Drop null dates
    terapia = terapia[terapia["Data"] != "1900-01-01"]

    # Creating the Total Daily Insulin received from the Insulin Pump Data
    infusori["OraA"] = infusori["OraA"].replace(0, 60 * 24)
    infusori["Hours"] = (infusori["OraA"] - infusori["OraDa"]) / 60
    infusori["Unita"] = infusori["Valore"] * infusori["Hours"]

    # Summing the Insulin Pump data by CPI, DataInput, and IDPrescr and Renaming the columns
    infusori["DataInput"] = pd.to_datetime(infusori["DataInput"])
    infusori = infusori[infusori["DataInput"] != "1900-01-01"]  # Drop null dates

    infusori = infusori.groupby([
        "CPI",
        "DataInput",
    ])["Unita"].sum().reset_index()

    infusori = infusori.rename(
        columns={
            "DataInput": "Data",
            "Unita": "Unita_infusori"
        }
    )

    # Join the two dataframes
    terap_antidiab = pd.merge(
        terapia,
        antidiabetica[antidiabetica['PrincipioAttivo'].str.contains('insulin', case=False, na=False)],
        on=["CPI", "IdTerapia"],
        how='left'
    )

    # Filter insulin rows and aggregate first to avoid duplication
    terap_antidiab_agg = (
        terap_antidiab
        .groupby(['CPI', 'Data'])['Unita']
        .sum()
        .reset_index()
        .rename(columns={'Unita': 'Unita_antidiab'})
    )

    insulin = pd.merge(
        terap_antidiab_agg,
        infusori,
        how='left',
        on=["CPI", "Data"]
    )

    # Fill missing antidiab values with 0 (in case some therapies had only pumps)
    insulin['Unita_infusori'] = insulin['Unita_infusori'].fillna(0)

    # Clean units
    insulin["Unita_antidiab"] = pd.to_numeric(insulin["Unita_antidiab"])

    # Incorporating insulin pump data into the total daily insulin
    insulin["Unita_total"] = insulin["Unita_infusori"] + insulin["Unita_antidiab"]

    # Rename columns
    rename_map = {
        "Unita_total": "Insulin (u/24h)",
        "Data": "Date",
        "Unita_antidiab": "Rapid Acting Insulin",
        "Unita_infusori": "Insulin Pump"}

    daily_insulin = insulin[[
        "CPI",
        "Data",
        "Unita_total",
        "Unita_antidiab",
        "Unita_infusori"]].rename(columns=rename_map)

    # Normalizing the Date Column
    daily_insulin["Date"] = daily_insulin["Date"].dt.normalize()

    # Store the results
    cleaned_dataframes["insulin"] = daily_insulin

    # -- GALILEO LAB OBSERVATIONS --
    lab = dataframes["lab"].copy()

    # Drop by string value and by exam name
    string_values_to_clean = [
        "Annullato da SIO",
        "Annullato da sio",
        ".",
        "---",
        "Esame eseguito presso il D.H. di Pediatria",
        "Esame eseguito presso il D.H. di Pediatria ",
        "Vedi allegato",
        "Vedi allegato.",
        "Contenitore giunto aperto",
        "Valore Non Calcolabile",
        "Non Valutabile",
        "il campione non coaugula",
        "RIP",
        "",
    ]

    exams_to_drop = [
        "URINE  - U-Bilirubina",
        "S-Anti HCV  anticorpi totali",
        "Ricerca Anticorpi Irregolari",
        "S-ANTICORPI ed ANTIGENE p24 HIV 1-2",
        "S-Anti Treponema pallidum anticorpi totaliBilirubina Totale (nBili)",
        "Prelievo Sangue Capillare",
        "Note Materiale Selezionato EGA",
        "SEDIMENTO - Esame microscopico:",
        "Valutazione morfologica del tracciato",
        "CORONA VIRUS - materiale esaminato",
        "Titolo",
        "COROAG - materiale esaminato",
        "COROAG - esito",
        "CORONA VIRUS - esito",
        "Numero riferimento",
    ]

    lab = lab[~lab["String_value"].isin(string_values_to_clean)]
    lab = lab[~lab["Method_text"].isin(exams_to_drop)]

    # Delocalize date
    lab["Date"] = pd.to_datetime(lab["Date"])
    lab["Date"] = lab["Date"].dt.tz_localize(None)

    # Rename patient index
    lab.rename(columns={"Subject_identifier": "CPI"}, inplace=True)
    lab["CPI"] = lab["CPI"].str.upper()

    # Store results
    cleaned_dataframes["lab"] = lab

    # -- GALILEO DOCUMENT OBSERVATIONS --
    # TODO Implement once documents are updated

    # Return dict of dataframes
    return cleaned_dataframes


def _replace_comma_with_period(series: pd.Series) -> pd.Series:
    """
    Replace comma delimiters with periods.
    """
    return series.astype(str).str.replace(",", ".").astype(float)


def save_data(dataframes: dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save each dataframe in a dictionary to file in a provided output directory.
    Args:
        dataframes (dict[str, pd.DataFrame]): Dictionary of dataframes to save.
        output_dir (str): Output path.
    """

    for filename, df in dataframes.items():
        path = os.path.join(output_dir, filename + ".csv")
        df.to_csv(path, index=False)
        logging.info(f"Saved {filename} in output directory, {output_dir}")
        logging.debug(f"\n{df.sample(4)}")


def log_data(dataframes: dict[str, pd.DataFrame], prefix: str = None):
    """Log metedata for each dataframe using mlflow.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of dataframes to log.
    """
    for key, df in dataframes.items():
        if "CPI" in df.columns:
            prefix = prefix or ""
            mlflow.log_metric(f"{prefix} {key} unique CPI", df["CPI"].nunique())


def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    """Parse input arguments. Add arguments here."""
    parser = argparse.ArgumentParser(description="Module for cleaning raw data")
    parser.add_argument("--meteda_path", type=str, help="Path to raw meteda data")
    parser.add_argument("--lab_path", type=str, help="Path to raw galileo lab data")
    parser.add_argument("--output_dir", type=str, help="Path to cleaned data")
    parser.add_argument("--min_age", type=int, default=2, help="Minimum age at diagnosis to keep (inclusive).")
    parser.add_argument("--max_age", type=int, default=18, help="Maximum age at diagnosis to keep (exclusive).")
    parser.add_argument(
        "--min_year", type=int, default=2008,
        help="Minimum diagnosis year to keep (inclusive) -- lab data is missing before this year.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Set up argument parsing and logging
    args = parse_args()
    setup_logging()
    main(args)