"""
Collect features for provided cohort.
Author: Edoardo Draetta
Date: 2025/05/28
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
from utils import clean_and_encode_exams, get_features_for_cohort_at_timepoint

ROOT_PATH = Path(__file__).parent


def main(args: argparse.Namespace):
    """Collect features from laboratory data and ouput the created file."""

    data_dir = args.data_dir
    target_dir = args.target_dir
    output_dir = args.output_dir

    remission_cols = ["IDAA1c_12m", "IDAA1c_12_24m", "IDAA1c_12_36m"]
    continuous_cols = ["IDAA1c_min_12m", "IDAA1c_min_12_24m", "IDAA1c_min_12_36m"]


    with mlflow.start_run():
        mlflow.log_param("min_year", args.min_year)

        dataframes = load_data(data_dir=data_dir, target_dir=target_dir)

        logging.info("Collecting features...")
        dataframes = collect_features(
            dataframes,
            remission_cols=remission_cols,
            continuous_cols=continuous_cols,
            min_year=args.min_year,
        )

        logging.info("Engineering features...")
        engineered_dataframes = {}
        for remission_col in remission_cols:
            engineered_dataframes[remission_col] = engineer_features(dataframes[remission_col])

        logging.info("Saving and logging data...")
        for remission_col in remission_cols:

            df = engineered_dataframes[remission_col]
            mlflow.log_metrics({f"{remission_col}_n_patients": df.shape[0], f"{remission_col}_n_features": df.shape[1]})

            save_data(df, output_dir, prefix=remission_col)


def load_data(data_dir: str, target_dir: str) -> dict[str, pd.DataFrame]:
    """Load cleaned data and target data"""

    dataframes = {}

    dataframes["cohort"] = pd.read_csv(Path(ROOT_PATH) / data_dir / "cohort.csv")
    dataframes["esami"] = pd.read_csv(Path(ROOT_PATH) / data_dir / "esami.csv")
    dataframes["insulin"] = pd.read_csv(Path(ROOT_PATH) / data_dir / "insulin.csv")
    dataframes["lab"] = pd.read_csv(Path(ROOT_PATH) / data_dir / "lab.csv")

    # Change to all_targets
    dataframes["target"] = pd.read_csv(Path(ROOT_PATH) / target_dir / "all_targets.csv")

    return dataframes


def collect_features(
    dataframes: dict[str, pd.DataFrame],
    remission_cols: list[str],
    continuous_cols: list[float],
    min_year: int = 2008,
) -> dict[str, pd.DataFrame]:
    """Collect features for the provided cohort.

    Args:
        dataframes (dict[str, pd.DataFrame]): Dictionary of dataframes to use.
        remission_cols (list[str]): List of remission columns.
        continuous_cols (list[float]): List of continuous target columns.
        min_year (int): Minimum diagnosis year to keep (inclusive) -- defensive
            re-check only; clean.py already enforces this upstream (see below).

    Returns:
        dict[str, pd.DataFrame]: Dictionary of dataframes.
    """

    # Get cohort dataframe
    cohort = dataframes["cohort"]
    cohort = cohort[["CPI", "Sesso", "T0", "Age at Diagnosis", "Year of Diagnosis"]]

    # Get target dataframe
    target = dataframes["target"]
    target = target[["CPI"] + remission_cols + continuous_cols]

    # Select cohort with defined target
    cohort = cohort[cohort["CPI"].isin(target["CPI"])]

    # -- COLLECT GALILEO LABORATORY FEATURES --

    # Get lab data
    lab = dataframes["lab"]

    # Defensive re-check: clean.py already drops patients diagnosed before
    # min_year upstream, so this should be a no-op. Logs a warning rather than
    # silently filtering if it isn't (Lab data is missing before 2008/2009,
    # so anything that slips through here would have no usable lab features).
    year_ok = pd.to_datetime(cohort["T0"]).dt.year >= min_year
    n_out_of_range = (~year_ok).sum()
    if n_out_of_range > 0:
        logging.warning(
            f"[collect_features] {n_out_of_range} patient(s) diagnosed before "
            f"{min_year} reached features.py -- clean.py's upstream year filter "
            f"should have already removed these."
        )
    cohort = cohort[year_ok]

    # Clean exams
    lab = clean_and_encode_exams(lab)

    # Build wide feature table (one row per patient, one column per exam at provided timepoint)
    wide_feature_table = get_features_for_cohort_at_timepoint(
        cohort_dataframe=cohort,
        target_date_col="T0",
        patient_index="CPI",
        feature_dataframe=lab,
        feature_names_to_find=lab["Method_text"].unique(),
        feature_name_col="Method_text",
        feature_value_col="Numeric_value",
        feature_date_col="Date",
        offset_days=60,
        tolerance_days=365,
        feature_availability_threshold=0.1,
    )

    # -- COLLECT METEDA FEATURES --

    # Get exam data
    esami = dataframes["esami"]

    # Get insulin data
    insulin = dataframes["insulin"]

    # Add insulin to exams

    # Transform insulin data to match esami format
    insulin = pd.DataFrame({
        'CPI': insulin['CPI'],
        'Date': pd.to_datetime(insulin['Date']),
        'Exam': 'Insulin',
        'Value': insulin['Insulin (u/24h)'],  # Use the insulin values
        'Measure': 'u/24h',
        'RangeMin': '',
        'RangeMax': ''
    })

    # Concatenate the dataframes
    meteda = pd.concat([esami, insulin], ignore_index=True)

    wide_feature_table = get_features_for_cohort_at_timepoint(
        cohort_dataframe=wide_feature_table,
        target_date_col="T0",
        patient_index="CPI",
        feature_dataframe=meteda,
        feature_names_to_find=meteda["Exam"].unique(),
        feature_name_col="Exam",
        feature_value_col="Value",
        feature_date_col="Date",
        offset_days=60,
        tolerance_days=365,
        feature_availability_threshold=0.1,
    )

    # SAVE RESULTS

    # Drop T0
    # wide_feature_table.drop(columns="T0", inplace=True)

    dataframes = {}
    for remission_col, continuous_col in zip(remission_cols, continuous_cols):

        # Attach Targets
        selection = wide_feature_table.copy()
        cols_to_merge = ["CPI", remission_col, continuous_col]
        selection = pd.merge(selection, target[cols_to_merge], on="CPI", how="left")
        selection.dropna(subset=remission_col, inplace=True)
        dataframes[remission_col] = selection

    return dataframes


def engineer_features(feature_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the dataset.

    Args:
        feature_dataframe (pd.DataFrame): Input feature dataframe.

    Returns:
        pd.DataFrame: Dataset with engineered features added.
    """
    df = feature_dataframe.copy()

    # Handle Diabetes features
    diabete_cols = [
        "ANTICORPI ANTI IA2",
        "ANTICORPI ANTI INSULINA",
        "ANTICORPI ANTI ZNT8",
        "S-ANTICORPI ANTI GAD",
        # "ANTICORPI VS. ISOLA PANCREATICA-ICA",
    ]
    diabete_cols = [col for col in diabete_cols if col in df.columns]

    # Number of positive antibody exams & Number of missing antibody exams:
    df["num_positive_diabetic_aa"] = (df[diabete_cols] == 1).sum(axis=1)
    df["num_missing_diabetic_aa"] = df[diabete_cols].isnull().sum(axis=1)

    # Handle Celiacs features
    celiac_cols = [
        "AC. ANTI TRANSGLUTAMINASI (IgG)",
        "AC. ANTI TRANSGLUTAMINASI (IgA)",
        "AC. IgA ANTI ENDOMISIO",
        "AC. vs.TRANSGLUTAMINASI (IgA) reflex",
    ]

    # Any positive exams
    df["AC. Celiachia"] = (df[celiac_cols] == 1).any(axis=1)

    # Encode sex
    df["Sesso"] = df["Sesso"].map({"M": 1, "F": 0}).astype(float)

    return df


def save_data(df: pd.DataFrame, output_dir: str, prefix: str):
    """Save data to output directory."""
    completeness = df.notna().mean().sort_values()
    path = os.path.join(output_dir, f"{prefix}_completeness.csv")
    completeness.to_csv(path)

    path = os.path.join(output_dir, f"{prefix}_data.csv")
    df.to_csv(path, index=False)


def parse_args():
    """Parse input arguments. Add arguments here."""
    parser = argparse.ArgumentParser(description="Collect features for cohort.")
    parser.add_argument("--data_dir", type=str, help="Path to input cleaned data.")
    parser.add_argument("--target_dir", type=str, help="Path to input target data.")
    parser.add_argument("--output_dir", type=str, help="Output path.")
    parser.add_argument("--min_year", type=int, default=2008, help="Minimum diagnosis year to keep (inclusive) -- defensive re-check, clean.py enforces this upstream.")
    args = parser.parse_args()
    return args


def setup_logging():
    """Configures logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == "__main__":
    # Set up argument parsing and logging
    args = parse_args()
    setup_logging()
    main(args)