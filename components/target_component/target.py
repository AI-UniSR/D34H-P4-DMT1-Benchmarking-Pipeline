'''
Explanation of File:
File that is used to create the dataframe with the target variable which will be used for further analysis.

Author: Muhammad Salaar Arslan
Date: 18/06/2025
Email: arslan.muhammad@hsr.it
'''

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def main(args: argparse.Namespace) -> pd.DataFrame:
    """In this function we prepare our dataframe so that we have all of the relevant columns that are related to our target features

    Args:
        args (argparse.Namespace): arguments that are being passed in such as input directory, output directory and tolerance

    Returns:
        pd.DataFrame: Returns two dataframes, one is the target.csv which is the one that will be passed onto the features.py script
        and the other one is for visualizations
    """
    input_dir = args.input_dir
    output_dir = args.output_dir

    with mlflow.start_run():

        mlflow.log_params(
            {
                "tolerance": args.tolerance,
                "min_age": args.min_age,
                "max_age": args.max_age,
            }
        )

        dataframes = load_data(input_dir)

        df = prepare_data(dataframes=dataframes, tolerance=args.tolerance)
        mlflow.log_metric("prepared_df_length", len(df))

        # Compute remission status with all methods
        df_comparison = compute_target(df=df, dataframes=dataframes)
        mlflow.log_metric("total_patients_all_methods", df_comparison.CPI.nunique())

        # Defensive re-check of the full age range -- clean.py already enforces
        # this upstream, so this should be a no-op (see clean_function docstring).
        # Metric name kept as-is (matches diagnose_idaa1c_attrition.py's
        # comparison target) even though the check now covers the full range,
        # not just age>=2 -- its value shouldn't change either way once
        # clean.py's filter is correct.
        clean_df = clean_function(df_comparison, min_age=args.min_age, max_age=args.max_age)
        mlflow.log_metric("total_patients_all_methods_older_than_2", clean_df.CPI.nunique())

        # Plot histograms of target values
        plot_data(clean_df, output_dir, )

        # Saving the data with all the variables
        save_data(output_dir, clean_df, 'all_targets.csv')


def load_data(input_dir: str,) -> dict[str, pd.DataFrame]:
    """Loads multiple datasets from different sources and returns them as a dictionary of pandas DataFrames.

    Args:
    - input_dir (str): The directory path where the cleaned CSV files are located.

    Returns:
    - dict(str, pd.DataFrame): A dictionary where keys are dataset names and values are the corresponding pandas
    DataFrames.
    """

    # 1. Load files created by clean.py
    cleaned_filenames = [
        "cohort.csv",
        "esami.csv",
        "insulin.csv",
        "lab.csv",
    ]

    dataframes = {file.split(".")[0]: pd.read_csv(os.path.join(input_dir, file)) for file in cleaned_filenames}

    return dataframes


def prepare_data(dataframes: dict[str, pd.DataFrame], tolerance: int) -> pd.DataFrame:
    """Preparing the Dataframes to create the target variables

    Args:
        dataframes (dict[str, pd.DataFrame]): dictionary of dataframes where the key is the name of the dataframe and the values
            are the dataframes
        tolerance (int): Number of days of tolerance for the merge_asof

    Returns:
        pd.DataFrame: Long dataframe of remission data
    """

    # --Insulina Data--
    # Selecting only the relevant columns from insulin
    insulin = dataframes["insulin"].copy()
    insulin = insulin[["CPI", "Date", "Insulin (u/24h)", "Rapid Acting Insulin", "Insulin Pump"]]

    # --Esami Data--
    esami_df = dataframes["esami"].copy()

    # Extracting all exams for HbA1c
    # Added rows that arent in % (i.e. CPI and date dont match) but were converted from mmol/mol
    glic = "Emoglob.Glicata HbA1c"
    esami_mmol = esami_df[(esami_df["Exam"] == glic) & (esami_df["Measure"] == "mmol/mol")].copy()
    esami_mmol["Value_percentage"] = (esami_mmol["Value"].astype(float) * 0.09148) + 2.152

    esami_mmol = esami_mmol.rename(
        columns={
            "Value_percentage": "HbA1c"
        }
    )

    esami_percentage = esami_df[(esami_df["Exam"] == glic) & (esami_df["Measure"] == "%")].copy()

    # Renaming the Value field to HbA1c
    esami_percentage.rename(
        columns={
            "Value": "HbA1c",
        },
        inplace=True,
    )

    # Concatenating the two esami dataframes to combine all the values together
    esami = pd.concat([
        esami_percentage,
        esami_mmol
    ])

    # Extracting the weight of the patients
    peso = esami_df[esami_df["Exam"] == "Peso"].copy()

    # Renaming the Peso Value field to Peso
    peso.rename(columns={"Value": "Peso"}, inplace=True)

    # Selecting only the relevant columns from peso and esami
    esami = esami[["CPI", "Date", "HbA1c"]]
    peso = peso[["CPI", "Date", "Peso"]]

    # --Galileo Data--
    lab = dataframes["lab"].copy()

    # Extracting all rows where there is a value for HbA1c in %
    # Also added rows that arent in % (i.e. CPI and date dont match) but can be converted from mmol/mol
    lab_mmol = lab[(lab["Method_text"] == "Sg-EMOGLOBINA GLICATA (HbA1c)") & (lab["Measure"] == "mmol/mol")].copy()
    lab_mmol["Value_percentage"] = (lab_mmol["Value"].astype(float) * 0.09148) + 2.152
    lab_mmol = lab_mmol.rename(
        columns={
            "Value_percentage": "HbA1c"
        }
    )
    lab_percentage = lab[(lab["Method_text"] == "Sg-EMOGLOBINA GLICATA (HbA1c)") & (lab["Measure"] == "%")].copy()
    lab_percentage = lab_percentage.rename(
        columns={
            "Value": "HbA1c",
        }
    )
    lab = pd.concat([
        lab_percentage,
        lab_mmol,
        ])

    lab = lab[["CPI", "Date", "HbA1c"]]

    # Adding the HbA1c dataset with the Galileo Dataset
    esami_lab = pd.concat([
        esami,
        lab,
    ])

    # Ensuring that there are no strings as values in the HbA1c column
    esami_lab["HbA1c"] = pd.to_numeric(esami_lab["HbA1c"]).round(2)

    # Converting Date into datetime format
    esami_lab["Date"] = pd.to_datetime(esami_lab["Date"])

    peso["Date"] = pd.to_datetime(peso["Date"])

    insulin["Date"] = pd.to_datetime(insulin["Date"])

    # esami_lab = esami_lab[esami_lab["Date"].dt.year != 1900]
    # peso = peso[peso["Date"].dt.year != 1900]
    # insulin = insulin[insulin["Date"].dt.year != 1900]

    # --Cohort Data--
    cohort = dataframes["cohort"].copy()

    # Merging Peso, Esami_lab, and Insulin data with CPI and T0
    esami_lab = esami_lab.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")
    peso = peso.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")
    insulin = insulin.merge(cohort[["CPI", "T0"]], on="CPI", how="inner")

    # Converting T0 into datetime format
    esami_lab["T0"] = pd.to_datetime(esami_lab["T0"])
    peso["T0"] = pd.to_datetime(peso["T0"])
    insulin["T0"] = pd.to_datetime(insulin["T0"])

    # Only including data that is with 36 Months after diagnosis per CPI
    cutoff = pd.Timedelta(days=30 * 36)

    esami_lab = esami_lab[
        (esami_lab["T0"] <= esami_lab["Date"]) & (esami_lab["Date"] <= esami_lab["T0"] + cutoff)
        ]
    peso = peso[
        (peso["T0"] <= peso["Date"]) & (peso["Date"] <= peso["T0"] + cutoff)
        ]
    insulin = insulin[
        (insulin["T0"] <= insulin["Date"]) & (insulin["Date"] <= insulin["T0"] + cutoff)
        ]

    # --Merging Datasets with Tolerance--
    # Ensuring that all the dates are ordered
    esami_lab.sort_values(["Date", "CPI"], inplace=True)
    peso.sort_values(["Date", "CPI"], inplace=True)
    insulin.sort_values(["Date", "CPI"], inplace=True)

    # Check to see if tolerance has been provided
    if tolerance is not None:
        days = pd.Timedelta(days=tolerance)

    # Merge HbA1c (from esami and lab) with the nearest previous Peso for each CPI.
    df = pd.merge_asof(esami_lab, peso, on="Date", by="CPI", direction="backward", suffixes=("", "_peso"), tolerance=days)

    # Merge the result with the nearest previous insulin dose for each CPI.
    df = pd.merge_asof(
        df.sort_values(["Date"]), insulin, on="Date", by="CPI", direction="backward", suffixes=("", "_insulin"), tolerance=days
    )

    # Calculate Derived Metrics Total Daily Dose Insulin and IDAA1c
    df["TDDI"] = df["Insulin (u/24h)"] / df["Peso"]
    df["IDAA1c"] = df["HbA1c"] + 4 * df["TDDI"]

    df = df.drop(columns=["T0_peso", "T0_insulin"]).sort_values(["CPI", "T0"])

    return df


def check_idaac_in_range(sub_df: pd.DataFrame, min_months: int, max_months: int) -> bool:
    """We are trying to check if within a specific time range, whether a patient's IDAA1c falls below 9

    Args:
        sub_df (pd.DataFrame): The dataframe that will be passed into this
        min_months (int): this is essentially month that we will start checking from
        max_months (int): this will be the last month that we will check up to.

    Returns:
        bool: return true if any IDAA1c within the time span falls below 9
    """
    mask = (sub_df["MonthsSinceDiagnosis"] >= min_months) & (sub_df["MonthsSinceDiagnosis"] < max_months)
    values = sub_df.loc[mask, "IDAA1c"].dropna()
    if values.empty:
        return np.nan
    return (values <= 9).any()


def get_idaac_values_in_range(sub_df: pd.DataFrame, min_months: int, max_months: int) -> list:
    """This function essentially just groups all the values within the specific time period and shows them as a list

    Args:
        sub_df (pd.DataFrame): The dataframe that will be passed into this
        min_months (int): this is essentially month that we will start checking from
        max_months (int): this will be the last month that we will check up to.

    Returns:
        list: returns a list of all IDAA1c values within the time period
    """
    return (
        sub_df[(sub_df["MonthsSinceDiagnosis"] >= min_months) & (sub_df["MonthsSinceDiagnosis"] < max_months)]["IDAA1c"]
        .dropna()
        .round(2)
        .tolist()
    )

def get_min_idaa1c_in_range(sub_df: pd.DataFrame, min_months: int, max_months: int) -> float:
    """Return the minimum IDAA1c value within a time range, or NaN if no values exist.

    Args:
        sub_df (pd.DataFrame): Patient-level dataframe.
        min_months (int): Start of range (inclusive).
        max_months (int): End of range (exclusive).

    Returns:
        float: Minimum IDAA1c value, or NaN if no data in range.
    """
    mask = (sub_df["MonthsSinceDiagnosis"] >= min_months) & (sub_df["MonthsSinceDiagnosis"] < max_months)
    values = sub_df.loc[mask, "IDAA1c"].dropna()
    if values.empty:
        return np.nan
    return round(values.min(), 1)


def _get_date_list(sub_df: pd.DataFrame) -> list:
    return sub_df['Date'].dt.date.tolist()


def _get_features_list(sub_df: pd.DataFrame, insulin: str) -> list:
    return sub_df[insulin].round(2).tolist()


def compute_target(df: pd.DataFrame, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create multiple target variables using predetermined time ranges.

    Args:
        df (pd.DataFrame): The dataframe that will be passed into this
        dataframes (dict[str, pd.DataFrame]): this is a dictionary of dataframes

    Returns:
        pd.DataFrame: The output would be a dataframe of all the relevant targets or features that contributed to calculating the target
    """
    # Drop rows where T0 is missing
    df = df.dropna(subset=["T0"]).copy()

    # Calculate months since diagnosis
    df["MonthsSinceDiagnosis"] = (df["Date"] - df["T0"]) / pd.Timedelta(days=30)

    # Group by CPI and compute values
    result = (
        df.groupby("CPI")
        .apply(
            lambda group: pd.Series({
                "Dates": _get_date_list(group),
                "Insulin": _get_features_list(group, "Insulin (u/24h)"),
                "Rapid Acting Insulin": _get_features_list(group, "Rapid Acting Insulin"),
                "Insulin Pump": _get_features_list(group, "Insulin Pump"),
                "TDDI": _get_features_list(group, "TDDI"),
                "Weight": _get_features_list(group, "Peso"),
                "IDAA1c": _get_features_list(group, "IDAA1c"),
                "HbA1c": _get_features_list(group, "HbA1c"),
                "IDAA1c_12m": check_idaac_in_range(group, 9, 16),
                "IDAA1c_12_24m": check_idaac_in_range(group, 9, 25),
                "IDAA1c_12_36m": check_idaac_in_range(group, 9, 37),
                "IDAA1c_vals_12m": get_idaac_values_in_range(group, 9, 16),
                "IDAA1c_vals_12_24m": get_idaac_values_in_range(group, 9, 25),
                "IDAA1c_vals_12_36m": get_idaac_values_in_range(group, 9, 37),
                "IDAA1c_min_12m": get_min_idaa1c_in_range(group, 9, 16),
                "IDAA1c_min_12_24m": get_min_idaa1c_in_range(group, 9, 25),
                "IDAA1c_min_12_36m": get_min_idaa1c_in_range(group, 9, 37),
                "Year_of_diagnosis": group["T0"].dt.year.iloc[0],
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Compute and merge age at diagnosis
    # TODO move somewhere else
    cohort = dataframes["cohort"].copy()
    cohort = cohort[["CPI", "Age at Diagnosis"]]
    result = cohort.merge(result, on="CPI", how="left")

    # Remove patients who have not shown remission in 12 months
    return result[
        result.IDAA1c_12m.notna() | result.IDAA1c_12_24m.notna() | result.IDAA1c_12_36m.notna()
    ]


def clean_function(df: pd.DataFrame, min_age: int = 2, max_age: int = 18) -> pd.DataFrame:
    """Defensive re-check of the age range clean.py already enforces upstream.

    clean.py's clean_data() now drops patients outside [min_age, max_age) before
    this script ever sees the cohort, so this should be a no-op -- if it isn't,
    that means the upstream filter has a bug, so this logs a warning rather than
    silently dropping the discrepancy (this previously only checked age >= 2,
    missing the upper bound entirely -- that's what let out-of-range patients
    through as far as benchmark.py).

    Args:
        df (pd.DataFrame): The dataframe that was created from the targets
        min_age (int): Minimum age at diagnosis to keep (inclusive).
        max_age (int): Maximum age at diagnosis to keep (exclusive).

    Returns:
        pd.DataFrame: Output will be to only have the correct patients
    """
    clean_df = df.copy()
    in_range = (clean_df["Age at Diagnosis"] >= min_age) & (clean_df["Age at Diagnosis"] < max_age)

    n_out_of_range = (~in_range).sum()
    if n_out_of_range > 0:
        logging.warning(
            f"[clean_function] {n_out_of_range} patient(s) outside age range "
            f"[{min_age}, {max_age}) reached target.py -- clean.py's upstream "
            f"age filter should have already removed these. Dropping them here "
            f"too, but this indicates the upstream filter isn't working as expected."
        )

    clean_df = clean_df[in_range]

    return clean_df


def save_data(output_dir: str, df: pd.DataFrame, file_name: str):
    """Function which specifies how and where to save the data

    Args:
        output_dir (str): the output directory
        df (pd.DataFrame): the dataframe that is to be saved
    """
    save_path = os.path.join(output_dir, file_name)
    df.to_csv(save_path, index=False)
    mlflow.log_artifact(save_path)


def plot_data(df: pd.DataFrame, output_dir: str):
    """Make a pieplot for every target variable.

    Args:
        df (pd.DataFrame): Input data containing target columns.
        output_dir (str): Output directory for plots.
    """

    targets = [
        "IDAA1c_12m", "IDAA1c_12_24m", "IDAA1c_12_36m"
    ]
    for target in targets:
        value_counts = df[target].value_counts(dropna=False)

        # Map boolean values to more readable labels and define color mapping
        label_map = {True: 'True', False: 'False', pd.NA: 'Missing/NaN'}
        labels = [f"{label_map.get(val, str(val))} ({count})"
              for val, count in zip(value_counts.index, value_counts.values)]  # noqa: B905

        # Create color mapping: True -> green, False -> red, NaN -> gray
        color_map = {True: 'lightgreen', False: 'lightcoral', pd.NA: 'lightgray'}
        colors = [color_map.get(val, 'lightblue') for val in value_counts.index]

        plt.figure(figsize=(8, 6))
        plt.pie(value_counts.values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors)

        plt.title(target)
        plt.axis('equal')

        path = os.path.join(output_dir, f"{target}_pie.png")
        plt.savefig(path, bbox_inches="tight", dpi=300)
        mlflow.log_artifact(path)
        plt.close()


def target_value_selection(target_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """We pass which target we would like to pass on to the features.py script

    Args:
        target_name (str): this will be one of the boolean targets that will be renamed as remission
        df (pd.DataFrame): the dataframe that will be passed with all of the target features

    Returns:
        pd.DataFrame: the final dataframe with 2 columns CPI and remission
    """
    clean_df = df.copy()

    clean_df = clean_df[["CPI", target_name]].rename(columns={target_name: "remission"})

    return clean_df[clean_df.remission.notna()]


def flag_oscillating_remission_nonremission(statuses: list) -> bool:
    """Identifying patients who experience remission after a period of nonremission

    Args:
        statuses (list): List of values from 12-36 months

    Returns:
        bool: False if there is no remission after a period of nonremission, else True
    """
    boolean = [True if value <= 9 else False for value in statuses]

    seen_remission = False
    seen_nonremission_after_remission = False

    for val in boolean:
        if val is True:
            if seen_nonremission_after_remission:
                return True
            seen_remission = True  # First remission seen
        elif val is False and seen_remission:
            seen_nonremission_after_remission = True  # False after a True
    return False


def setup_logging():
    """
    Configures logging to output to both the console and a log file.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for detailed logs during development
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # Logs to console
            logging.FileHandler("development.log", mode="w"),  # Logs to file
        ],
    )


def parse_args() -> argparse.Namespace:
    """
    Set up argument parser

    Returns:
        Namespace: returning the arguments from the command line
    """
    parser = argparse.ArgumentParser(description="Calculating multiple targets for maximal cohort of patients")

    parser.add_argument("--input_dir", type=str, help="Path to input data.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--tolerance", help="Adding the Tolerance for the merge_asof function in pandas", type=int, default=None)
    parser.add_argument("--min_age", type=int, default=2, help="Minimum age at diagnosis to keep (inclusive) -- defensive re-check, clean.py enforces this upstream.")
    parser.add_argument("--max_age", type=int, default=18, help="Maximum age at diagnosis to keep (exclusive) -- defensive re-check, clean.py enforces this upstream.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)