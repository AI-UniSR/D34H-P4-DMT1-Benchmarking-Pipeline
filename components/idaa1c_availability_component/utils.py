"""
Utils for feature calculation.
Author: Edoardo Draetta
Date: 2025-05-22
"""
import re
from collections.abc import Iterable

import pandas as pd

# ---------------------------------------------------------------------------------
# COLLECT FEATURES
# ---------------------------------------------------------------------------------


def get_features_for_cohort_at_timepoint(
    cohort_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    feature_names_to_find: Iterable[str],
    feature_name_col: str,
    feature_value_col: str,
    feature_date_col: str,
    patient_index: str,
    target_date_col: str,
    offset_days: int,
    tolerance_days: int | None,
    feature_availability_threshold: float,
) -> pd.DataFrame:
    """Given a wide cohort dataframe and a tall feature dataframe, find all features at a provided timepoint.

    Args:
        cohort_dataframe (pd.DataFrame): Wide dataframe defining the cohort
        feature_dataframe (pd.DataFrame): Tall dataframe of features.
        feature_names_to_find (Iterable[str]): List of features to find.
        feature_name_col (str): Column in which to find feature names.
        feature_value_col (str): Column in which to find feature values.
        feature_date_col (str): Column in which to find feature dates.
        patient_index (str): Patient index column (must match between cohort and feature dataframes).
        target_date_col (str): Column in which to find cohort target date.
        offset_days (int): Days to add to target date in cohort dataframe.
        tolerance_days (int | None): Number of days in the past to seek valid features.
        feature_availability_threshold (foat): Minimum percentage of adherence for a given exam.

    Returns:
        pd.DataFrame: Dataframe of features found.
    """

    # 1. Prepare dataframes

    # Cohort: Add a target date for each patient (T0 + offset days)
    cohort_dataframe = cohort_dataframe.copy()
    cohort_dataframe[target_date_col] = pd.to_datetime(cohort_dataframe[target_date_col])
    cohort_dataframe["TargetDate"] = cohort_dataframe[target_date_col] + pd.Timedelta(days=offset_days)

    # Features dataframe:
    feature_dataframe = feature_dataframe.copy()
    # Filter to patients in cohort
    feature_dataframe = feature_dataframe[feature_dataframe[patient_index].isin(cohort_dataframe[patient_index])]

    # Enforce matching datetime types
    cohort_dataframe[target_date_col] = pd.to_datetime(cohort_dataframe[target_date_col]).dt.tz_localize(None)
    feature_dataframe[feature_date_col] = pd.to_datetime(feature_dataframe[feature_date_col]).dt.tz_localize(None)

    # Drop exams below threshold
    counts = feature_dataframe.groupby(feature_name_col)[patient_index].nunique()
    threshold = feature_dataframe[patient_index].nunique() * feature_availability_threshold
    exams_above_threshold = counts[counts > threshold].index  # type: ignore
    feature_dataframe = feature_dataframe[feature_dataframe[feature_name_col].isin(exams_above_threshold)]

    # Filter exams to feature_names_to_find
    feature_dataframe = feature_dataframe[feature_dataframe[feature_name_col].isin(feature_names_to_find)]

    # 3. Merge cohort onto feature dataframe to get TargetDate
    feature_dataframe = feature_dataframe.merge(
        cohort_dataframe[[patient_index, "TargetDate"]], on=patient_index, how="left"
    )

    # 4. Compute Time Difference
    feature_dataframe[feature_date_col] = pd.to_datetime(feature_dataframe[feature_date_col])
    feature_dataframe["TimeDiff"] = (feature_dataframe["TargetDate"] - feature_dataframe[feature_date_col]).dt.days

    # Since direction is 'backward', we only consider ExamDate <= TargetDate
    feature_dataframe = feature_dataframe[feature_dataframe["TimeDiff"] >= 0]

    # Apply tolerance
    if tolerance_days is not None:
        feature_dataframe = feature_dataframe[feature_dataframe["TimeDiff"] <= tolerance_days]

    # 5. For each patient and exam, find the record with minimum TimeDiff
    idx = feature_dataframe.groupby([patient_index, feature_name_col])["TimeDiff"].idxmin()
    feature_dataframe = feature_dataframe.loc[idx]

    # 6. Pivot the data to have one row per patient
    feature_pivot = feature_dataframe.pivot(index=patient_index, columns=feature_name_col, values=feature_value_col)
    feature_pivot.columns = [f"{col}" for col in feature_pivot.columns]

    # 7. Merge back to cohort dataframe
    cohort_dataframe = cohort_dataframe.merge(feature_pivot, on=patient_index, how="left")

    # Drop 'TargetDate' column
    cohort_dataframe.drop(columns=["TargetDate"], inplace=True)

    return cohort_dataframe

# ---------------------------------------------------------------------------------
# CLEAN AND ENCODE
# ---------------------------------------------------------------------------------


def clean_and_encode_exams(lab_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    The main entry point to clean & encode all lab exams.
    This function orchestrates the sub-cleaning functions.
    """
    df = lab_dataframe.copy()
    df["Encoded_value"] = pd.NA
    df = clean_antibody_exams(df)
    df = convert_lab_units(df)
    df = clean_urine_exams(df)
    df = clean_misc_exams(df)

    return df


def convert_lab_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert units of measure (and unit names) for a selection of exams.

    Args:
        df (pd.DataFrame): Lab dataframe.

    Returns:
        pd.DataFrame: Lab dataframe with converted units.
    """

    # Update unit names
    df["Measure"] = df["Measure"].str.replace("IU", "UI")  # map IU to UI
    df["Measure"] = df["Measure"].str.replace("µU/", "µUI/")
    df["Measure"] = df["Measure"].str.replace("Elementi/µL", "/µL")

    # Choose a unit for these exams
    exams_unit_choice = {
        "FORMULA  - Monociti": "10^9/L",
        "FORMULA  - Linfociti": "10^9/L",
        "FORMULA  - Basofili": "10^9/L",
        "FORMULA  - Eosinofili": "10^9/L",
        "FORMULA  - Neutrofili": "10^9/L",
        "Alfa 1 globuline": "g/L",
        "Alfa 2 globuline": "g/L",
        "Beta 1 globuline": "g/L",
        "Beta 2 globuline": "g/L",
        "Gamma globuline": "g/L",
        "Albumina": "g/L",
    }
    mask = df.apply(
        lambda row: pd.isna(row["Measure"])
        or (exams_unit_choice.get(row["Method_text"], row["Measure"]) == row["Measure"]),
        axis=1,
    )

    df = df[mask]

    # Convert exams to a specific unit
    exam = "EMOCROMO - Emoglobina"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "g/dL")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 10
    df.loc[mask, "Measure"] = "g/L"

    exam = "EMOCROMO - MCHC (Conc.Emoglobinica Corpusc.Media)"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "g Hb /dL")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 10
    df.loc[mask, "Measure"] = "g/L"

    exam = "Sg-EMOGLOBINA GLICATA (HbA1c)"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "mmol/mol")
    df.loc[mask, "Numeric_value"] = (df.loc[mask, "Numeric_value"] * 0.09148) + 2.152
    df.loc[mask, "Measure"] = "%"

    exam = "S-TRIIODOTIRONINA LIBERA (FT3)"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "pg/mg")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 1000
    df.loc[mask, "Measure"] = "pg/mL"

    exam = "PT - Rapporto"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "%")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 100
    df.loc[mask, "Measure"] = ""

    exam = "S-FOSFATASI ALCALINA"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "U/mL")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 1000
    df.loc[mask, "Measure"] = "U/L"

    exam = "S-TROPONINA T"
    mask = (df["Method_text"] == exam) & (df["Measure"] == "ng/L")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 1000
    df.loc[mask, "Measure"] = "µg/L"

    # Drop some units
    exam = "SEDIMENTO - U-Eritrociti"
    mask = (df["Method_text"] != exam) | ((df["Method_text"] == exam) & (df["Measure"] != "/µL"))
    df = df[mask]

    return df


def clean_urine_exams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning and encoding for all URINE-related exams.
    """
    exam = "U-CREATININA"  # Already clean
    exam = "Albumin excretion rate"  # Already clean

    exam = "URINE  - U-Nitriti"
    positive_negative_encoding = {
        "Negativo": 0,
        "Positivo": 1,
    }
    strings_to_drop = ["-"]
    df = drop_all_numeric_values_from_exam(df, exam)
    df = encode_string_values(df, exam, positive_negative_encoding)
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "URINE  - U-Leucociti"
    assenti_presenti_encoding = {
        "Assenti": 0,
        "Rari": 0,
        "Presenti": 1,
    }
    df = drop_all_numeric_values_from_exam(df, exam)
    df = encode_string_values(df, exam, assenti_presenti_encoding)

    exam = "URINE  - U-Chetoni"
    df = drop_all_string_values_from_exam(df, exam)

    exam = "URINE  - U-Glucosio"  # == 'U-GLUCOSIO'
    exam_to_convert = "U-GLUCOSIO"
    mask = (df["Method_text"] == exam_to_convert) & (df["Measure"] == "g/L")
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Numeric_value"] * 100
    df.loc[mask, "Measure"] = "mg/dL"
    df = rename_exam(df, exam_to_convert, exam)
    df = drop_all_string_values_from_exam(df, exam)

    exam = "URINE  - U-Aspetto"
    urine_aspetto_encoding = {"Limpido": 0, "Sublimpido": 1, "Torbido": 1, "Molto Torbido": 1}  # nonlimpida
    df = drop_all_numeric_values_from_exam(df, exam)
    df = encode_string_values(df, exam, urine_aspetto_encoding)

    exam = "URINE  - U-Colore"
    urine_colore_encoding = {  # Extremely low cardinality
        "Giallo Paglierino": 0,
        "Giallo Oro": 0,
        "Giallo Rossastro": 1,
        "Ambra": 1,
        "Bruno": 1,
        "Ematico": 1,
    }
    df = encode_string_values(df, exam, urine_colore_encoding)

    exam = "URINE  - U-Peso specifico"  # mixed: string & numeric. encode into bins.
    urine_peso_string_encoding = {
        ">1.030": 1.5,
        "<1.005": 0.5,
    }
    df = encode_string_values(df, exam, urine_peso_string_encoding)
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] < 1.005)
    df.loc[mask, "Encoded_value"] = 0
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] <= 1.030) & (df["Numeric_value"] >= 1.005)
    df.loc[mask, "Encoded_value"] = 1
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] > 1.030)
    df.loc[mask, "Encoded_value"] = 2
    mask = df["Method_text"] == exam
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "U-ALBUMINA"  # encode strings, then bin. == 'U-albuminuria'?
    exam_to_rename = "U-albuminuria"
    df = rename_exam(df, exam_to_rename, exam)
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains("<"))
    df.loc[mask, "Numeric_value"] = 0
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains("300 mg/L"))
    df.loc[mask, "Numeric_value"] = 450
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains(">"))
    df.loc[mask, "Numeric_value"] = 450
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] < 30)
    df.loc[mask, "Encoded_value"] = 0
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] <= 300) & (df["Numeric_value"] >= 30)
    df.loc[mask, "Encoded_value"] = 1
    mask = (df["Method_text"] == exam) & (df["Numeric_value"] > 300)
    df.loc[mask, "Encoded_value"] = 1
    mask = df["Method_text"] == exam
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "DIURESI"  # qta di urina prodotta, drop strings
    strings_to_drop = ["Tempo di raccolta non segnalato.", "Diuresi Non Segnalata", "Singola Minzione"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "Rapporto U-ALBUMINA/U-CREATININA"
    strings_to_drop = ["Tempo di raccolta non segnalato.", "Singola Minzione"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "SEDIMENTO - U-Eritrociti"
    strings_to_drop = ["5-10 / campo"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "U-PROTEINE"
    urine_proteine_encoding = {"<0.04": 0}
    df = encode_string_values(df, exam, urine_proteine_encoding)

    exam = "Minuti"
    strings_to_drop = ["Tempo di raccolta non segnalato.", "Singola Minzione"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "COLT. URINE MITTO INTERMEDIO - Colturale"
    strings_to_drop = [  # notice use of multiline strings
        "Sviluppo di flora batterica polimicrobica in alta carica. Possibile contaminazione.",
        "Sviluppo di flora batterica polimicrobica in bassa carica. Probabile contaminazione.  "
        "Il reperto ha scarso significato clinico.",
        "Sviluppo di flora batterica polimicrobica in bassa carica. Probabile contaminazione. "
        "Il reperto ha scarso significato clinico.",
    ]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)
    appt_rapp_encoding = {
        "Nessuna crescita. Limite di sensibilita' del metodo : < 1000 CFU/mL.": 0,
        "Negativo": 0,
        "Positivo": 1,
        "Sviluppo di miceti lievitiformi": 1,
    }
    df = encode_string_values(df, exam, appt_rapp_encoding)

    exams_that_are_too_sparse = [
        "U-FOSFATO INORGANICO",
        "SEDIMENTO - U-Batteri",
        "URINE  - U-Bilirubina",
        "U-UREA",
    ]
    for exam in exams_that_are_too_sparse:
        df = drop_exam(df, exam)

    return df


def clean_antibody_exams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning and encoding for Antibody-related exams.
    """
    # Rename some columns for consistency
    df["Method_text"] = df["Method_text"].replace(
        {
            "S-ANTICORPI ANTI INSULINA": "ANTICORPI ANTI INSULINA",
            "S-ANTICORPI ANTI ZNT8": "ANTICORPI ANTI ZNT8",
            "S-ANTICORPI ANTI IA2": "ANTICORPI ANTI IA2",
            "AC. ANTI  TRANSGLUTAMINASI (IgG)": "AC. ANTI TRANSGLUTAMINASI (IgG)",
            "AC. ANTI  TRANSGLUTAMINASI (IgA)": "AC. ANTI TRANSGLUTAMINASI (IgA)",
            "ANTICORPI VS. TRANSGLUTAMINASI (IgA)": "AC. ANTI TRANSGLUTAMINASI (IgA)",
            "AC. IgA  ANTI ENDOMISIO": "AC. IgA ANTI ENDOMISIO",
        }
    )

    # ---------------------------------------------------------------------------------
    # 1. DIABETE (categorical encodings)
    # ---------------------------------------------------------------------------------

    exam = "ANTICORPI ANTI INSULINA"
    mask = (df["Method_text"] == exam) & (df["String_value"] == "0,2")
    df.loc[mask, "Numeric_value"] = 0.2
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)

    exam = "S-ANTICORPI ANTI GAD"
    strings_to_drop = ["160\n.sp.4"]  # type: ignore
    df = drop_string_values_from_exam(df, exam, strings_to_drop)
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)

    exam = "ANTICORPI ANTI ZNT8"
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)

    exam = "ANTICORPI ANTI IA2"
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)

    exam = "ANTICORPI VS. ISOLA PANCREATICA-ICA"
    ica_encoding = {  # TODO: change to contains "positivo"
        "Negativo": 0,
        "Assenti": 0,
        "dubbio": 0,
        "Tracce": 1,
        "Positivo": 1,
        "Positivo debole": 1,
        "Positivo++": 1,
        "Positivo ++": 1,
        "Positivo++ 1/5": 1,
        "Debolmente Positivo": 1,
    }
    df = encode_string_values(df, exam, ica_encoding)

    # ---------------------------------------------------------------------------------
    # 2. DIABETE (numeric) - copy original numeric values to new "(QUANT)" columns
    # ---------------------------------------------------------------------------------

    diabete_exams = [
        "ANTICORPI ANTI INSULINA",
        "S-ANTICORPI ANTI GAD",
        "ANTICORPI ANTI ZNT8",
        "ANTICORPI ANTI IA2",
        # Ignore ICA (not quant)
    ]

    # new_rows = []
    # for exam in diabete_exams:
    #     # Select the rows matching the current exam
    #     mask = df["Method_text"] == exam
    #     selected_rows = df[mask].copy()

    #     # Create a new column for original values
    #     selected_rows["Method_text"] = exam + " (QUANT)"
    #     selected_rows["Numeric_value"] = selected_rows["Numeric_value"]

    #     # Append modified rows to the list
    #     new_rows.append(selected_rows)

    # df = pd.concat([df] + new_rows, ignore_index=True)

    # Keep the categorical encoded columns
    for exam in diabete_exams:
        # Select the rows matching the current exam
        mask = df["Method_text"] == exam
        # Keep encoded value as a numeric value for this version
        df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    # ---------------------------------------------------------------------------------
    # 3. THYROID
    # ---------------------------------------------------------------------------------

    exam = "ANTICORPI ANTI TIREOPEROSSIDASI"
    df = encode_antibody_by_range(df, exam)
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "ANTICORPI ANTI TIREOGLOBULINA"
    df = encode_antibody_by_range(df, exam)
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    # ---------------------------------------------------------------------------------
    # 4. CELIAC
    # ---------------------------------------------------------------------------------

    exam = "AC. ANTI TRANSGLUTAMINASI (IgA)"
    mask = (df["Method_text"] == exam) & (df["String_value"] == "0.1s")
    df.loc[mask, "Numeric_value"] = 0.1
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)
    mask = df["Method_text"] == exam
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "AC. vs.TRANSGLUTAMINASI (IgA) reflex"
    mask = df["Method_text"] == exam
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "AC. ANTI TRANSGLUTAMINASI (IgG)"
    mask = df["Method_text"] == exam
    df = encode_antibody_low_high(df, exam)
    df = encode_antibody_by_range(df, exam)
    df.loc[mask, "Numeric_value"] = df.loc[mask, "Encoded_value"]

    exam = "AC. IgA ANTI ENDOMISIO"
    endomisio_encoding = {
        "Assenti": 0,
        "Positivo": 0,
        "Tracce": 1,
        "Positivo debole": 1,
        "Tracce 1:5": 1,
        "Presenti 1:5": 1,
        "Presenti 1:640": 1,
        "Presenti": 1,
    }
    df = encode_string_values(df, exam, endomisio_encoding)

    # ---------------------------------------------------------------------------------
    # GASTRITE AUTOIMMUNE
    # ---------------------------------------------------------------------------------

    exam = "AC. ANTI CELL.PARIETALI GASTRICHE"
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains("Assenti"))
    df.loc[mask, "Numeric_value"] = 0
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains("Tracce"))
    df.loc[mask, "Numeric_value"] = 0
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains("Presenti"))
    df.loc[mask, "Numeric_value"] = 1

    # ---------------------------------------------------------------------------------
    # 5. DROP LOW-VALUE EXAMS
    # ---------------------------------------------------------------------------------

    exams_to_drop = [
        "S-Anti HCV  anticorpi totali",  # Extremely low cardinality
        "Ricerca Anticorpi Irregolari",  # Low patient count
        "S-ANTICORPI ed ANTIGENE p24 HIV 1-2",  # Low patient count
        "S-Anti Treponema pallidum anticorpi totali",  # Low patient count, low cardinality
    ]

    for exam in exams_to_drop:
        df = drop_exam(df, exam)

    return df


def clean_misc_exams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning and encoding for miscellaneous labs (TSH, Creatinina, CRP, etc.)
    """

    # tiroide
    exam = "S-TIREOTROPINA (TSH)"
    strings_to_drop = ["Non Determinato"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)
    df = encode_sensitivity_lower_bound(df, exam)
    exam = "S-PEPTIDE C"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-TIROXINA LIBERA (FT4)"
    df = encode_sensitivity_lower_bound(df, exam)
    df = encode_sensitivity_upper_bound(df, exam)
    exam = "S-TRIIODOTIRONINA LIBERA (FT3)"
    df = encode_sensitivity_upper_bound(df, exam)

    exam = "S-CREATININA"
    mask = (df["Method_text"] == exam) & (df["String_value"] == "0.27.")
    df.loc[mask, "Numeric_value"] = 0.27

    exam = "COLESTEROLO / HDL COLESTEROLO"
    mask = (df["Method_text"] == exam) & (df["String_value"].str.contains(","))
    df.loc[mask, "Numeric_value"] = df.loc[mask, "String_value"].str.replace(",", ".").astype("float32")

    exam = "S-ALANINA AMINOTRANSFERASI"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-ASPARTATO AMINOTRANSFERASI"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-SODIO"
    mask = (df["Method_text"] == exam) & (df["String_value"] == "137.4.")
    df.loc[mask, "Numeric_value"] = 137.4

    exam = "S-CALCIO"
    mask = (df["Method_text"] == exam) & (df["String_value"] == "2.51.")
    df.loc[mask, "Numeric_value"] = 2.51

    exam = "S-IMMUNOGLOBULINE A"
    df = encode_sensitivity_lower_bound(df, exam)
    imm_a_encoding = {"Valore inferiore alla sensibilita' del metodo": 0}
    df = encode_string_values(df, exam, imm_a_encoding)

    exam = "S-PROTEINA C REATTIVA"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-CREATINFOSFOCHINASI"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-AMILASI"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "APTT - Rapporto"
    df = encode_sensitivity_lower_bound(df, exam)
    appt_rapp_encoding = {"Valore inferiore alla sensibilita' del metodo": 0}
    df = encode_string_values(df, exam, appt_rapp_encoding)

    exam = "APTT - Secondi"
    df = encode_sensitivity_lower_bound(df, exam)
    appt_secondi_encoding = {"Valore inferiore alla sensibilita' del metodo": 0}
    df = encode_string_values(df, exam, appt_secondi_encoding)

    exam = "S-BILIRUBINA DIRETTA"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "Bilirubina Totale (nBili)"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-FOLATI"
    df = encode_sensitivity_upper_bound(df, exam)

    exam = "S-VITAMINA B 12"
    df = encode_sensitivity_lower_bound(df, exam)
    df = encode_sensitivity_upper_bound(df, exam)

    exam = "S-25 OH VITAMINA D"
    df = encode_sensitivity_lower_bound(df, exam)
    df = encode_sensitivity_upper_bound(df, exam)

    exam = "S-OSMOLALITA'"
    strings_to_drop = ["(FE)"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "Glucosio (Glucometro)"
    strings_to_drop = ["Analisi non eseguibile", "NI"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "S-ORMONE FOLLICOLO STIMOLANTE FSH"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-LUTEOTROPINA (LH)"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "Chetoni (Glucometro)"
    chetoni_encoding = {"Valore inferiore alla sensibilita' del metodo": 0}
    df = encode_string_values(df, exam, chetoni_encoding)
    strings_to_drop = ["Analisi non eseguibile"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    exam = "S-HBsAg (Antigene di superficie HBV)"
    df = drop_exam(df, exam)

    exam = "Fattore RH"
    strings_to_drop = ["Provetta non firmata", "Du"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)
    fh_encoding = {"Positivo": 1, "Negativo": 0}
    df = encode_string_values(df, exam, fh_encoding)

    exam = "S-ESTRADIOLO"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-INSULINA"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "S-ANTIGENE AUSTRALIA (HBsAg)"
    austr_encoding = {
        "Negativo": 0,
        "Positivo": 1,
    }
    df = encode_string_values(df, exam, austr_encoding)

    exam = "S-TROPONINA T"
    df = encode_sensitivity_lower_bound(df, exam)

    exam = "D-DIMERO FIBRINA"
    df = encode_sensitivity_upper_bound(df, exam)

    exam = "P-PARATORMONE"
    strings_to_drop = ["Campioni fuori ghiaccio"]
    df = drop_string_values_from_exam(df, exam, strings_to_drop)

    return df


def rename_exam(df: pd.DataFrame, fro: str, to: str) -> pd.DataFrame:
    """Rename an exam in the dataframe"""
    mask = df["Method_text"] == fro
    df.loc[mask, "Method_text"] = to
    return df


def drop_string_values_from_exam(df: pd.DataFrame, exam: str, values: list[str]) -> pd.DataFrame:
    """
    Drop rows for the specified exam where String_value is in the given values.
    """
    mask_exam = df["Method_text"] == exam
    mask_drop = df["String_value"].isin(values)
    return df[~(mask_exam & mask_drop)]


def drop_all_string_values_from_exam(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    Drop rows for the specified exam if they have non-null String_value.
    """
    mask_exam = df["Method_text"] == exam
    mask_string = df["String_value"].notna()
    return df[~(mask_exam & mask_string)]


def drop_all_numeric_values_from_exam(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    Drop rows for the specified exam if they have non-null Numeric_value.
    """
    mask_exam = df["Method_text"] == exam
    mask_numeric = df["Numeric_value"].notna()
    return df[~(mask_exam & mask_numeric)]


def drop_exam(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    Drop all rows for the specified exam.
    """
    return df[df["Method_text"] != exam]


def encode_string_values(df: pd.DataFrame, exam: str, mapping: dict) -> pd.DataFrame:
    """
    Encode String_value -> Numeric_value for the specified exam using the mapping.
    """
    mask_exam = (df["Method_text"] == exam) & (df["String_value"].isin(mapping.keys()))
    df.loc[mask_exam, "Numeric_value"] = df.loc[mask_exam, "String_value"].map(mapping)
    df.loc[mask_exam, "Encoded_value"] = df.loc[mask_exam, "Numeric_value"]
    return df


def encode_sensitivity_lower_bound(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    Encodes rows with String_value containing '<' as 0 for the specified exam.
    """
    mask_exam = (df["Method_text"] == exam) & df["String_value"].notna()
    mask_lt = df["String_value"].str.contains("<", regex=False)
    df.loc[mask_exam & mask_lt, "Numeric_value"] = 0
    return df


def encode_sensitivity_upper_bound(df: pd.DataFrame, exam: str, plus_value: float = 1.0) -> pd.DataFrame:
    """
    Encodes rows with String_value containing '>' as numeric_value + plus_value for the specified exam.
    Uses a regex to extract the float in the string.
    """
    mask_exam = (df["Method_text"] == exam) & df["String_value"].notna()
    mask_gt = df["String_value"].str.contains(">", regex=False)

    def extract_number(s: str) -> float | None:
        match = re.findall(r"\d+\.?\d*", s)
        return float(match[0]) if match else None

    df.loc[mask_exam & mask_gt, "Numeric_value"] = (
        df.loc[mask_exam & mask_gt, "String_value"].apply(extract_number) + plus_value
    )
    return df


def encode_antibody_low_high(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    For antibody exams with < or > in String_value, encode based on Range_min / Range_max.
    """
    # Example: if string_value contains "<" => set Numeric_value = Range_min
    #          if string_value contains ">" => set Numeric_value = Range_max + 1
    mask_exam = df["Method_text"] == exam
    mask_has_string_value = df["String_value"].notna()

    # < case
    mask_lt = mask_exam & mask_has_string_value & df["String_value"].str.contains("<")
    df.loc[mask_lt & df["Range_min"].notna(), "Numeric_value"] = df.loc[mask_lt, "Range_min"]

    # > case
    mask_gt = mask_exam & mask_has_string_value & df["String_value"].str.contains(">")
    df.loc[mask_gt & df["Range_max"].notna(), "Numeric_value"] = df.loc[mask_gt, "Range_max"] + 1

    return df


def encode_antibody_by_range(df: pd.DataFrame, exam: str) -> pd.DataFrame:
    """
    If Numeric_value > Range_max => 1 else 0,
    None if range missing
    """
    mask_exam = (df["Method_text"] == exam) & df["Numeric_value"].notna()

    # Split into two groups: with Range_max, without Range_max
    has_range = mask_exam & df["Range_max"].notna()
    no_range = mask_exam & df["Range_max"].isna()

    # Use the provided range for those that have it
    df.loc[has_range, "Encoded_value"] = (df.loc[has_range, "Numeric_value"] > df.loc[has_range, "Range_max"]).astype(
        int
    )

    # Use a fallback for those missing a Range_max
    df.loc[no_range, "Encoded_value"] = pd.NA

    # # Overwrite Numeric_value with 0/1
    # df.loc[mask_exam, "Numeric_value"] = df.loc[mask_exam, "Encoded_value"]

    return df
