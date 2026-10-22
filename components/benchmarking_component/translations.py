"""
English translations for Italian/abbreviated clinical feature names.
Used for plot labels and table displays only — internal column names are unchanged.
"""

VARIABLE_TRANSLATION = {
    "AC. Celiachia": "Celiac Autoimmunity",
    "ANTICORPI ANTI IA2": "Anti-IA2",
    "ANTICORPI ANTI INSULINA": "Anti-Insulin",
    "ANTICORPI ANTI ZNT8": "Anti-ZnT8",
    "Altezza": "Height",
    "BMI": "BMI",
    "COLESTEROLO / HDL COLESTEROLO": "Cholesterol/HDL Cholesterol Ratio",
    "DIURESI": "Diuresis",
    "EMOCROMO - Ematocrito": "Hematocrit",
    "EMOCROMO - Emoglobina": "Hemoglobin",
    "EMOCROMO - Globuli bianchi": "White Blood Cells",
    "EMOCROMO - Globuli rossi": "Red Blood Cells",
    "EMOCROMO - MCH (Emoglobina Corpuscolare Media)": "MCH",
    "EMOCROMO - MCHC (Conc.Emoglobinica Corpusc.Media)": "MCHC",
    "EMOCROMO - MCV (Volume Globulare Medio)": "MCV",
    "EMOCROMO - RDW (Ind.Distribuzione Eritrocitaria)": "RDW",
    "FORMULA  - Basofili": "Basophils",
    "FORMULA  - Eosinofili": "Eosinophils",
    "FORMULA  - Linfociti": "Lymphocytes",
    "FORMULA  - Monociti": "Monocytes",
    "FORMULA  - Neutrofili": "Neutrophils",
    "Insulin": "Insulin",
    "PIASTRINE - MPV (Volume Piastrinico Medio)": "MPV",
    "PIASTRINE - Piastrine": "Platelets",
    "Peso": "Weight",
    "S-ANTICORPI ANTI GAD": "Anti-GAD Antibodies",
    "S-CALCIO": "Calcium",
    "S-COLESTEROLO": "Cholesterol",
    "S-COLESTEROLO HDL": "HDL Cholesterol",
    "S-CREATININA": "Creatinine",
    "S-GLUCOSIO": "Glucose",
    "S-IMMUNOGLOBULINE A": "Immunoglobulin A",
    "S-PEPTIDE C": "C-Peptide",
    "S-POTASSIO": "Potassium",
    "S-SODIO": "Sodium",
    "S-TIREOTROPINA (TSH)": "TSH",
    "S-TRIGLICERIDI": "Triglycerides",
    "Sg-EMOGLOBINA GLICATA (HbA1c)": "HbA1c",
    "U-ALBUMINA": "Urine Albumin",
    "URINE  - U-Aspetto": "Urine Appearance",
    "URINE  - U-Chetoni": "Urine Ketones",
    "URINE  - U-Colore": "Urine Color",
    "URINE  - U-Emoglobina": "Urine Hemoglobin",
    "URINE  - U-Glucosio": "Urine Glucose",
    "URINE  - U-Leucociti": "Urine Leukocytes",
    "URINE  - U-Nitriti": "Urine Nitrites",
    "URINE  - U-Peso specifico": "Urine Specific Gravity",
    "URINE  - U-Proteine": "Urine Protein",
    "URINE  - U-Urobilinogeno": "Urine Urobilinogen",
    "URINE  - U-pH": "Urine pH",
    "num_missing_diabetic_aa": "N of Missing Diabetic Autoab",
    "num_positive_diabetic_aa": "N of Positive Diabetic Autoab",
    "Sesso": "Sex",
    "Age at Diagnosis": "Age at Diagnosis",
}


def translate(name: str) -> str:
    """Translate a single feature name. Returns original if not in dictionary."""
    return VARIABLE_TRANSLATION.get(name, name)


def translate_list(names: list) -> list:
    """Translate a list of feature names."""
    return [translate(n) for n in names]