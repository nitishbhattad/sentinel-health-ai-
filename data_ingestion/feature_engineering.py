"""
Enhanced Feature Engineering for MIMIC-III
Adds: Age, Charlson Comorbidity Index, ICU stays, diagnosis count
Target: ROC-AUC 0.80+
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/healthcare_db"
)
MIMIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mimic")

engine = create_engine(DATABASE_URL)


def load_file(filename):
    path_gz  = os.path.join(MIMIC_DIR, filename)
    path_csv = path_gz.replace(".csv.gz", ".csv")
    if os.path.exists(path_gz):
        print(f"  Loading {filename}...")
        return pd.read_csv(path_gz, compression="gzip", low_memory=False)
    elif os.path.exists(path_csv):
        print(f"  Loading {filename.replace('.gz','')}...")
        return pd.read_csv(path_csv, low_memory=False)
    else:
        raise FileNotFoundError(f"Cannot find {filename} in {MIMIC_DIR}")


# ── Charlson ICD-9 Code Mappings ───────────────────────────────────────────
# Each condition maps to a list of ICD-9 prefixes and a CCI weight
CHARLSON_CODES = {
    "myocardial_infarction":      (["410", "412"], 1),
    "congestive_heart_failure":   (["428"], 1),
    "peripheral_vascular":        (["440", "441", "443", "444", "445"], 1),
    "cerebrovascular":            (["430","431","432","433","434","435","436","437","438"], 1),
    "dementia":                   (["290"], 1),
    "chronic_pulmonary":          (["490","491","492","493","494","495","496",
                                    "500","501","502","503","504","505"], 1),
    "connective_tissue":          (["710","714","725"], 1),
    "peptic_ulcer":               (["531","532","533","534"], 1),
    "mild_liver":                 (["571","573"], 1),
    "diabetes_uncomplicated":     (["2500","2501","2502","2503"], 1),
    "diabetes_complicated":       (["2504","2505","2506","2507","2508","2509"], 2),
    "hemiplegia":                 (["342","343"], 2),
    "renal_disease":              (["582","583","585","586","588"], 2),
    "cancer":                     (["140","141","142","143","144","145","146","147",
                                    "148","149","150","151","152","153","154","155",
                                    "156","157","158","159","160","161","162","163",
                                    "164","165","166","167","168","169","170","171",
                                    "172","174","175","176","177","178","179","180",
                                    "181","182","183","184","185","186","187","188",
                                    "189","190","191","192","193","194","195"], 2),
    "severe_liver":               (["5722","5723","5724","5728"], 3),
    "metastatic_cancer":          (["196","197","198","199"], 6),
    "aids":                       (["042","043","044"], 6),
}


def compute_charlson(diagnoses_df):
    """Compute Charlson Comorbidity Index per subject_id."""
    print("  Computing Charlson Comorbidity Index...")

    diag = diagnoses_df.copy()
    diag.columns = diag.columns.str.lower()

    # Keep only ICD-9 codes
    if "icd9_code" in diag.columns:
        diag = diag[["subject_id", "icd9_code"]].dropna()
        diag["icd9_code"] = diag["icd9_code"].astype(str).str.strip()
    else:
        print("  Warning: icd9_code column not found")
        return pd.DataFrame(columns=["subject_id", "charlson_score"])

    # Build charlson score per subject
    results = {}

    for condition, (prefixes, weight) in CHARLSON_CODES.items():
        # Check if any diagnosis matches this condition
        mask = diag["icd9_code"].apply(
            lambda x: any(x.startswith(p) for p in prefixes)
        )
        matched = diag[mask][["subject_id"]].drop_duplicates()
        matched[condition] = weight

        for _, row in matched.iterrows():
            sid = row["subject_id"]
            if sid not in results:
                results[sid] = 0
            results[sid] += weight

    charlson_df = pd.DataFrame(
        list(results.items()),
        columns=["subject_id", "charlson_score"]
    )

    print(f"  Charlson scores computed for {len(charlson_df)} patients")
    print(f"  Score distribution:\n{charlson_df['charlson_score'].describe()}")
    return charlson_df


def compute_diagnosis_counts(diagnoses_df):
    """Count number of unique diagnoses per subject."""
    print("  Computing diagnosis counts...")
    diag = diagnoses_df.copy()
    diag.columns = diag.columns.str.lower()

    counts = diag.groupby("subject_id")["icd9_code"].nunique().reset_index()
    counts.columns = ["subject_id", "num_diagnoses"]
    print(f"  Diagnosis counts for {len(counts)} patients")
    return counts


def compute_icu_features(icustays_df):
    """Compute ICU-related features per subject."""
    print("  Computing ICU features...")
    icu = icustays_df.copy()
    icu.columns = icu.columns.str.lower()

    # Count ICU stays
    icu_counts = icu.groupby("subject_id").agg(
        num_icu_stays=("icustay_id", "count"),
        total_icu_hours=("los", "sum")
    ).reset_index()

    icu_counts["had_icu"] = 1
    print(f"  ICU features for {len(icu_counts)} patients")
    return icu_counts


def compute_age_features(patients_df, admissions_df):
    """Compute age at first admission."""
    print("  Computing age features...")
    pat = patients_df.copy()
    adm = admissions_df.copy()
    pat.columns = pat.columns.str.lower()
    adm.columns = adm.columns.str.lower()

    pat["dob"] = pd.to_datetime(pat["dob"], errors="coerce")
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")

    # Get first admission per patient
    first_adm = adm.groupby("subject_id")["admittime"].min().reset_index()
    first_adm.columns = ["subject_id", "first_admit"]

    # Merge with patients
    merged = pat.merge(first_adm, on="subject_id", how="left")

    # Compute age (MIMIC uses shifted dates for privacy, cap at 90)
    merged["age"] = (
        (merged["first_admit"] - merged["dob"]).dt.days / 365.25
    ).clip(0, 90)

    # Gender encoding
    merged["gender_male"] = (merged["gender"] == "M").astype(int)

    age_df = merged[["subject_id", "age", "gender_male"]].dropna()
    print(f"  Age features for {len(age_df)} patients")
    print(f"  Age distribution:\n{age_df['age'].describe()}")
    return age_df


def update_patient_features(enhanced_df):
    """Update patient_features table with new columns."""
    print("\n  Updating patient_features table...")

    # Add new columns if they don't exist
    new_columns = {
        "age":             "FLOAT",
        "gender_male":     "INTEGER",
        "charlson_score":  "FLOAT",
        "num_diagnoses":   "INTEGER",
        "num_icu_stays":   "INTEGER",
        "total_icu_hours": "FLOAT",
        "had_icu":         "INTEGER",
    }

    with engine.connect() as conn:
        for col, dtype in new_columns.items():
            try:
                conn.execute(text(f"""
                    ALTER TABLE patient_features
                    ADD COLUMN IF NOT EXISTS {col} {dtype} DEFAULT 0
                """))
            except Exception as e:
                print(f"  Warning for {col}: {e}")
        conn.commit()

    # Update rows
    updated = 0
    for _, row in enhanced_df.iterrows():
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE patient_features SET
                        age            = :age,
                        gender_male    = :gender_male,
                        charlson_score = :charlson_score,
                        num_diagnoses  = :num_diagnoses,
                        num_icu_stays  = :num_icu_stays,
                        total_icu_hours= :total_icu_hours,
                        had_icu        = :had_icu
                    WHERE subject_id = :subject_id
                """), {
                    "subject_id":     int(row["subject_id"]),
                    "age":            float(row.get("age", 0) or 0),
                    "gender_male":    int(row.get("gender_male", 0) or 0),
                    "charlson_score": float(row.get("charlson_score", 0) or 0),
                    "num_diagnoses":  int(row.get("num_diagnoses", 0) or 0),
                    "num_icu_stays":  int(row.get("num_icu_stays", 0) or 0),
                    "total_icu_hours":float(row.get("total_icu_hours", 0) or 0),
                    "had_icu":        int(row.get("had_icu", 0) or 0),
                })
                conn.commit()
                updated += 1
        except Exception as e:
            pass

        if updated % 5000 == 0 and updated > 0:
            print(f"  Updated {updated} rows...")

    print(f"  Total updated: {updated} rows")


def main():
    print("="*55)
    print("Enhanced Feature Engineering — MIMIC-III")
    print("Target: ROC-AUC 0.80+")
    print("="*55)

    # ── Load files ─────────────────────────────────────────
    print("\nLoading MIMIC files...")
    patients_df   = load_file("PATIENTS.csv.gz")
    admissions_df = load_file("ADMISSIONS.csv.gz")
    diagnoses_df  = load_file("DIAGNOSES_ICD.csv.gz")

    # Load ICU stays if available
    icu_available = True
    try:
        icustays_df = load_file("ICUSTAYS.csv.gz")
    except FileNotFoundError:
        print("  ICUSTAYS.csv.gz not found — skipping ICU features")
        icu_available = False

    # ── Compute features ───────────────────────────────────
    print("\nComputing features...")
    age_df      = compute_age_features(patients_df, admissions_df)
    charlson_df = compute_charlson(diagnoses_df)
    diag_df     = compute_diagnosis_counts(diagnoses_df)

    if icu_available:
        icu_df = compute_icu_features(icustays_df)
    else:
        icu_df = pd.DataFrame(columns=[
            "subject_id", "num_icu_stays", "total_icu_hours", "had_icu"
        ])

    # ── Merge all features ─────────────────────────────────
    print("\nMerging features...")
    enhanced = age_df
    enhanced = enhanced.merge(charlson_df, on="subject_id", how="left")
    enhanced = enhanced.merge(diag_df,     on="subject_id", how="left")
    enhanced = enhanced.merge(icu_df,      on="subject_id", how="left")

    # Fill NaN with 0
    enhanced = enhanced.fillna(0)

    print(f"\nEnhanced features shape: {enhanced.shape}")
    print(f"Columns: {list(enhanced.columns)}")
    print(f"\nSample stats:")
    print(enhanced[["age","charlson_score","num_diagnoses","num_icu_stays"]].describe())

    # ── Update database ────────────────────────────────────
    print("\nUpdating database...")
    update_patient_features(enhanced)

    print("\n" + "="*55)
    print("Feature Engineering Complete!")
    print("="*55)
    print("\nNew features added:")
    print("  age              → patient age at first admission")
    print("  gender_male      → gender encoding")
    print("  charlson_score   → comorbidity burden (0-37)")
    print("  num_diagnoses    → number of unique diagnoses")
    print("  num_icu_stays    → number of ICU admissions")
    print("  total_icu_hours  → total time in ICU")
    print("  had_icu          → ever had ICU stay (binary)")
    print("\nNext step: python ml/train_enhanced.py")


if __name__ == "__main__":
    main()