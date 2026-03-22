"""
MIMIC-III Data Ingestion Script
Loads PATIENTS.csv.gz, ADMISSIONS.csv.gz, NOTEEVENTS.csv.gz
into the healthcare_db PostgreSQL database.
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

# ── Config ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/healthcare_db"
)
MIMIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mimic")
NOTES_LIMIT = 50000  # Load only first 50k notes to save memory

engine = create_engine(DATABASE_URL)


def check_files():
    """Check that all required MIMIC files exist."""
    required = ["PATIENTS.csv.gz", "ADMISSIONS.csv.gz", "NOTEEVENTS.csv.gz"]
    missing = []
    for f in required:
        path = os.path.join(MIMIC_DIR, f)
        if not os.path.exists(path):
            # Also check without .gz
            path_no_gz = path.replace(".csv.gz", ".csv")
            if not os.path.exists(path_no_gz):
                missing.append(f)
            else:
                print(f"  Found: {f.replace('.gz', '')} (uncompressed)")
        else:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  Found: {f} ({size_mb:.1f} MB)")

    if missing:
        print(f"\nMissing files: {missing}")
        print(f"Please download from: https://physionet.org/content/mimiciii/1.4/")
        print(f"Save to: {MIMIC_DIR}/")
        sys.exit(1)


def load_file(filename):
    """Load a CSV file, supporting both .gz and plain .csv."""
    path_gz = os.path.join(MIMIC_DIR, filename)
    path_csv = path_gz.replace(".csv.gz", ".csv")

    if os.path.exists(path_gz):
        print(f"  Loading {filename} (compressed)...")
        return pd.read_csv(path_gz, compression="gzip", low_memory=False)
    elif os.path.exists(path_csv):
        print(f"  Loading {filename.replace('.gz','')} (uncompressed)...")
        return pd.read_csv(path_csv, low_memory=False)
    else:
        raise FileNotFoundError(f"Cannot find {filename} in {MIMIC_DIR}")


def clear_existing_data():
    """Clear existing data from tables before loading MIMIC data."""
    print("\nClearing existing data...")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM patient_shap_values"))
        conn.execute(text("DELETE FROM patient_risk_scores"))
        conn.execute(text("DELETE FROM patient_features"))
        conn.execute(text("DELETE FROM notes"))
        conn.execute(text("DELETE FROM admissions"))
        conn.execute(text("DELETE FROM patients"))
        conn.commit()
    print("  Cleared all tables.")


def ingest_patients():
    """Load PATIENTS.csv.gz into patients table."""
    print("\nLoading PATIENTS...")
    df = load_file("PATIENTS.csv.gz")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Select and rename columns
    patients = pd.DataFrame({
        "subject_id": df["subject_id"],
        "gender": df["gender"],
        "dob": pd.to_datetime(df["dob"], errors="coerce").dt.date
    })

    patients = patients.drop_duplicates(subset=["subject_id"])

    patients.to_sql(
        "patients",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )
    print(f"  Loaded {len(patients)} patients.")
    return patients["subject_id"].tolist()


def ingest_admissions(valid_subject_ids):
    """Load ADMISSIONS.csv.gz into admissions table."""
    print("\nLoading ADMISSIONS...")
    df = load_file("ADMISSIONS.csv.gz")
    df.columns = df.columns.str.lower()

    # Filter to valid patients only
    df = df[df["subject_id"].isin(valid_subject_ids)]

    admissions = pd.DataFrame({
        "hadm_id": df["hadm_id"],
        "subject_id": df["subject_id"],
        "admittime": pd.to_datetime(df["admittime"], errors="coerce"),
        "dischtime": pd.to_datetime(df["dischtime"], errors="coerce"),
        "admission_type": df["admission_type"]
    })

    admissions = admissions.drop_duplicates(subset=["hadm_id"])

    admissions.to_sql(
        "admissions",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )
    print(f"  Loaded {len(admissions)} admissions.")
    return admissions["hadm_id"].tolist()


def ingest_notes(valid_subject_ids, valid_hadm_ids):
    """Load NOTEEVENTS.csv.gz into notes table (limited to NOTES_LIMIT rows)."""
    print(f"\nLoading NOTEEVENTS (limit: {NOTES_LIMIT})...")
    df = load_file("NOTEEVENTS.csv.gz")
    df.columns = df.columns.str.lower()

    # Filter to valid patients and admissions
    df = df[df["subject_id"].isin(valid_subject_ids)]
    df = df[df["hadm_id"].isin(valid_hadm_ids)]

    # Limit rows
    df = df.head(NOTES_LIMIT)

    notes = pd.DataFrame({
        "subject_id": df["subject_id"],
        "hadm_id": df["hadm_id"],
        "charttime": pd.to_datetime(df["charttime"], errors="coerce"),
        "category": df["category"] if "category" in df.columns else "Unknown",
        "text": df["text"]
    })

    # Drop rows with null text
    notes = notes.dropna(subset=["text"])

    notes.to_sql(
        "notes",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500
    )
    print(f"  Loaded {len(notes)} notes.")


def build_patient_features():
    """Rebuild patient_features table from MIMIC admissions data."""
    print("\nBuilding patient features...")
    query = """
        INSERT INTO patient_features (
            subject_id,
            admission_count,
            avg_los_days,
            max_los_days,
            emergency_ratio,
            days_since_last_admission,
            has_previous_admission,
            high_risk_label
        )
        SELECT
            p.subject_id,
            COUNT(a.hadm_id) AS admission_count,
            AVG(EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400) AS avg_los_days,
            MAX(EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400) AS max_los_days,
            SUM(CASE WHEN a.admission_type = 'EMERGENCY' THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(a.hadm_id), 0) AS emergency_ratio,
            EXTRACT(EPOCH FROM (NOW() - MAX(a.admittime)))/86400 AS days_since_last_admission,
            CASE WHEN COUNT(a.hadm_id) > 1 THEN 1 ELSE 0 END AS has_previous_admission,
            CASE WHEN AVG(EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400) > 6.5
                 THEN 1 ELSE 0 END AS high_risk_label
        FROM patients p
        LEFT JOIN admissions a ON p.subject_id = a.subject_id
        WHERE a.dischtime IS NOT NULL AND a.admittime IS NOT NULL
        GROUP BY p.subject_id
        ON CONFLICT (subject_id) DO UPDATE SET
            admission_count = EXCLUDED.admission_count,
            avg_los_days = EXCLUDED.avg_los_days,
            max_los_days = EXCLUDED.max_los_days,
            emergency_ratio = EXCLUDED.emergency_ratio,
            days_since_last_admission = EXCLUDED.days_since_last_admission,
            has_previous_admission = EXCLUDED.has_previous_admission,
            high_risk_label = EXCLUDED.high_risk_label
    """
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()

    count = pd.read_sql("SELECT COUNT(*) FROM patient_features", engine).iloc[0, 0]
    print(f"  Built features for {count} patients.")


def verify_ingestion():
    """Print summary of loaded data."""
    print("\n" + "="*50)
    print("MIMIC-III Ingestion Complete!")
    print("="*50)
    tables = ["patients", "admissions", "notes", "patient_features"]
    for table in tables:
        count = pd.read_sql(f"SELECT COUNT(*) FROM {table}", engine).iloc[0, 0]
        print(f"  {table:25s}: {count:,} rows")
    print("="*50)
    print("\nNext steps:")
    print("  1. python ml/train.py")
    print("  2. python ml/predict.py")
    print("  3. python ml/ward_model.py")
    print("  4. docker compose down && docker compose build && docker compose up -d")


if __name__ == "__main__":
    print("="*50)
    print("MIMIC-III Data Ingestion")
    print("="*50)

    # Step 1 - Check files exist
    print("\nChecking files...")
    check_files()

    # Step 2 - Clear existing data
    clear_existing_data()

    # Step 3 - Load patients
    valid_subject_ids = ingest_patients()

    # Step 4 - Load admissions
    valid_hadm_ids = ingest_admissions(valid_subject_ids)

    # Step 5 - Load notes
    ingest_notes(valid_subject_ids, valid_hadm_ids)

    # Step 6 - Build features
    build_patient_features()

    # Step 7 - Verify
    verify_ingestion()