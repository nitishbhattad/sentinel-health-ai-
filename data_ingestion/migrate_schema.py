"""
Schema Migration Script for MIMIC-III
Run this BEFORE mimic_ingest.py to ensure
the database schema is compatible with MIMIC-III data.
"""

from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/healthcare_db"
)

engine = create_engine(DATABASE_URL)


def migrate():
    print("="*50)
    print("Running Schema Migration for MIMIC-III")
    print("="*50)

    with engine.connect() as conn:

        # 1 - Add primary key to patient_risk_scores if missing
        print("\n[1] Checking patient_risk_scores primary key...")
        result = conn.execute(text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'patient_risk_scores'
            AND constraint_type = 'PRIMARY KEY'
        """)).fetchone()

        if not result:
            conn.execute(text("""
                ALTER TABLE patient_risk_scores
                ADD PRIMARY KEY (subject_id)
            """))
            print("  Added primary key to patient_risk_scores.")
        else:
            print("  Primary key already exists.")

        # 2 - Add primary key to patient_shap_values if missing
        print("\n[2] Checking patient_shap_values primary key...")
        result = conn.execute(text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'patient_shap_values'
            AND constraint_type = 'PRIMARY KEY'
        """)).fetchone()

        if not result:
            conn.execute(text("""
                ALTER TABLE patient_shap_values
                ADD PRIMARY KEY (subject_id)
            """))
            print("  Added primary key to patient_shap_values.")
        else:
            print("  Primary key already exists.")

        # 3 - Add predicted_ward column if missing
        print("\n[3] Checking predicted_ward column...")
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'patient_risk_scores'
            AND column_name = 'predicted_ward'
        """)).fetchone()

        if not result:
            conn.execute(text("""
                ALTER TABLE patient_risk_scores
                ADD COLUMN predicted_ward VARCHAR(20)
            """))
            print("  Added predicted_ward column.")
        else:
            print("  predicted_ward already exists.")

        # 4 - Add estimated_los_days column if missing
        print("\n[4] Checking estimated_los_days column...")
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'patient_risk_scores'
            AND column_name = 'estimated_los_days'
        """)).fetchone()

        if not result:
            conn.execute(text("""
                ALTER TABLE patient_risk_scores
                ADD COLUMN estimated_los_days FLOAT
            """))
            print("  Added estimated_los_days column.")
        else:
            print("  estimated_los_days already exists.")

        # 5 - Add high_risk_label to patient_features if missing
        print("\n[5] Checking high_risk_label column...")
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'patient_features'
            AND column_name = 'high_risk_label'
        """)).fetchone()

        if not result:
            conn.execute(text("""
                ALTER TABLE patient_features
                ADD COLUMN high_risk_label INTEGER DEFAULT 0
            """))
            print("  Added high_risk_label column.")
        else:
            print("  high_risk_label already exists.")

        # 6 - Add indexes for performance
        print("\n[6] Adding indexes...")
        indexes = [
            ("idx_risk_tier", "patient_risk_scores", "risk_tier"),
            ("idx_risk_score", "patient_risk_scores", "risk_score"),
        ]
        for idx_name, table, col in indexes:
            result = conn.execute(text(f"""
                SELECT indexname FROM pg_indexes
                WHERE indexname = '{idx_name}'
            """)).fetchone()
            if not result:
                conn.execute(text(f"""
                    CREATE INDEX {idx_name} ON {table}({col})
                """))
                print(f"  Created index {idx_name}.")
            else:
                print(f"  Index {idx_name} already exists.")

        conn.commit()

    print("\n" + "="*50)
    print("Migration Complete!")
    print("="*50)
    print("\nNext step: python data_ingestion/mimic_ingest.py")


if __name__ == "__main__":
    migrate()