import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta

engine = create_engine("postgresql://nitishbhattad@localhost/healthcare_db")

num_patients = 1000

patients = pd.DataFrame({
    "subject_id": range(1, num_patients + 1),
    "gender": np.random.choice(["M", "F"], num_patients),
    "dob": pd.date_range(start="1960-01-01", periods=num_patients, freq="7D")
})

admissions_list = []
hadm_id = 1

for pid in patients["subject_id"]:
    num_adm = np.random.randint(1, 4)
    for _ in range(num_adm):
        admit = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 100))
        discharge = admit + timedelta(days=np.random.randint(1, 10))
        admissions_list.append({
            "hadm_id": hadm_id,
            "subject_id": pid,
            "admittime": admit,
            "dischtime": discharge,
            "admission_type": np.random.choice(["EMERGENCY", "ELECTIVE"])
        })
        hadm_id += 1

admissions = pd.DataFrame(admissions_list)

notes_list = []

for _, row in admissions.iterrows():
    num_notes = np.random.randint(1, 4)
    for _ in range(num_notes):
        notes_list.append({
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "charttime": row["admittime"],
            "category": np.random.choice(["Physician", "Nursing"]),
            "text": "Sample clinical note"
        })

notes = pd.DataFrame(notes_list)

patients.to_sql("patients", engine, if_exists="append", index=False)
admissions.to_sql("admissions", engine, if_exists="append", index=False)
notes.to_sql("notes", engine, if_exists="append", index=False)

print("Synthetic data inserted successfully.")