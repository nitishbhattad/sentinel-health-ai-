import pandas as pd
from sqlalchemy import create_engine

import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/healthcare_db")
engine = create_engine(DATABASE_URL)

def get_patient_risk(subject_id: int) -> dict:
    query = f"""
        SELECT r.subject_id, r.risk_score, r.risk_tier,
               r.predicted_ward,
               r.estimated_los_days,
               f.admission_count, f.emergency_ratio,
               f.avg_los_days,
               COALESCE(f.days_since_last_admission, -1) 
                   AS days_since_last_admission
        FROM patient_risk_scores r
        JOIN patient_features f ON r.subject_id = f.subject_id
        WHERE r.subject_id = {subject_id};
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    return df.iloc[0].to_dict()
def get_patient_shap(subject_id: int) -> dict:
    query = f"""
        SELECT shap_admission_count,
               shap_emergency_ratio,
               shap_days_since_last_admission,
               shap_has_previous_admission
        FROM patient_shap_values
        WHERE subject_id = {subject_id};
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def get_high_risk_patients(limit: int = 20) -> list:
    query = f"""
        SELECT r.subject_id, r.risk_score, r.risk_tier,
               f.admission_count, f.emergency_ratio
        FROM patient_risk_scores r
        JOIN patient_features f ON r.subject_id = f.subject_id
        WHERE r.risk_tier = 'HIGH'
        ORDER BY r.risk_score DESC
        LIMIT {limit};
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")

def get_all_patients(limit: int = 100) -> list:
    query = f"""
        SELECT r.subject_id, r.risk_score, r.risk_tier,
               f.admission_count, f.emergency_ratio
        FROM patient_risk_scores r
        JOIN patient_features f ON r.subject_id = f.subject_id
        ORDER BY r.risk_score DESC
        LIMIT {limit};
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")