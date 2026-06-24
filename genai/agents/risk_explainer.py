import ollama
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://localhost:5432/healthcare_db")

def get_patient_risk(subject_id: int) -> dict:
    """Pull risk score from patient_risk_scores table"""
    query = f"""
        SELECT r.subject_id, r.risk_score, r.risk_tier,
               f.admission_count, f.emergency_ratio,
               f.avg_los_days,
               COALESCE(f.days_since_last_admission, -1) AS days_since_last_admission
        FROM patient_risk_scores r
        JOIN patient_features f ON r.subject_id = f.subject_id
        WHERE r.subject_id = {subject_id};
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def get_patient_shap(subject_id: int) -> dict:
    """Pull SHAP values for this patient"""
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

def build_prompt(patient: dict, shap: dict) -> str:
    """Build a structured prompt for the LLM"""

    # Sort SHAP values by absolute impact
    shap_sorted = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_lines = "\n".join([
        f"  - {k.replace('shap_', '')}: SHAP={v:+.4f} ({'increases' if v > 0 else 'decreases'} risk)"
        for k, v in shap_sorted
    ])

    prompt = f"""You are a clinical decision support assistant helping doctors understand patient risk.

PATIENT DATA:
  Patient ID     : {patient['subject_id']}
  Risk Score     : {patient['risk_score']:.1%}
  Risk Tier      : {patient['risk_tier']}
  Admissions     : {patient['admission_count']}
  Emergency Ratio: {patient['emergency_ratio']:.1%}
  Avg Stay (days): {patient['avg_los_days']:.1f}
  Days Since Last Admission: {patient['days_since_last_admission']:.0f} days

SHAP FEATURE CONTRIBUTIONS (what drove this prediction):
{shap_lines}

Write a clear 3-sentence clinical explanation of this patient's risk for a doctor.
Focus on the top SHAP drivers. Be specific with the numbers. Do not make up information."""

    return prompt

def explain_patient(subject_id: int):
    """Full pipeline: DB → SHAP → LLM → explanation"""

    print(f"\n{'='*55}")
    print(f"  Risk Explanation for Patient {subject_id}")
    print(f"{'='*55}")

    # 1. Fetch data
    patient = get_patient_risk(subject_id)
    if not patient:
        print(f"❌ Patient {subject_id} not found")
        return

    shap = get_patient_shap(subject_id)
    if not shap:
        print(f"❌ SHAP values not found for patient {subject_id}")
        return

    # 2. Show raw data
    print(f"\n📊 Risk Score : {patient['risk_score']:.1%}")
    print(f"🏷  Risk Tier  : {patient['risk_tier']}")
    print(f"🏥 Admissions : {patient['admission_count']}")
    print(f"🚨 Emergency  : {patient['emergency_ratio']:.1%}")

    # 3. Build prompt and call LLM
    print(f"\n🤖 Generating explanation...\n")
    prompt = build_prompt(patient, shap)

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    explanation = response["message"]["content"]
    print(f"📝 Clinical Explanation:\n")
    print(explanation)
    print(f"\n{'='*55}\n")

    return explanation


if __name__ == "__main__":
    # Test on top 3 highest risk patients
    for patient_id in [284, 470, 731]:
        explain_patient(patient_id)