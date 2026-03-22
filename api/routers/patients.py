from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import pandas as pd

from api.services.ml_service import (
    get_patient_risk,
    get_patient_shap,
    get_high_risk_patients,
    get_all_patients,
    engine
)
from api.services.genai_service import (
    explain_risk,
    generate_discharge_timeline
)
from api.services.pdf_service import generate_patient_report

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/")
def list_patients(limit: int = 100, tier: str = None):
    query = """
        SELECT DISTINCT
            r.subject_id,
            r.risk_score,
            r.risk_tier,
            f.admission_count,
            f.emergency_ratio
        FROM patient_risk_scores r
        JOIN patient_features f ON r.subject_id = f.subject_id
    """
    if tier and tier.upper() in ["HIGH", "MEDIUM", "LOW"]:
        query += " WHERE r.risk_tier = '" + tier.upper() + "'"
    query += " ORDER BY r.risk_score DESC LIMIT " + str(limit)
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")


@router.get("/highrisk")
def high_risk_patients(limit: int = 20):
    patients = get_high_risk_patients(limit)
    return {"count": len(patients), "patients": patients}


@router.get("/{subject_id}/risk")
def patient_risk(subject_id: int):
    patient = get_patient_risk(subject_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/{subject_id}/explain")
def patient_explanation(subject_id: int):
    patient = get_patient_risk(subject_id)
    shap = get_patient_shap(subject_id)
    if not patient or not shap:
        raise HTTPException(status_code=404, detail="Patient not found")
    explanation = explain_risk(patient, shap)
    return {
        "subject_id": subject_id,
        "risk_score": patient["risk_score"],
        "risk_tier": patient["risk_tier"],
        "explanation": explanation
    }


@router.get("/{subject_id}/ward")
def patient_ward(subject_id: int):
    patient = get_patient_risk(subject_id)
    shap = get_patient_shap(subject_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    timeline = generate_discharge_timeline(patient, shap or {})
    return {
        "subject_id": subject_id,
        "risk_score": patient["risk_score"],
        "risk_tier": patient["risk_tier"],
        "predicted_ward": patient.get("predicted_ward", "General"),
        "estimated_los_days": patient.get("estimated_los_days", 3),
        "discharge_timeline": timeline
    }


@router.get("/{subject_id}/report")
def download_report(subject_id: int):
    patient = get_patient_risk(subject_id)
    shap = get_patient_shap(subject_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    explanation = explain_risk(patient, shap or {})
    timeline = generate_discharge_timeline(patient, shap or {})
    pdf_bytes = generate_patient_report(
        patient=patient,
        shap=shap or {},
        explanation=explanation,
        discharge_timeline=timeline
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=patient_"
            + str(subject_id)
            + "_report.pdf"
        }
    )