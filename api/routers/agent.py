from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from genai.agents.supervisor import analyze_patient

router = APIRouter(prefix="/agent", tags=["Multi-Agent"])


class AnalysisRequest(BaseModel):
    query: str = "Assess this patient's risk and provide clinical recommendations."


@router.post("/{patient_id}/analyze")
def analyze(patient_id: int, request: AnalysisRequest):
    """
    Run full multi-agent clinical analysis for a patient.

    Runs 3 agents in sequence:
    1. Risk Agent    → fetches risk score + clinical features
    2. Case Agent    → retrieves relevant clinical notes
    3. Report Agent  → synthesizes structured clinical report
    """
    try:
        result = analyze_patient(
            patient_id=patient_id,
            query=request.query
        )

        if result.get("error") and not result.get("clinical_report"):
            raise HTTPException(
                status_code=500,
                detail=f"Agent pipeline failed: {result['error']}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/{patient_id}/analyze")
def analyze_get(patient_id: int, query: str = "Assess this patient's risk."):
    """
    GET version for easy browser/curl testing.
    """
    try:
        result = analyze_patient(
            patient_id=patient_id,
            query=query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
