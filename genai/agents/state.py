from typing import TypedDict, List, Optional

class ClinicalState(TypedDict):
    # Input
    patient_id: int
    query: str

    # Risk Agent output
    risk_score: Optional[float]
    risk_tier: Optional[str]
    predicted_ward: Optional[str]
    estimated_los_days: Optional[float]
    admission_count: Optional[int]
    emergency_ratio: Optional[float]
    age: Optional[float]
    charlson_score: Optional[float]
    num_diagnoses: Optional[int]
    num_icu_stays: Optional[int]
    total_icu_hours: Optional[float]
    shap_values: Optional[dict]

    # Case Agent output
    similar_notes: Optional[List[str]]

    # Report Agent output
    final_report: Optional[str]

    # Error tracking
    error: Optional[str]
