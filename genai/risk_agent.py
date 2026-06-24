import requests
from genai.agents.state import ClinicalState

# Your existing API endpoint
API_BASE = "http://localhost:8000"


def risk_agent(state: ClinicalState) -> ClinicalState:
    """
    Agent 1 — Fetches patient risk score and SHAP values
    Calls your existing /patients/{id}/risk endpoint
    """
    patient_id = state["patient_id"]
    print(f"  [Risk Agent] Fetching risk data for patient {patient_id}...")

    try:
        # Call your existing risk endpoint
        risk_response = requests.get(
            f"{API_BASE}/patients/{patient_id}/risk",
            timeout=10
        )
        risk_response.raise_for_status()
        risk_data = risk_response.json()

        # Update state with risk data
        state["risk_score"]        = risk_data.get("risk_score")
        state["risk_tier"]         = risk_data.get("risk_tier")
        state["predicted_ward"]    = risk_data.get("predicted_ward")
        state["estimated_los_days"]= risk_data.get("estimated_los_days")
        state["admission_count"]   = risk_data.get("admission_count")
        state["emergency_ratio"]   = risk_data.get("emergency_ratio")
        state["age"]               = risk_data.get("age")
        state["charlson_score"]    = risk_data.get("charlson_score")
        state["num_diagnoses"]     = risk_data.get("num_diagnoses")
        state["num_icu_stays"]     = risk_data.get("num_icu_stays")
        state["total_icu_hours"]   = risk_data.get("total_icu_hours")

        print(f"  [Risk Agent] ✅ Risk: {state['risk_score']:.2%} {state['risk_tier']} → {state['predicted_ward']}")

    except Exception as e:
        print(f"  [Risk Agent] ❌ Error: {e}")
        state["error"] = f"Risk Agent failed: {str(e)}"

    return state
