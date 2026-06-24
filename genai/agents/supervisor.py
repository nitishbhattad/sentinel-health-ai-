from langgraph.graph import StateGraph, END
from genai.agents.state import ClinicalState
from genai.agents.risk_agent import risk_agent
from genai.agents.case_agent import case_agent
from genai.agents.report_agent import report_agent


def build_graph():
    """
    Build the LangGraph workflow:
    Risk Agent → Case Agent → Report Agent → END
    """
    workflow = StateGraph(ClinicalState)

    # Add all three agents as nodes
    workflow.add_node("risk", risk_agent)
    workflow.add_node("cases", case_agent)
    workflow.add_node("report", report_agent)

    # Define the flow
    workflow.set_entry_point("risk")
    workflow.add_edge("risk", "cases")
    workflow.add_edge("cases", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# Build graph once at module level
graph = build_graph()


def analyze_patient(patient_id: int, query: str) -> dict:
    """
    Run the full multi-agent pipeline for a patient.
    Returns structured clinical report.
    """
    print(f"\n{'='*55}")
    print(f"  Multi-Agent Analysis — Patient {patient_id}")
    print(f"  Query: {query}")
    print(f"{'='*55}")

    # Initialize state
    initial_state = ClinicalState(
        patient_id=patient_id,
        query=query,
        risk_score=None,
        risk_tier=None,
        predicted_ward=None,
        estimated_los_days=None,
        admission_count=None,
        emergency_ratio=None,
        age=None,
        charlson_score=None,
        num_diagnoses=None,
        num_icu_stays=None,
        total_icu_hours=None,
        shap_values=None,
        similar_notes=None,
        final_report=None,
        error=None
    )

    # Run through LangGraph
    result = graph.invoke(initial_state)

    print(f"{'='*55}")
    print(f"  ✅ Analysis complete!")
    print(f"{'='*55}\n")

    return {
        "patient_id":          result["patient_id"],
        "query":               result["query"],
        "risk_score":          result["risk_score"],
        "risk_tier":           result["risk_tier"],
        "predicted_ward":      result["predicted_ward"],
        "estimated_los_days":  result["estimated_los_days"],
        "age":                 result["age"],
        "charlson_score":      result["charlson_score"],
        "num_diagnoses":       result["num_diagnoses"],
        "num_icu_stays":       result["num_icu_stays"],
        "total_icu_hours":     result["total_icu_hours"],
        "similar_notes_count": len(result.get("similar_notes") or []),
        "clinical_report":     result["final_report"],
        "error":               result.get("error")
    }
