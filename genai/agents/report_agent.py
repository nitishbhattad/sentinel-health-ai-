import ollama
from genai.agents.state import ClinicalState


def report_agent(state: ClinicalState) -> ClinicalState:
    """
    Agent 3 — Synthesizes risk data + clinical notes into structured report
    Uses your existing Ollama llama3.2 model
    """
    patient_id = state["patient_id"]
    print(f"  [Report Agent] Generating clinical report for patient {patient_id}...")

    try:
        # Build context from previous agents
        risk_summary = f"""
Patient ID:        {patient_id}
Risk Score:        {state.get('risk_score', 0):.1%}
Risk Tier:         {state.get('risk_tier', 'Unknown')}
Assigned Ward:     {state.get('predicted_ward', 'Unknown')}
Est. LOS:          {state.get('estimated_los_days', 0):.1f} days
Age:               {state.get('age', 0):.0f} years
Charlson Score:    {state.get('charlson_score', 0):.0f}
Num Diagnoses:     {state.get('num_diagnoses', 0)}
ICU Stays:         {state.get('num_icu_stays', 0)}
Total ICU Hours:   {state.get('total_icu_hours', 0):.1f}
Admission Count:   {state.get('admission_count', 0)}
Emergency Ratio:   {state.get('emergency_ratio', 0):.1%}
"""
        notes_context = "\n\n".join(
            state.get("similar_notes", ["No clinical notes available."])
        )

        prompt = f"""You are a senior clinical decision support system.
Analyze this patient and generate a structured clinical report.

PATIENT RISK DATA:
{risk_summary}

RELEVANT CLINICAL NOTES:
{notes_context}

CLINICIAN QUERY: {state.get('query', 'Assess this patient')}

Generate a structured clinical report with these sections:
1. RISK ASSESSMENT — summarize the risk level and key drivers
2. CLINICAL CONTEXT — what the notes tell us about this patient
3. WARD RECOMMENDATION — justify the ward assignment
4. ACTION ITEMS — 3 specific clinical actions to take
5. DISCHARGE OUTLOOK — estimated timeline and considerations

Be concise, clinical, and actionable. Maximum 300 words."""

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        state["final_report"] = response["message"]["content"]
        print(f"  [Report Agent] ✅ Report generated ({len(state['final_report'])} chars)")

    except Exception as e:
        print(f"  [Report Agent] ❌ Error: {e}")
        state["final_report"] = f"Report generation failed: {str(e)}"
        state["error"] = f"Report Agent failed: {str(e)}"

    return state
