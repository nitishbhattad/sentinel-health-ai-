import ollama
import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/healthcare_db")
engine = create_engine(DATABASE_URL)
PERSIST_DIR = "models/chromadb"

# ── Load or build vector store once at startup ────────────────────
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    # Build from scratch
    df = pd.read_sql("SELECT * FROM notes", engine)
    documents = [
        Document(
            page_content=row["text"],
            metadata={
                "subject_id": int(row["subject_id"]),
                "hadm_id"   : int(row["hadm_id"]),
                "category"  : str(row["category"])
            }
        )
        for _, row in df.iterrows()
    ]
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

# Load once when module imports
vectorstore = get_vector_store()

# ── RAG Chat ──────────────────────────────────────────────────────
def rag_chat(subject_id: int, question: str) -> dict:
    results = vectorstore.similarity_search(
        query=question,
        k=3,
        filter={"subject_id": subject_id}
    )

    if not results:
        return {
            "answer": "No clinical notes found for this patient.",
            "sources": []
        }

    context = "\n\n".join([
        f"[{doc.metadata['category']}]\n{doc.page_content}"
        for doc in results
    ])

    prompt = f"""You are a clinical decision support assistant.

PATIENT ID: {subject_id}

CLINICAL NOTES:
{context}

QUESTION: {question}

Answer based only on the notes. Be concise and clinical.
If notes lack enough info, say so clearly."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response["message"]["content"],
        "sources": [doc.metadata for doc in results]
    }

# ── Risk Explainer ────────────────────────────────────────────────
def explain_risk(patient: dict, shap: dict) -> str:
    shap_sorted = sorted(
        shap.items(), key=lambda x: abs(x[1]), reverse=True
    )
    shap_lines = "\n".join([
        f"  - {k.replace('shap_', '')}: "
        f"{'increases' if v > 0 else 'decreases'} risk (impact={abs(v):.3f})"
        for k, v in shap_sorted
    ])

    prompt = f"""You are a clinical decision support assistant.

PATIENT DATA:
  Patient ID     : {patient['subject_id']}
  Risk Score     : {patient['risk_score']:.1%}
  Risk Tier      : {patient['risk_tier']}
  Admissions     : {patient['admission_count']}
  Emergency Ratio: {patient['emergency_ratio']:.1%}
  Avg Stay (days): {patient['avg_los_days']:.1f}
  Days Since Last: {patient['days_since_last_admission']:.0f}

TOP RISK DRIVERS:
{shap_lines}

Write a clear 3-sentence clinical explanation for a doctor.
Use the actual numbers. Do not invent information."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def generate_discharge_timeline(patient: dict, shap: dict) -> str:
    """LLM generates discharge timeline narrative"""  # ← 4 spaces indent

    ward = patient.get("predicted_ward", "General")
    risk = patient.get("risk_score", 0)
    los  = patient.get("estimated_los_days", 3)
    # Map ward to clinical context
    ward_context = {
        "ICU"    : "Intensive Care Unit — requires continuous monitoring, ventilator possible",
        "MICU"   : "Medical ICU — step-down from ICU, close monitoring required",
        "Private": "Private Room — stable but needs individual nursing attention",
        "General": "General Ward — routine monitoring, approaching discharge readiness"
    }.get(ward, "General Ward")

    shap_sorted = sorted(
        shap.items(), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    top_drivers = ", ".join([
        f"{k.replace('shap_', '')} ({'high' if v > 0 else 'low'} impact)"
        for k, v in shap_sorted
    ])

    prompt = f"""You are a senior hospital physician creating a discharge planning note.

PATIENT SUMMARY:
  Patient ID       : {patient['subject_id']}
  Risk Score       : {risk:.1%}
  Risk Tier        : {patient['risk_tier']}
  Assigned Ward    : {ward}
  Ward Description : {ward_context}
  Estimated Stay   : {los:.1f} days
  Key Risk Drivers : {top_drivers}
  Total Admissions : {patient['admission_count']}
  Emergency Rate   : {patient['emergency_ratio']:.1%}

Write a clinical discharge planning narrative with:
1. Ward placement rationale (1 sentence)
2. Expected timeline to discharge (specific days)
3. Key milestones before discharge can occur (2-3 bullet points)
4. Readmission prevention recommendation (1 sentence)

Be specific with numbers. Use clinical language."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]