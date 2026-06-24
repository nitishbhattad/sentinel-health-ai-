from genai.agents.state import ClinicalState
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

PERSIST_DIR = "models/chromadb"

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("  [Case Agent] Loading vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return _vectorstore


def case_agent(state: ClinicalState) -> ClinicalState:
    """
    Agent 2 — Retrieves similar clinical notes from ChromaDB
    Queries your existing vector store for relevant context
    """
    patient_id = state["patient_id"]
    query = state.get("query", "patient risk assessment")

    print(f"  [Case Agent] Searching clinical notes for patient {patient_id}...")

    try:
        vectorstore = get_vectorstore()

        # Search for relevant notes for this specific patient
        results = vectorstore.similarity_search(
            query=query,
            k=3,
            filter={"subject_id": patient_id}
        )

        if results:
            notes = [
                f"[{doc.metadata.get('category', 'Note')} - "
                f"{doc.metadata.get('charttime', 'Unknown')}]\n"
                f"{doc.page_content[:300]}"
                for doc in results
            ]
            state["similar_notes"] = notes
            print(f"  [Case Agent] ✅ Found {len(notes)} relevant notes")
        else:
            # Fallback — search without patient filter
            results = vectorstore.similarity_search(query=query, k=3)
            notes = [
                f"[Similar Case - {doc.metadata.get('category', 'Note')}]\n"
                f"{doc.page_content[:300]}"
                for doc in results
            ]
            state["similar_notes"] = notes
            print(f"  [Case Agent] ✅ Found {len(notes)} similar case notes")

    except Exception as e:
        print(f"  [Case Agent] ❌ Error: {e}")
        state["similar_notes"] = ["No clinical notes available."]
        state["error"] = f"Case Agent failed: {str(e)}"

    return state
