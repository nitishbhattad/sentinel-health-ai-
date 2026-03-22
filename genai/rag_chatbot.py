import ollama
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

engine = create_engine("postgresql://localhost:5432/healthcare_db")

# ── 1. Load notes from PostgreSQL ────────────────────────────────
def load_notes(subject_id: int = None) -> list[Document]:
    """Load patient notes as LangChain Documents"""
    if subject_id:
        query = f"""
            SELECT subject_id, hadm_id, charttime, category, text
            FROM notes
            WHERE subject_id = {subject_id}
            ORDER BY charttime;
        """
    else:
        query = """
            SELECT subject_id, hadm_id, charttime, category, text
            FROM notes
            ORDER BY subject_id, charttime;
        """

    df = pd.read_sql(query, engine)
    print(f"📄 Loaded {len(df)} notes from database")

    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["text"],
            metadata={
                "subject_id": int(row["subject_id"]),
                "hadm_id"   : int(row["hadm_id"]),
                "charttime" : str(row["charttime"]),
                "category"  : str(row["category"])
            }
        )
        documents.append(doc)

    return documents

# ── 2. Build Vector Store ─────────────────────────────────────────
def build_vector_store(documents: list[Document], persist_dir="models/chromadb"):
    """Convert notes to embeddings and store in ChromaDB"""

    print("🔢 Creating embeddings (this takes ~1-2 minutes)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store built with {len(documents)} documents")
    print(f"✅ Saved to {persist_dir}")
    return vectorstore

# ── 3. Load existing Vector Store ────────────────────────────────
def load_vector_store(persist_dir="models/chromadb"):
    """Load already-built vector store from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print(f"✅ Vector store loaded from {persist_dir}")
    return vectorstore

# ── 4. RAG Query ──────────────────────────────────────────────────
def ask_patient(vectorstore, subject_id: int, question: str):
    """Answer a question about a patient using RAG"""

    print(f"\n{'='*55}")
    print(f"  Patient {subject_id} — {question}")
    print(f"{'='*55}")

    # Search for relevant notes for this patient
    results = vectorstore.similarity_search(
        query=question,
        k=3,
        filter={"subject_id": subject_id}
    )

    if not results:
        print(f"❌ No notes found for patient {subject_id}")
        return

    # Build context from retrieved notes
    context = "\n\n".join([
        f"[{doc.metadata['category']} - {doc.metadata['charttime']}]\n{doc.page_content}"
        for doc in results
    ])

    print(f"📄 Found {len(results)} relevant notes\n")

    # Build RAG prompt
    prompt = f"""You are a clinical assistant helping doctors review patient history.

PATIENT ID: {subject_id}

RELEVANT CLINICAL NOTES:
{context}

DOCTOR'S QUESTION: {question}

Answer based only on the notes provided. Be concise and clinical.
If the notes don't contain enough information, say so clearly."""

    # Call LLM
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]
    print(f"🤖 Answer:\n{answer}")
    print(f"\n{'='*55}\n")
    return answer


# ── 5. Main ───────────────────────────────────────────────────────
if __name__ == "__main__":

    persist_dir = "models/chromadb"

    # Build or load vector store
    if not os.path.exists(persist_dir):
        print("🏗  Building vector store for the first time (~2 mins)...")
        documents = load_notes()
        vectorstore = build_vector_store(documents, persist_dir)
    else:
        print("📂 Loading existing vector store...")
        vectorstore = load_vector_store(persist_dir)

    print("\n" + "="*55)
    print("  🏥 Healthcare RAG Chatbot")
    print("  Ask anything about your patients")
    print("  Type 'exit' to quit")
    print("="*55)

    # Ask for patient ID once
    while True:
        try:
            subject_id = int(input("\n👤 Enter Patient ID (e.g. 284): "))
            break
        except ValueError:
            print("❌ Please enter a valid number")

    print(f"\n✅ Loaded context for Patient {subject_id}")
    print("💬 Now ask any question about this patient\n")

    # Chat loop
    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        # Switch patient
        if question.lower().startswith("switch patient"):
            try:
                subject_id = int(question.split()[-1])
                print(f"✅ Switched to Patient {subject_id}")
            except:
                print("❌ Usage: switch patient 470")
            continue

        ask_patient(vectorstore, subject_id, question)