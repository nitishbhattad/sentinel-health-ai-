from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.genai_service import rag_chat

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    subject_id: int
    question  : str

@router.post("/")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = rag_chat(request.subject_id, request.question)
    return {
        "subject_id": request.subject_id,
        "question"  : request.question,
        "answer"    : result["answer"],
        "sources"   : result["sources"]
    }