"""Agent / AI assistant routes."""
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.schemas.underwriting import AgentChatRequest, AgentChatResponse
from backend.services.agent_service import agent_service
from backend.db.database import get_db
from backend.db.models import AuditLog

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest, db: Session = Depends(get_db)):
    """
    Ask the AI assistant a question about an applicant's underwriting result.

    The assistant uses the deterministic underwriting system as its source
    of truth. It does NOT invent applicant data, financial information,
    or regulatory requirements.

    LLM (Ollama) must be running locally. If unavailable, returns a
    deterministic summary instead.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        response = agent_service.chat(
            applicant_id=request.applicant_id,
            question=request.question,
            underwriting_context=request.underwriting_context,
        )

        # Log the interaction
        try:
            log = AuditLog(
                action="agent_chat",
                applicant_id=request.applicant_id,
                status="ok" if response.llm_available else "llm_unavailable",
                detail=f"Q: {request.question[:200]}",
            )
            db.add(log)
            db.commit()
        except Exception:
            db.rollback()

        return response

    except Exception as e:
        logger.error("Agent chat failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent chat failed.")


@router.get("/status")
def agent_status():
    """Check if the LLM is available."""
    return {
        "llm_available": agent_service.llm_available,
        "llm_provider": "ollama",
        "ollama_model": "llama3.1:8b",
        "note": (
            "Deterministic underwriting works without the LLM. "
            "The LLM is only needed for the AI assistant chat."
        ),
    }
