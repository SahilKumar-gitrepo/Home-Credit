"""
Agentic AI Credit Underwriting System — FastAPI Backend

RESEARCH/EDUCATIONAL PROTOTYPE ONLY.
This application is built for a final-year B.Tech research project.
It does NOT constitute a real lending system.
It is NOT legally compliant or production-ready.
Regulatory material is for informational/research context only.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src/ to Python path BEFORE any service imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from backend.config import settings
from backend.db.database import create_tables

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.
    Load all heavy resources once — model, features, RAG.
    """
    logger.info("=" * 60)
    logger.info("Starting Agentic AI Credit Underwriting System")
    logger.info("=" * 60)

    # Create audit database tables
    logger.info("Initialising audit database...")
    try:
        create_tables()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.error("Database init failed: %s", e)

    # Load ML model and features
    logger.info("Loading ML model and feature dataset...")
    try:
        from backend.services.model_service import model_service
        model_service.load()
        logger.info(
            "Model loaded: %d applicants, %d features",
            model_service.total_applicants,
            len(model_service.feature_names),
        )
    except Exception as e:
        logger.error("Model load failed: %s", e)

    # Load policy RAG
    logger.info("Loading policy knowledge base...")
    try:
        from backend.services.policy_service import policy_service
        policy_service.load()
        logger.info("Policy knowledge base loaded.")
    except Exception as e:
        logger.warning("Policy load failed (RAG will be unavailable): %s", e)

    # Try loading LLM (non-blocking)
    logger.info("Checking LLM availability...")
    try:
        from backend.services.agent_service import agent_service
        agent_service.load()
        if agent_service.llm_available:
            logger.info("LLM available: Ollama %s", settings.ollama_model)
        else:
            logger.warning(
                "LLM unavailable. Deterministic underwriting still works. "
                "Start Ollama to enable AI assistant."
            )
    except Exception as e:
        logger.warning("LLM check failed: %s", e)

    logger.info("=" * 60)
    logger.info("Backend ready. API docs: http://localhost:8002/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down backend.")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Agentic AI Credit Underwriting System — Research/Educational Prototype\n\n"
        "**DISCLAIMER**: This is a B.Tech research prototype. It does NOT constitute "
        "a real credit decision system. Not for production use. "
        "Regulatory material is for informational context only."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

from backend.routes import (  # noqa: E402
    health, applicants, underwriting, policy, model_info, agent, dashboard
)

app.include_router(health.router)
app.include_router(applicants.router)
app.include_router(underwriting.router)
app.include_router(policy.router)
app.include_router(model_info.router)
app.include_router(agent.router)
app.include_router(dashboard.router)


# Convenience alias for POST /underwrite (no prefix)
from backend.schemas.underwriting import UnderwriteRequest, UnderwritingResult
from backend.services.underwriting_service import run_underwriting
from fastapi import Depends
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.db.models import UnderwritingDecision, AuditLog


@app.post("/underwrite", response_model=UnderwritingResult, tags=["underwriting"])
def underwrite_root(request: UnderwriteRequest, db: Session = Depends(get_db)):
    """POST /underwrite — main underwriting endpoint."""
    from backend.routes.underwriting import underwrite
    return underwrite(request, db)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "RESEARCH PROTOTYPE ONLY. Not for production use.",
    }


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please check server logs."},
    )
