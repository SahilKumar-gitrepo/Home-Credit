"""Health check route."""
import platform
from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Agentic AI Credit Underwriting System",
        "python": platform.python_version(),
    }
