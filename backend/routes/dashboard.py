"""Dashboard stats route."""
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.db.database import get_db
from backend.db.models import UnderwritingDecision
from backend.services.model_service import model_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
logger = logging.getLogger(__name__)


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Return aggregate statistics for the dashboard."""
    try:
        total_in_dataset = model_service.total_applicants
    except Exception:
        total_in_dataset = 0

    # Aggregate from audit DB
    total_analyzed = db.query(func.count(UnderwritingDecision.id)).scalar() or 0

    approve_count = db.query(func.count(UnderwritingDecision.id)).filter(
        UnderwritingDecision.decision == "APPROVE"
    ).scalar() or 0

    refer_count = db.query(func.count(UnderwritingDecision.id)).filter(
        UnderwritingDecision.decision == "REFER"
    ).scalar() or 0

    decline_count = db.query(func.count(UnderwritingDecision.id)).filter(
        UnderwritingDecision.decision == "DECLINE"
    ).scalar() or 0

    avg_pd = db.query(func.avg(UnderwritingDecision.probability_of_default)).scalar() or 0.0

    high_risk_count = db.query(func.count(UnderwritingDecision.id)).filter(
        UnderwritingDecision.risk_tier.in_(["HIGH", "VERY_HIGH"])
    ).scalar() or 0

    high_risk_pct = (high_risk_count / total_analyzed * 100) if total_analyzed > 0 else 0.0

    # Risk tier distribution
    tier_dist_rows = (
        db.query(UnderwritingDecision.risk_tier, func.count(UnderwritingDecision.id))
        .group_by(UnderwritingDecision.risk_tier)
        .all()
    )
    tier_distribution = {t: c for t, c in tier_dist_rows}

    return {
        "total_applicants_in_dataset": total_in_dataset,
        "total_analyzed": total_analyzed,
        "approve_count": approve_count,
        "refer_count": refer_count,
        "decline_count": decline_count,
        "average_pd": round(float(avg_pd), 4) if avg_pd else 0.0,
        "high_risk_percentage": round(float(high_risk_pct), 2),
        "risk_tier_distribution": tier_distribution,
    }
