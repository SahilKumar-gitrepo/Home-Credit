"""Policy routes."""
import logging
from fastapi import APIRouter, HTTPException

from backend.schemas.policy import PolicySearchRequest, PolicySearchResponse
from backend.services.policy_service import policy_service

router = APIRouter(prefix="/policy", tags=["policy"])
logger = logging.getLogger(__name__)


@router.post("/search", response_model=PolicySearchResponse)
def search_policy(request: PolicySearchRequest):
    """
    Search the RBI Guidelines on Digital Lending knowledge base.

    Returns relevant passages with source, authority, page, and content.

    IMPORTANT: Results are for informational/research context only.
    They do NOT constitute legal advice or confirm regulatory compliance.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    try:
        return policy_service.search_with_schema(request.query, k=request.k)
    except Exception as e:
        logger.error("Policy search failed: %s", e)
        raise HTTPException(status_code=500, detail="Policy search failed.")


@router.get("/thresholds")
def get_policy_thresholds():
    """Return active research policy thresholds and risk tier cutoffs."""
    from backend.config import settings
    return {
        "policy_version": settings.policy_version,
        "pd_approve_max": settings.pd_approve_max,
        "pd_decline_min": settings.pd_decline_min,
        "pd_risk_tier_low_max": settings.pd_risk_tier_low_max,
        "pd_risk_tier_moderate_max": settings.pd_risk_tier_moderate_max,
        "pd_risk_tier_high_max": settings.pd_risk_tier_high_max,
        "min_income": settings.min_income,
        "max_credit_to_income": settings.max_credit_to_income,
        "max_cc_utilization": settings.max_cc_utilization,
        "risk_tier_rules": {
            "LOW": f"PD < {settings.pd_risk_tier_low_max:.2%}",
            "MODERATE": f"{settings.pd_risk_tier_low_max:.2%} <= PD < {settings.pd_risk_tier_moderate_max:.2%}",
            "HIGH": f"{settings.pd_risk_tier_moderate_max:.2%} <= PD < {settings.pd_risk_tier_high_max:.2%}",
            "VERY_HIGH": f"PD >= {settings.pd_risk_tier_high_max:.2%}",
        },
        "decision_rules": {
            "DECLINE": f"PD >= {settings.pd_decline_min:.2%}",
            "APPROVE": f"PD < {settings.pd_approve_max:.2%} (and all financial guardrails pass)",
            "REFER": f"Income < ₹{settings.min_income:,.0f}, Credit/Income > {settings.max_credit_to_income:.1f}x, CC Util > {settings.max_cc_utilization:.0%}, or {settings.pd_approve_max:.2%} <= PD < {settings.pd_decline_min:.2%}",
        },
    }
