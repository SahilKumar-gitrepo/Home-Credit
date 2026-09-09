"""Underwriting routes."""
import logging
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.schemas.underwriting import (
    UnderwriteRequest, UnderwritingResult, UnderwritingListItem
)
from backend.services.underwriting_service import run_underwriting
from backend.db.database import get_db
from backend.db.models import UnderwritingDecision, AuditLog
from backend.config import settings

router = APIRouter(prefix="/underwriting", tags=["underwriting"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=UnderwritingResult)
def underwrite(request: UnderwriteRequest, db: Session = Depends(get_db)):
    """
    Run the complete credit underwriting workflow for an applicant.

    1. Retrieves applicant features
    2. Predicts Probability of Default (XGBoost)
    3. Assigns risk tier
    4. Applies research underwriting policy
    5. Generates SHAP explanation
    6. Retrieves RBI regulatory evidence
    7. Saves audit record
    8. Returns structured assessment

    DISCLAIMER: RESEARCH PROTOTYPE ONLY. Not a real credit decision.
    """
    try:
        result = run_underwriting(request.applicant_id, model_id=request.model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Underwriting failed for %d: %s", request.applicant_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Underwriting workflow failed.")

    # Save audit record
    try:
        record = UnderwritingDecision(
            request_id=result.request_id,
            applicant_id=result.applicant_id,
            timestamp=result.timestamp,
            probability_of_default=result.probability_of_default,
            risk_tier=result.risk_tier,
            decision=result.decision,
            policy_reason=result.policy_reason,
            model_version=result.model_information.model_version,
            policy_version=settings.policy_version,
            risk_increasing_factors=[
                {"feature": f.feature, "value": f.value, "shap": f.shap}
                for f in result.risk_increasing_factors
            ],
            risk_reducing_factors=[
                {"feature": f.feature, "value": f.value, "shap": f.shap}
                for f in result.risk_reducing_factors
            ],
        )
        db.add(record)

        log = AuditLog(
            action="underwrite",
            applicant_id=result.applicant_id,
            request_id=result.request_id,
            status=result.decision,
            detail=result.policy_reason,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning("Failed to save audit record: %s", e)
        db.rollback()

    return result


# Keep POST /underwrite (no trailing slash) as alias
@router.post("/underwrite", response_model=UnderwritingResult, include_in_schema=False)
def underwrite_alias(request: UnderwriteRequest, db: Session = Depends(get_db)):
    return underwrite(request, db)


@router.get("/{applicant_id}", response_model=UnderwritingResult)
def get_underwriting(applicant_id: int, db: Session = Depends(get_db)):
    """
    Get the most recent underwriting result for an applicant.
    If no cached result exists, runs the underwriting workflow.
    """
    # Check if we have a recent result in DB
    existing = (
        db.query(UnderwritingDecision)
        .filter(UnderwritingDecision.applicant_id == applicant_id)
        .order_by(UnderwritingDecision.timestamp.desc())
        .first()
    )

    # Always re-run for fresh results (the workflow is fast after model load)
    try:
        result = run_underwriting(applicant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Underwriting GET failed for %d: %s", applicant_id, e)
        raise HTTPException(status_code=500, detail="Underwriting workflow failed.")

    return result


@router.get("/history/list", response_model=list[UnderwritingListItem])
def list_underwriting_history(limit: int = 50, db: Session = Depends(get_db)):
    """List recent underwriting decisions from the audit trail."""
    records = (
        db.query(UnderwritingDecision)
        .order_by(UnderwritingDecision.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        UnderwritingListItem(
            request_id=r.request_id,
            applicant_id=r.applicant_id,
            timestamp=r.timestamp,
            probability_of_default=r.probability_of_default,
            risk_tier=r.risk_tier,
            decision=r.decision,
        )
        for r in records
    ]
