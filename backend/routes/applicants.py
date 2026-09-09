"""Applicant routes."""
import logging
from fastapi import APIRouter, HTTPException

from backend.services.model_service import model_service
from backend.services.underwriting_service import build_applicant_profile
from backend.schemas.applicant import ApplicantProfile

router = APIRouter(prefix="/applicants", tags=["applicants"])
logger = logging.getLogger(__name__)


@router.get("/{applicant_id}", response_model=ApplicantProfile)
def get_applicant(applicant_id: int):
    """
    Retrieve the applicant profile from the Home Credit dataset.
    
    Returns human-readable financial and demographic information.
    """
    if applicant_id < 100000 or applicant_id > 9999999:
        raise HTTPException(status_code=422, detail="Invalid applicant ID format.")

    try:
        applicant = model_service.get_applicant(applicant_id)
        return build_applicant_profile(applicant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error retrieving applicant %d: %s", applicant_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve applicant profile.")


@router.get("/")
def list_sample_applicants(limit: int = 20, offset: int = 0):
    """
    Return a sample of applicant IDs from the feature dataset.
    """
    try:
        df = model_service.features_df
        ids = df["SK_ID_CURR"].iloc[offset:offset + limit].tolist()
        return {
            "total": len(df),
            "offset": offset,
            "limit": limit,
            "applicant_ids": [int(i) for i in ids],
        }
    except Exception as e:
        logger.error("Error listing applicants: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list applicants.")
