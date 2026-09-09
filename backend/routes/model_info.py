"""Model info and registry routes."""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from backend.services.model_service import model_service

router = APIRouter(prefix="/model", tags=["model"])
logger = logging.getLogger(__name__)


class SetActiveModelRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model or pipeline to set as active default")


@router.get("/info")
def get_model_info(model_id: Optional[str] = Query(None, description="Optional model ID")):
    """
    Return specification and governance metadata for a model or pipeline.
    Defaults to the current active model.
    """
    try:
        return model_service.get_model_info(model_id=model_id)
    except Exception as e:
        logger.error("Model info failed for '%s': %s", model_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model information: {e}")


@router.get("/list")
def list_models():
    """
    List all available models and chained pipelines with current active status.
    """
    try:
        models = model_service.get_available_models()
        return {
            "active_model_id": model_service.active_model_id,
            "models": models,
        }
    except Exception as e:
        logger.error("List models failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list models.")


@router.get("/comparison")
def get_model_comparison():
    """
    Return comparative benchmark performance across all models (ROC-AUC, PR-AUC, Brier score, latency).
    """
    try:
        return model_service.get_model_comparison()
    except Exception as e:
        logger.error("Model comparison failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve model comparisons.")


@router.post("/active")
def set_active_model(req: SetActiveModelRequest):
    """
    Set the default active model or chained pipeline for underwriting evaluations.
    """
    try:
        model_service.set_active_model(req.model_id)
        return {
            "status": "success",
            "active_model_id": model_service.active_model_id,
            "message": f"Active underwriting model switched to '{req.model_id}'.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Set active model failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update active model.")
