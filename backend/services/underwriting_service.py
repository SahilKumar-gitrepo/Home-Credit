"""
Underwriting Service.

Orchestrates:
1. Applicant retrieval
2. PD prediction
3. Risk tier assignment
4. Deterministic decision policy
5. SHAP explanation
6. Audit record creation
"""

import sys
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from backend.config import settings
from backend.services.model_service import model_service
from backend.schemas.underwriting import (
    UnderwritingResult, SHAPFactor, ModelInformation
)
from backend.schemas.applicant import ApplicantProfile

logger = logging.getLogger(__name__)

# Add src/ to sys.path
_src_path = str(settings.model_path.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# In-memory SHAP cache keyed by applicant_id
_shap_cache: dict = {}


# ============================================================
# RISK TIER
# ============================================================

def get_risk_tier(pd_value: float) -> str:
    if pd_value < settings.pd_risk_tier_low_max:
        return "LOW"
    elif pd_value < settings.pd_risk_tier_moderate_max:
        return "MODERATE"
    elif pd_value < settings.pd_risk_tier_high_max:
        return "HIGH"
    else:
        return "VERY_HIGH"


# ============================================================
# DECISION ENGINE
# (Mirrors src/underwriting_agent.py but uses settings for thresholds)
# ============================================================

def make_decision(applicant: pd.DataFrame, pd_value: float) -> tuple[str, str]:
    """
    Research underwriting policy.
    NOT an official bank policy.
    Returns (decision, reason).
    """

    # Rule 1: Very high PD -> DECLINE
    if pd_value >= settings.pd_decline_min:
        return (
            "DECLINE",
            "Predicted probability of default is at or above the research decline threshold",
        )

    # Rule 2: Income validation
    income = applicant["AMT_INCOME_TOTAL"].iloc[0]

    if pd.isna(income):
        return "REFER", "Applicant income information is missing"

    if income <= 0:
        return "REFER", "Applicant income is invalid or non-positive"

    if income < settings.min_income:
        return "REFER", "Applicant income is below the research policy minimum threshold"

    # Rule 3: Credit-to-income ratio
    credit_to_income = applicant.get("credit_to_income", pd.Series([np.nan])).iloc[0]
    if pd.notna(credit_to_income) and credit_to_income > settings.max_credit_to_income:
        return "REFER", "Requested credit is high relative to reported income"

    # Rule 4: Credit-card utilization
    cc_util = applicant.get("max_cc_utilization", pd.Series([np.nan])).iloc[0]
    if pd.notna(cc_util) and cc_util > settings.max_cc_utilization:
        return "REFER", "Credit-card utilization exceeds the research policy threshold"

    # Rule 5: PD-based approval
    if pd_value < settings.pd_approve_max:
        return (
            "APPROVE",
            "Predicted probability of default is below the research approval threshold",
        )

    return "REFER", "Predicted probability of default falls within the manual-review range"


# ============================================================
# SHAP EXPLANATION
# ============================================================

def calculate_shap(
    applicant: pd.DataFrame, top_n: int = 5, model_id: Optional[str] = None
) -> tuple[list, list]:
    """
    Calculate SHAP or feature attribution explanation for one applicant.
    Supports tree models (XGBoost, LightGBM) and linear models (Logistic Scorecard).
    """
    mid = model_id or model_service.active_model_id
    if mid in ("chained_ensemble", "cascade_hurdle"):
        # For chained pipelines, derive attribution from the primary tree model
        mid = "xgboost" if "xgboost" in model_service._models else "lightgbm"

    X = model_service.prepare_model_input(applicant, mid)

    if mid == "logistic" and "logistic" in model_service._models:
        # Linear feature attribution: coeff * standardized_x
        pipe = model_service._models["logistic"]["model"]
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        classifier = pipe.named_steps["classifier"]
        x_imp = imputer.transform(X)
        x_scaled = scaler.transform(x_imp)
        shap_row = classifier.coef_[0] * x_scaled[0]
    else:
        import shap
        model_obj = model_service._models.get(mid, {}).get("model", model_service.model)
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_row = np.asarray(shap_values[-1])[0]
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                shap_row = shap_array[0, :, -1]
            elif shap_array.ndim == 2:
                shap_row = shap_array[0]
            else:
                shap_row = shap_array

    # Build explanation list
    from feature_descriptions import describe_feature

    feature_names = list(X.columns)
    explanation = []

    for i, feat in enumerate(feature_names):
        shap_val = float(shap_row[i])

        if feat in applicant.columns:
            orig_val = applicant[feat].iloc[0]
        else:
            orig_val = X.iloc[0, i]

        desc = describe_feature(feat, orig_val, shap_val)
        explanation.append({
            "feature": feat,
            "label": desc.get("label", feat),
            "category": desc.get("category", "General"),
            "value": str(orig_val),
            "shap": shap_val,
            "effect": desc["effect"],
            "description": desc["description"],
        })

    explanation.sort(key=lambda x: abs(x["shap"]), reverse=True)

    increasing = [x for x in explanation if x["shap"] > 0][:top_n]
    reducing = [x for x in explanation if x["shap"] < 0][:top_n]

    return increasing, reducing


# ============================================================
# APPLICANT PROFILE BUILDER
# ============================================================

def build_applicant_profile(applicant: pd.DataFrame) -> ApplicantProfile:
    """Build a human-readable applicant profile from the feature row."""
    row = applicant.iloc[0]

    def safe_get(col, default=None):
        val = row.get(col, default)
        if pd.isna(val):
            return None
        return val

    return ApplicantProfile(
        applicant_id=int(row["SK_ID_CURR"]),
        income=safe_get("AMT_INCOME_TOTAL"),
        credit_amount=safe_get("AMT_CREDIT"),
        annuity=safe_get("AMT_ANNUITY"),
        days_birth=safe_get("DAYS_BIRTH"),
        days_employed=safe_get("DAYS_EMPLOYED"),
        family_size=safe_get("CNT_FAM_MEMBERS"),
        children_count=safe_get("CNT_CHILDREN"),
        external_score_1=safe_get("EXT_SOURCE_1"),
        external_score_2=safe_get("EXT_SOURCE_2"),
        external_score_3=safe_get("EXT_SOURCE_3"),
        external_score_mean=safe_get("ext_source_mean"),
        credit_to_income=safe_get("credit_to_income"),
        annuity_to_credit=safe_get("annuity_to_credit"),
        bureau_debt_to_credit=safe_get("bureau_debt_to_credit_ratio"),
        max_cc_utilization=safe_get("max_cc_utilization"),
        prev_application_count=safe_get("prev_app_count"),
        pos_record_count=safe_get("pos_record_count"),
        installment_count=safe_get("installment_count"),
        cc_balance_count=safe_get("cc_balance_count"),
        name_income_type=safe_get("NAME_INCOME_TYPE"),
        name_education_type=safe_get("NAME_EDUCATION_TYPE"),
        name_family_status=safe_get("NAME_FAMILY_STATUS"),
        name_housing_type=safe_get("NAME_HOUSING_TYPE"),
        occupation_type=safe_get("OCCUPATION_TYPE"),
        organization_type=safe_get("ORGANIZATION_TYPE"),
    )


# ============================================================
# MAIN UNDERWRITING FUNCTION
# ============================================================

def run_underwriting(
    applicant_id: int, include_agent: bool = False, model_id: Optional[str] = None
) -> UnderwritingResult:
    """
    Complete underwriting workflow for one applicant.

    1. Retrieve applicant features
    2. Calculate PD using requested or active model
    3. Assign risk tier
    4. Apply deterministic decision policy
    5. Generate SHAP explanation
    6. Return structured result
    """
    # Step 1: Get applicant
    applicant = model_service.get_applicant(applicant_id)

    # Step 2: PD prediction with pipeline diagnostics
    pd_value, details, effective_model_name = model_service.predict_pd_detailed(
        applicant, model_id=model_id
    )

    # Step 3: Risk tier
    risk_tier = get_risk_tier(pd_value)

    # Step 4: Decision
    decision, policy_reason = make_decision(applicant, pd_value)

    # Step 5: SHAP (cache keyed by (applicant_id, model_id))
    cache_key = f"{applicant_id}_{model_id or model_service.active_model_id}"
    if cache_key in _shap_cache:
        increasing_raw, reducing_raw = _shap_cache[cache_key]
        logger.debug("SHAP cache hit for %s", cache_key)
    else:
        try:
            increasing_raw, reducing_raw = calculate_shap(applicant, model_id=model_id)
            _shap_cache[cache_key] = (increasing_raw, reducing_raw)
        except Exception as e:
            logger.warning("SHAP calculation failed for %s: %s", cache_key, e)
            increasing_raw, reducing_raw = [], []

    # Convert to Pydantic models
    def to_shap_factors(raw_list: list) -> list[SHAPFactor]:
        result = []
        for item in raw_list:
            result.append(SHAPFactor(
                feature=item["feature"],
                label=item.get("label", item["feature"]),
                category=item.get("category", "General"),
                value=str(item["value"]),
                shap=float(item["shap"]),
                effect=item.get("effect", ""),
                description=item.get("description", "Model feature."),
            ))
        return result

    # Step 6: Policy evidence
    from backend.services.policy_service import policy_service
    evidence_query = (
        f"borrower consent data privacy transparency digital lending "
        f"applicant protection credit risk {decision.lower()}"
    )
    regulatory_evidence = policy_service.search(evidence_query, k=3)

    return UnderwritingResult(
        request_id=str(uuid.uuid4()),
        applicant_id=int(applicant_id),
        timestamp=datetime.now(timezone.utc),
        probability_of_default=float(pd_value),
        risk_tier=risk_tier,
        decision=decision,
        policy_reason=policy_reason,
        risk_increasing_factors=to_shap_factors(increasing_raw),
        risk_reducing_factors=to_shap_factors(reducing_raw),
        regulatory_evidence=regulatory_evidence,
        model_information=ModelInformation(
            model_id=model_id or model_service.active_model_id,
            model_type=effective_model_name,
            model_version=f"{model_id or model_service.active_model_id}_v1",
            feature_count=len(model_service.feature_names),
            pipeline_details=details,
        ),
        policy_version=settings.policy_version,
        policy_thresholds={
            "policy_version": settings.policy_version,
            "pd_approve_max": settings.pd_approve_max,
            "pd_decline_min": settings.pd_decline_min,
            "pd_risk_tier_low_max": settings.pd_risk_tier_low_max,
            "pd_risk_tier_moderate_max": settings.pd_risk_tier_moderate_max,
            "pd_risk_tier_high_max": settings.pd_risk_tier_high_max,
            "min_income": settings.min_income,
            "max_credit_to_income": settings.max_credit_to_income,
            "max_cc_utilization": settings.max_cc_utilization,
        },
    )
