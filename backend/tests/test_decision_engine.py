"""Tests for the decision engine logic (unit tests)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Make sure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.services.underwriting_service import get_risk_tier, make_decision
from backend.config import settings


# ============================================================
# Risk Tier Tests
# ============================================================

def test_risk_tier_low():
    assert get_risk_tier(0.01) == "LOW"
    assert get_risk_tier(0.049) == "LOW"


def test_risk_tier_moderate():
    assert get_risk_tier(0.05) == "MODERATE"
    assert get_risk_tier(0.099) == "MODERATE"


def test_risk_tier_high():
    assert get_risk_tier(0.10) == "HIGH"
    assert get_risk_tier(0.199) == "HIGH"


def test_risk_tier_very_high():
    assert get_risk_tier(0.20) == "VERY_HIGH"
    assert get_risk_tier(0.99) == "VERY_HIGH"


# ============================================================
# Decision Logic Tests
# ============================================================

def _make_applicant(income, cc_util=None, credit_to_income=None):
    """Helper: create a minimal applicant DataFrame for decision testing."""
    data = {
        "SK_ID_CURR": [999999],
        "AMT_INCOME_TOTAL": [income],
        "AMT_CREDIT": [100000.0],
        "credit_to_income": [credit_to_income if credit_to_income is not None else np.nan],
        "max_cc_utilization": [cc_util if cc_util is not None else np.nan],
    }
    return pd.DataFrame(data)


def test_decision_high_pd_decline():
    """PD >= 0.20 should always DECLINE."""
    applicant = _make_applicant(income=200000)
    decision, reason = make_decision(applicant, pd_value=0.20)
    assert decision == "DECLINE"


def test_decision_missing_income_refer():
    """Missing income should REFER."""
    applicant = _make_applicant(income=np.nan)
    decision, reason = make_decision(applicant, pd_value=0.05)
    assert decision == "REFER"


def test_decision_zero_income_refer():
    """Zero or negative income should REFER."""
    applicant = _make_applicant(income=0)
    decision, reason = make_decision(applicant, pd_value=0.05)
    assert decision == "REFER"


def test_decision_low_income_refer():
    """Income below minimum should REFER."""
    applicant = _make_applicant(income=settings.min_income - 1)
    decision, reason = make_decision(applicant, pd_value=0.05)
    assert decision == "REFER"


def test_decision_high_credit_to_income_refer():
    """High credit-to-income should REFER."""
    applicant = _make_applicant(income=200000, credit_to_income=15.0)
    decision, reason = make_decision(applicant, pd_value=0.05)
    assert decision == "REFER"


def test_decision_high_cc_utilization_refer():
    """High credit-card utilization should REFER."""
    applicant = _make_applicant(income=200000, cc_util=2.0)
    decision, reason = make_decision(applicant, pd_value=0.05)
    assert decision == "REFER"


def test_decision_approve():
    """PD below 0.10 with good income should APPROVE."""
    applicant = _make_applicant(income=200000)
    decision, reason = make_decision(applicant, pd_value=0.099)
    assert decision == "APPROVE"


def test_decision_moderate_pd_refer():
    """PD from 0.10 to below 0.20 should REFER."""
    applicant = _make_applicant(income=200000)
    decision, reason = make_decision(applicant, pd_value=0.10)
    assert decision == "REFER"
