"""Applicant schemas."""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ApplicantProfile(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "applicant_id": 370920,
                "income": 135000.0,
                "credit_amount": 450000.0,
            }
        }
    )

    applicant_id: int
    income: Optional[float] = None
    credit_amount: Optional[float] = None
    annuity: Optional[float] = None
    days_birth: Optional[float] = Field(None, description="Age in negative days (Home Credit encoding)")
    days_employed: Optional[float] = Field(None, description="Employment duration in negative days")
    family_size: Optional[float] = None
    children_count: Optional[float] = None
    external_score_1: Optional[float] = None
    external_score_2: Optional[float] = None
    external_score_3: Optional[float] = None
    external_score_mean: Optional[float] = None
    credit_to_income: Optional[float] = None
    annuity_to_credit: Optional[float] = None
    bureau_debt_to_credit: Optional[float] = None
    max_cc_utilization: Optional[float] = None
    prev_application_count: Optional[float] = None
    pos_record_count: Optional[float] = None
    installment_count: Optional[float] = None
    cc_balance_count: Optional[float] = None
    name_income_type: Optional[str] = None
    name_education_type: Optional[str] = None
    name_family_status: Optional[str] = None
    name_housing_type: Optional[str] = None
    occupation_type: Optional[str] = None
    organization_type: Optional[str] = None
