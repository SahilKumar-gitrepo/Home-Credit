"""
SQLAlchemy ORM models for the audit trail.

IMPORTANT:
These tables store application state and audit records only.
The full Home Credit dataset stays in Parquet/DuckDB.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, DateTime, Integer, Text, JSON, Boolean
)

from backend.db.database import Base


def _now():
    return datetime.now(timezone.utc)


def _uuid():
    return str(uuid.uuid4())


class UnderwritingDecision(Base):
    """Audit record for each underwriting request."""
    __tablename__ = "underwriting_decisions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, index=True, default=_uuid)
    applicant_id = Column(Integer, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=_now, nullable=False)

    # Model output
    probability_of_default = Column(Float, nullable=False)
    risk_tier = Column(String(20), nullable=False)

    # Decision
    decision = Column(String(20), nullable=False)
    policy_reason = Column(Text, nullable=True)

    # Versions
    model_version = Column(String(100), nullable=True)
    policy_version = Column(String(100), nullable=True)

    # SHAP factors (stored as JSON)
    risk_increasing_factors = Column(JSON, nullable=True)
    risk_reducing_factors = Column(JSON, nullable=True)

    # Agent
    agent_assessment = Column(Text, nullable=True)
    llm_used = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "applicant_id": self.applicant_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "probability_of_default": self.probability_of_default,
            "risk_tier": self.risk_tier,
            "decision": self.decision,
            "policy_reason": self.policy_reason,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
        }


class AuditLog(Base):
    """General audit log for API actions."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=_now, nullable=False)
    action = Column(String(100), nullable=False)
    applicant_id = Column(Integer, nullable=True)
    request_id = Column(String(36), nullable=True)
    status = Column(String(20), nullable=True)
    detail = Column(Text, nullable=True)


class DashboardStats(Base):
    """Cached dashboard statistics (rebuilt on each request)."""
    __tablename__ = "dashboard_stats_cache"

    id = Column(Integer, primary_key=True)
    updated_at = Column(DateTime(timezone=True), default=_now)
    total_applicants_in_dataset = Column(Integer, default=0)
    total_analyzed = Column(Integer, default=0)
    approve_count = Column(Integer, default=0)
    refer_count = Column(Integer, default=0)
    decline_count = Column(Integer, default=0)
    average_pd = Column(Float, default=0.0)
    high_risk_percentage = Column(Float, default=0.0)
