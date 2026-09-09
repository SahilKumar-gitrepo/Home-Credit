"""Policy schemas."""
from typing import Optional
from pydantic import BaseModel


class PolicySearchRequest(BaseModel):
    query: str
    k: int = 4


class PolicyEvidence(BaseModel):
    source: str
    authority: str
    title: str
    page: Optional[str] = None
    version_date: Optional[str] = None
    jurisdiction: Optional[str] = None
    content: str
    interpretation_note: str = (
        "This evidence is retrieved from a regulatory document for "
        "informational/research context only. It does not constitute "
        "legal advice or confirm regulatory compliance."
    )


class PolicySearchResponse(BaseModel):
    query: str
    results: list[PolicyEvidence]
    total_results: int
