"""Typed data contracts. ApplicantRecord is the single shared state every agent reads/writes."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Decision(str, Enum):
    SANCTION = "sanction"
    DECLINE = "decline"
    REFER = "refer"


class RedFlag(BaseModel):
    code: str
    severity: Literal["hard", "review", "info"]
    reason: str


class CreditResult(BaseModel):
    record_found: bool
    credit_score: Optional[int] = None
    flags: list[RedFlag] = Field(default_factory=list)

    @property
    def has_hard(self) -> bool:
        return any(f.severity == "hard" for f in self.flags)

    @property
    def has_review(self) -> bool:
        return any(f.severity == "review" for f in self.flags)


class AffordabilityResult(BaseModel):
    fits: bool
    obligation_ratio: float
    cap: float
    max_affordable_emi: float
    rule: str


class CounterOffer(BaseModel):
    kind: Literal["reduce_amount", "extend_tenure"]
    amount: float
    tenure_months: int
    emi: float
    obligation_ratio: float


class ApplicantRecord(BaseModel):
    """Shared applicant record. validate_assignment=True => every write is type-checked."""
    model_config = ConfigDict(validate_assignment=True)

    application_id: str
    raw_text: str
    # --- written by Document Agent
    applicant_id: Optional[str] = None
    applicant_name: Optional[str] = None
    monthly_net_income: Optional[float] = Field(default=None, ge=0)
    existing_emis: float = Field(default=0.0, ge=0)
    requested_amount: Optional[float] = Field(default=None, gt=0)
    tenure_months: Optional[int] = Field(default=None, gt=0)
    annual_rate_pct: float = Field(default=10.5, ge=0)
    missing_fields: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # --- written by specialist workers
    emi: Optional[float] = None
    total_payable: Optional[float] = None
    total_interest: Optional[float] = None
    affordability: Optional[AffordabilityResult] = None
    credit: Optional[CreditResult] = None
    counter_offers: list[CounterOffer] = Field(default_factory=list)
    # --- written by Underwriting Manager only
    decision: Optional[Decision] = None
    justification: str = ""
    status: str = "received"


class AgentReport(BaseModel):
    """What a worker hands back to the manager after finishing its delegated task."""
    agent: str
    ok: bool
    summary: str
