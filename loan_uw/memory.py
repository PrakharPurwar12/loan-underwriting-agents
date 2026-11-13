"""Shared applicant memory. Wraps the Pydantic record and enforces two rules:
  1. field ownership  - only the owning agent may write a field (clear data-flow story), and
  2. every read/write/tool call is logged to the in-memory trace and the append-only run log.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic_core import to_jsonable_python

from loan_uw.models import ApplicantRecord
from loan_uw.runlog import AppendOnlyRunLog

FIELD_OWNERS: dict[str, str] = {
    **{f: "DocumentAgent" for f in (
        "applicant_id", "applicant_name", "monthly_net_income", "existing_emis",
        "requested_amount", "tenure_months", "annual_rate_pct", "missing_fields", "assumptions")},
    **{f: "EMIAgent" for f in ("emi", "total_payable", "total_interest")},
    "affordability": "AffordabilityAgent",
    "credit": "CreditHistoryAgent",
    "counter_offers": "CounterOfferAgent",
    **{f: "UnderwritingManager" for f in ("decision", "justification", "status")},
}


class SharedMemory:
    def __init__(self, record: ApplicantRecord, runlog: Optional[AppendOnlyRunLog] = None):
        self._record = record
        self._runlog = runlog
        self.events: list[dict] = []

    # ---- trace ---------------------------------------------------------
    def log(self, agent: str, kind: str, **detail: Any) -> dict:
        event = {
            "seq": len(self.events) + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "application_id": self._record.application_id,
            "agent": agent,
            "kind": kind,
            "detail": to_jsonable_python(detail, fallback=str),
        }
        self.events.append(event)
        if self._runlog is not None:
            self._runlog.append(event)
        return event

    # ---- read / write --------------------------------------------------
    def read(self, agent: str, *fields: str) -> dict[str, Any]:
        values = {f: getattr(self._record, f) for f in fields}
        self.log(agent, "READ", fields=list(fields))
        return values

    def write(self, agent: str, **fields: Any) -> None:
        for name in fields:
            owner = FIELD_OWNERS.get(name)
            if owner != agent:
                raise PermissionError(f"{agent} may not write '{name}' (owner: {owner})")
        for name, value in fields.items():
            setattr(self._record, name, value)   # validate_assignment type-checks here
        self.log(agent, "WRITE", fields=fields)

    def snapshot(self) -> ApplicantRecord:
        return self._record.model_copy(deep=True)
