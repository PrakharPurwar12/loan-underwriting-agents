"""Credit-History Agent - checks the in-memory prior-record store for red flags.

Tools:
  1. lookup_prior_record - fetch the applicant's prior record from the store (by applicant id)
  2. evaluate_red_flags  - rule table: bankruptcy / default / low score (hard), repeated late
                           payments / borderline score (review), thin file (info)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from semantic_kernel.functions import kernel_function

from loan_uw.agents.base import BaseAgent
from loan_uw.config import DEFAULT_RECORDS_PATH, POLICY
from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport, CreditResult, RedFlag


class CreditTools:
    def __init__(self, store_path: Path):
        self._store: dict[str, dict] = json.loads(Path(store_path).read_text(encoding="utf-8"))

    @kernel_function(name="lookup_prior_record", description="Look up an applicant's prior credit record")
    def lookup_prior_record(self, applicant_id: Annotated[str, "applicant id, e.g. APP-1001"]) -> dict:
        rec = self._store.get(applicant_id)
        return {"found": rec is not None, "record": rec or {}}

    @kernel_function(name="evaluate_red_flags", description="Apply red-flag rules to a prior record")
    def evaluate_red_flags(self, found: Annotated[bool, "was a record found"],
                           record: Annotated[dict, "the prior record"]) -> dict:
        if not found:
            return {"flags": [{"code": "NO_HISTORY", "severity": "info",
                               "reason": "no prior record on file (thin file); no adverse history found"}]}
        flags, score = [], record.get("credit_score")
        if record.get("bankruptcy"):
            flags.append({"code": "BANKRUPTCY", "severity": "hard", "reason": "prior bankruptcy on record"})
        if record.get("defaults", 0) > 0:
            flags.append({"code": "DEFAULT", "severity": "hard",
                          "reason": f"{record['defaults']} prior default(s) on record"})
        if score is not None and score < POLICY.hard_min_credit_score:
            flags.append({"code": "LOW_SCORE", "severity": "hard",
                          "reason": f"credit score {score} below minimum {POLICY.hard_min_credit_score}"})
        elif score is not None and score < POLICY.review_credit_score:
            flags.append({"code": "BORDERLINE_SCORE", "severity": "review",
                          "reason": f"credit score {score} is borderline (< {POLICY.review_credit_score})"})
        late = record.get("late_payments_12m", 0)
        if late >= POLICY.late_payments_review:
            flags.append({"code": "LATE_PAYMENTS", "severity": "review",
                          "reason": f"{late} late payments in the last 12 months"})
        return {"flags": flags}


class CreditHistoryAgent(BaseAgent):
    name = "CreditHistoryAgent"
    plugin_name = "credit"
    role = "Checks the prior-record store for red flags"

    def __init__(self, store_path: Path = DEFAULT_RECORDS_PATH):
        self._store_path = store_path
        super().__init__()

    def build_plugin(self):
        return CreditTools(self._store_path)

    async def run(self, memory: SharedMemory) -> AgentReport:
        applicant_id = memory.read(self.name, "applicant_id")["applicant_id"]
        lookup = await self.call_tool(memory, "lookup_prior_record", applicant_id=applicant_id or "UNKNOWN")
        flags = (await self.call_tool(memory, "evaluate_red_flags", found=lookup["found"],
                                      record=lookup["record"]))["flags"]
        result = CreditResult(record_found=lookup["found"], credit_score=lookup["record"].get("credit_score"),
                              flags=[RedFlag(**f) for f in flags])
        memory.write(self.name, credit=result)
        worst = "HARD" if result.has_hard else "REVIEW" if result.has_review else "CLEAR"
        return AgentReport(agent=self.name, ok=True, summary=f"{worst}: {', '.join(f.code for f in result.flags) or 'no flags'}")
