"""Underwriting Manager - the MANAGER in the manager-worker architecture.

It owns the control flow: it delegates to each specialist, reads the shared record, and is the
only agent allowed to write `decision`, `justification` and `status`.

Tools:
  1. apply_policy        - deterministic decision table (rules R1-R5)
  2. compose_justification - builds the one-paragraph, number-grounded justification

Failures handled:
  * income (or amount/tenure) missing  -> HALT and REFER; income is never assumed
  * any worker/tool raises an exception -> caught, logged, application REFERRED to a human
"""
from __future__ import annotations

import uuid
from typing import Annotated, Optional

from semantic_kernel.functions import kernel_function

from loan_uw.agents.affordability_agent import AffordabilityAgent
from loan_uw.agents.base import BaseAgent
from loan_uw.agents.counter_offer_agent import CounterOfferAgent
from loan_uw.agents.credit_history_agent import CreditHistoryAgent
from loan_uw.agents.document_agent import DocumentAgent
from loan_uw.agents.emi_agent import EMIAgent
from loan_uw.config import DEFAULT_RECORDS_PATH, POLICY, inr
from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport, ApplicantRecord, Decision
from loan_uw.runlog import AppendOnlyRunLog

RULES = {
    "R1": "mandatory data missing -> REFER (never assume)",
    "R2": "hard credit red flag -> DECLINE",
    "R3": "credit review flag -> REFER to human underwriter",
    "R4": "affordability rule failed -> DECLINE (counter-offer attached if one exists)",
    "R5": "all checks passed -> SANCTION",
}


class ManagerTools:
    @kernel_function(name="apply_policy", description="Deterministic decision table over worker signals")
    def apply_policy(self, missing_count: Annotated[int, "number of missing mandatory fields"],
                     credit_hard: Annotated[bool, "hard credit flag present"],
                     credit_review: Annotated[bool, "review credit flag present"],
                     affordable: Annotated[bool, "affordability rule passed"]) -> dict:
        if missing_count > 0:
            rule, decision = "R1", Decision.REFER
        elif credit_hard:
            rule, decision = "R2", Decision.DECLINE
        elif credit_review:
            rule, decision = "R3", Decision.REFER
        elif not affordable:
            rule, decision = "R4", Decision.DECLINE
        else:
            rule, decision = "R5", Decision.SANCTION
        return {"decision": decision.value, "rule": rule, "rule_text": RULES[rule]}

    @kernel_function(name="compose_justification", description="Write the one-paragraph justification")
    def compose_justification(self, decision: Annotated[str, "sanction/decline/refer"],
                              rule: Annotated[str, "rule id"],
                              summary: Annotated[str, "facts: amount, tenure, EMI, ratio"],
                              credit_text: Annotated[str, "credit-history finding"],
                              extra: Annotated[str, "counter-offer or missing-data note"]) -> dict:
        text = f"Decision: {decision.upper()} (rule {rule}: {RULES[rule]}). {summary} {credit_text} {extra}"
        return {"justification": " ".join(text.split())}


class UnderwritingManager(BaseAgent):
    name = "UnderwritingManager"
    plugin_name = "manager"
    role = "Delegates to specialists, then decides sanction / decline / refer"

    def __init__(self, runlog: Optional[AppendOnlyRunLog] = None, records_path=DEFAULT_RECORDS_PATH):
        super().__init__()
        self.runlog = runlog
        self.document = DocumentAgent()
        self.emi = EMIAgent()
        self.affordability = AffordabilityAgent()
        self.credit = CreditHistoryAgent(records_path)
        self.counter_offer = CounterOfferAgent()
        self.last_trace: list[dict] = []   # events of the most recent run (for demos/tests)

    def build_plugin(self):
        return ManagerTools()

    # ---- delegation ----------------------------------------------------
    async def _delegate(self, memory: SharedMemory, worker: BaseAgent, task: str) -> AgentReport:
        memory.log(self.name, "DELEGATE", to=worker.name, task=task)
        report = await worker.run(memory)
        memory.log(worker.name, "REPORT", ok=report.ok, summary=report.summary)
        return report

    # ---- public entry point -------------------------------------------
    async def process(self, raw_text: str, application_id: Optional[str] = None) -> ApplicantRecord:
        app_id = application_id or f"LN-{uuid.uuid4().hex[:8].upper()}"
        memory = SharedMemory(ApplicantRecord(application_id=app_id, raw_text=raw_text), self.runlog)
        memory.log(self.name, "RUN_START", raw_text=raw_text)
        try:
            await self._pipeline(memory)
        except Exception as exc:  # failure handling #2: never crash, never guess - refer to a human
            memory.log(self.name, "FAILURE", error=repr(exc))
            memory.write(self.name, decision=Decision.REFER, status="failed_referred",
                         justification=f"Decision: REFER. A processing error occurred ({type(exc).__name__}); "
                                       "the application is sent to a human underwriter without any assumption.")
        self.last_trace = memory.events
        final = memory.snapshot()
        memory.log(self.name, "RUN_END", decision=final.decision, status=final.status)
        return final

    # ---- control flow -------------------------------------------------
    async def _pipeline(self, memory: SharedMemory) -> None:
        await self._delegate(memory, self.document, "extract income, existing EMIs, amount, tenure")
        missing = memory.read(self.name, "missing_fields")["missing_fields"]
        if missing:                                              # failure handled #1: HALT
            memory.log(self.name, "HALT", reason=f"missing mandatory field(s): {missing}; not assuming values")
            await self._finalise(memory, status="halted_missing_data")
            return

        await self._delegate(memory, self.emi, "compute reducing-balance EMI")
        await self._delegate(memory, self.affordability, "apply the 40% income rule")
        await self._delegate(memory, self.credit, "check prior-record store for red flags")

        r = memory.read(self.name, "affordability", "credit")
        if not r["affordability"].fits and not (r["credit"].has_hard or r["credit"].has_review):
            await self._delegate(memory, self.counter_offer, "affordability failed: find a passing counter-offer")
        await self._finalise(memory, status="decided")

    async def _finalise(self, memory: SharedMemory, status: str) -> None:
        r = memory.read(self.name, "applicant_id", "missing_fields", "requested_amount", "tenure_months",
                        "annual_rate_pct", "emi", "existing_emis", "monthly_net_income",
                        "affordability", "credit", "counter_offers")
        aff, credit = r["affordability"], r["credit"]
        verdict = await self.call_tool(
            memory, "apply_policy", missing_count=len(r["missing_fields"]),
            credit_hard=bool(credit and credit.has_hard), credit_review=bool(credit and credit.has_review),
            affordable=bool(aff and aff.fits))

        if r["missing_fields"]:
            summary = "The application cannot be assessed because the following mandatory input is missing: " \
                      f"{', '.join(r['missing_fields'])}; it was not assumed or estimated."
            credit_text, extra = "No EMI, affordability or credit checks were run.", \
                "A human underwriter must obtain the missing information."
        else:
            summary = (f"Requested {inr(r['requested_amount'])} over {r['tenure_months']} months at "
                       f"{r['annual_rate_pct']}% gives a reducing-balance EMI of {inr(r['emi'])}; with existing EMIs of "
                       f"{inr(r['existing_emis'])}, total obligations are {aff.obligation_ratio:.1%} of net monthly income "
                       f"{inr(r['monthly_net_income'])} against the {aff.cap:.0%} cap "
                       f"({'within' if aff.fits else 'above'} the limit).")
            credit_text = "Credit history: " + ("; ".join(f.reason for f in credit.flags) or "no adverse findings") + "."
            extra = ""
            if r["counter_offers"]:
                extra = "Counter-offer(s) that pass the rule: " + " | ".join(
                    f"{o.kind.replace('_', ' ')}: {inr(o.amount)} over {o.tenure_months} months, EMI {inr(o.emi)} "
                    f"({o.obligation_ratio:.1%})" for o in r["counter_offers"]) + "."
            elif verdict["rule"] == "R4":
                extra = "No affordable counter-offer exists within product limits."
        j = await self.call_tool(memory, "compose_justification", decision=verdict["decision"], rule=verdict["rule"],
                                 summary=summary, credit_text=credit_text, extra=extra)
        memory.write(self.name, decision=Decision(verdict["decision"]), justification=j["justification"], status=status)
        memory.log(self.name, "DECISION", decision=verdict["decision"], rule=verdict["rule"])
