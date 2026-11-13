"""EMI Agent - computes the reducing-balance EMI with a deterministic formula (no LLM).

Tools:
  1. compute_emi          - EMI = P*r*(1+r)^n / ((1+r)^n - 1)
  2. amortisation_summary - total payable and total interest over the tenure
"""
from __future__ import annotations

from typing import Annotated

from semantic_kernel.functions import kernel_function

from loan_uw import finance
from loan_uw.agents.base import BaseAgent
from loan_uw.config import inr
from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport


class EMITools:
    @kernel_function(name="compute_emi", description="Reducing-balance EMI for principal, annual rate %, months")
    def compute_emi(self, principal: Annotated[float, "loan amount"],
                    annual_rate_pct: Annotated[float, "annual interest rate in percent"],
                    months: Annotated[int, "tenure in months"]) -> dict:
        return {"emi": finance.reducing_balance_emi(principal, annual_rate_pct, months)}

    @kernel_function(name="amortisation_summary", description="Total payable and total interest for an EMI")
    def amortisation_summary(self, principal: Annotated[float, "loan amount"],
                             emi: Annotated[float, "monthly EMI"],
                             months: Annotated[int, "tenure in months"]) -> dict:
        total = round(emi * months, 2)
        return {"total_payable": total, "total_interest": round(total - principal, 2)}


class EMIAgent(BaseAgent):
    name = "EMIAgent"
    plugin_name = "emi"
    role = "Computes the reducing-balance EMI deterministically"

    def build_plugin(self):
        return EMITools()

    async def run(self, memory: SharedMemory) -> AgentReport:
        r = memory.read(self.name, "requested_amount", "annual_rate_pct", "tenure_months")
        emi = (await self.call_tool(memory, "compute_emi", principal=r["requested_amount"],
                                    annual_rate_pct=r["annual_rate_pct"], months=r["tenure_months"]))["emi"]
        s = await self.call_tool(memory, "amortisation_summary", principal=r["requested_amount"],
                                 emi=emi, months=r["tenure_months"])
        memory.write(self.name, emi=emi, total_payable=s["total_payable"], total_interest=s["total_interest"])
        return AgentReport(agent=self.name, ok=True, summary=f"EMI = {inr(emi)}")
