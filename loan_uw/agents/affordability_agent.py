"""Affordability Agent - applies the stated income rule.

Rule: (new EMI + existing EMIs) must be <= 40% of net monthly income.
Tools:
  1. compute_obligation_ratio - (new EMI + existing EMIs) / net income
  2. max_affordable_emi       - the largest new EMI the rule allows
  3. check_rule               - ratio <= cap ?
"""
from __future__ import annotations

from typing import Annotated

from semantic_kernel.functions import kernel_function

from loan_uw import finance
from loan_uw.agents.base import BaseAgent
from loan_uw.config import POLICY
from loan_uw.memory import SharedMemory
from loan_uw.models import AffordabilityResult, AgentReport


class AffordabilityTools:
    @kernel_function(name="compute_obligation_ratio", description="(new EMI + existing EMIs) / net income")
    def compute_obligation_ratio(self, emi: Annotated[float, "new EMI"],
                                 existing_emis: Annotated[float, "existing monthly EMIs"],
                                 net_income: Annotated[float, "net monthly income"]) -> dict:
        return {"ratio": finance.obligation_ratio(emi, existing_emis, net_income)}

    @kernel_function(name="max_affordable_emi", description="Largest new EMI allowed by the cap")
    def max_affordable_emi(self, net_income: Annotated[float, "net monthly income"],
                           existing_emis: Annotated[float, "existing monthly EMIs"],
                           cap: Annotated[float, "cap as a fraction, e.g. 0.40"]) -> dict:
        return {"max_emi": finance.max_affordable_emi(net_income, existing_emis, cap)}

    @kernel_function(name="check_rule", description="True if ratio is within the cap")
    def check_rule(self, ratio: Annotated[float, "obligation ratio"],
                   cap: Annotated[float, "cap as a fraction"]) -> dict:
        return {"fits": ratio <= cap}


class AffordabilityAgent(BaseAgent):
    name = "AffordabilityAgent"
    plugin_name = "affordability"
    role = "Applies the 40%-of-net-income rule"

    def build_plugin(self):
        return AffordabilityTools()

    async def run(self, memory: SharedMemory) -> AgentReport:
        r = memory.read(self.name, "emi", "existing_emis", "monthly_net_income")
        cap = POLICY.foir_cap
        ratio = (await self.call_tool(memory, "compute_obligation_ratio", emi=r["emi"],
                                      existing_emis=r["existing_emis"], net_income=r["monthly_net_income"]))["ratio"]
        max_emi = (await self.call_tool(memory, "max_affordable_emi", net_income=r["monthly_net_income"],
                                        existing_emis=r["existing_emis"], cap=cap))["max_emi"]
        fits = (await self.call_tool(memory, "check_rule", ratio=ratio, cap=cap))["fits"]
        result = AffordabilityResult(fits=fits, obligation_ratio=ratio, cap=cap, max_affordable_emi=max_emi,
                                     rule=f"EMI + existing EMIs <= {cap:.0%} of net monthly income")
        memory.write(self.name, affordability=result)
        return AgentReport(agent=self.name, ok=True,
                           summary=f"{'PASS' if fits else 'FAIL'}: obligations {ratio:.1%} vs cap {cap:.0%}")
