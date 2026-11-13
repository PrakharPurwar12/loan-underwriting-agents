"""Counter-Offer Agent (Group-of-3 bonus) - when the requested amount fails affordability,
propose an alternative that DOES pass.

Offers produced:
  A. reduce_amount : the largest amount (rounded down) that passes at the REQUESTED tenure
  B. extend_tenure : the same amount at the longest allowed tenure, if it passes
Tools:
  1. max_principal_for_emi - inverse EMI formula (largest principal for the allowed EMI)
  2. verify_offer          - independently re-computes EMI and ratio for an offer before it is made
"""
from __future__ import annotations

from typing import Annotated

from semantic_kernel.functions import kernel_function

from loan_uw import finance
from loan_uw.agents.base import BaseAgent
from loan_uw.config import POLICY, inr
from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport, CounterOffer


class CounterOfferTools:
    @kernel_function(name="max_principal_for_emi", description="Largest principal (rounded down) whose EMI fits")
    def max_principal_for_emi(self, max_emi: Annotated[float, "allowed EMI"],
                              annual_rate_pct: Annotated[float, "annual rate %"],
                              months: Annotated[int, "tenure in months"]) -> dict:
        raw = finance.max_principal_for_emi(max_emi, annual_rate_pct, months)
        return {"amount": finance.round_down(raw, POLICY.amount_step)}

    @kernel_function(name="verify_offer", description="Re-compute EMI and obligation ratio for an offer")
    def verify_offer(self, principal: Annotated[float, "offer amount"],
                     annual_rate_pct: Annotated[float, "annual rate %"],
                     months: Annotated[int, "offer tenure"],
                     existing_emis: Annotated[float, "existing EMIs"],
                     net_income: Annotated[float, "net monthly income"]) -> dict:
        emi = finance.reducing_balance_emi(principal, annual_rate_pct, months)
        ratio = finance.obligation_ratio(emi, existing_emis, net_income)
        return {"emi": emi, "ratio": ratio, "passes": ratio <= POLICY.foir_cap}


class CounterOfferAgent(BaseAgent):
    name = "CounterOfferAgent"
    plugin_name = "counter_offer"
    role = "Proposes the largest passing amount or the longest allowed passing tenure"

    def build_plugin(self):
        return CounterOfferTools()

    async def run(self, memory: SharedMemory) -> AgentReport:
        r = memory.read(self.name, "requested_amount", "tenure_months", "annual_rate_pct",
                        "existing_emis", "monthly_net_income", "affordability")
        max_emi = r["affordability"].max_affordable_emi
        rate, months, amount = r["annual_rate_pct"], r["tenure_months"], r["requested_amount"]
        offers: list[CounterOffer] = []

        if max_emi > 0:
            a = (await self.call_tool(memory, "max_principal_for_emi", max_emi=max_emi,
                                      annual_rate_pct=rate, months=months))["amount"]
            if POLICY.min_offer_amount <= a < amount:
                v = await self.call_tool(memory, "verify_offer", principal=a, annual_rate_pct=rate, months=months,
                                         existing_emis=r["existing_emis"], net_income=r["monthly_net_income"])
                if v["passes"]:
                    offers.append(CounterOffer(kind="reduce_amount", amount=a, tenure_months=months,
                                               emi=v["emi"], obligation_ratio=v["ratio"]))
            longest_tenure = POLICY.max_tenure_months
            if months < longest_tenure:
                v = await self.call_tool(memory, "verify_offer", principal=amount, annual_rate_pct=rate,
                                         months=longest_tenure, existing_emis=r["existing_emis"],
                                         net_income=r["monthly_net_income"])
                if v["passes"]:
                    offers.append(CounterOffer(kind="extend_tenure", amount=amount,
                                               tenure_months=longest_tenure, emi=v["emi"],
                                               obligation_ratio=v["ratio"]))
        memory.write(self.name, counter_offers=offers)
        if not offers:
            return AgentReport(agent=self.name, ok=True, summary="no passing counter-offer exists")
        return AgentReport(agent=self.name, ok=True, summary="; ".join(
            f"{o.kind}: {inr(o.amount)} / {o.tenure_months}m / EMI {inr(o.emi)}" for o in offers))
