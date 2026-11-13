"""Pure, deterministic finance maths. No LLM, no I/O. Every tool in the system that
touches money delegates to one of these functions, so the numbers are reproducible."""
from __future__ import annotations

import math
from typing import Optional


def reducing_balance_emi(principal: float, annual_rate_pct: float, months: int) -> float:
    """EMI = P * r * (1+r)^n / ((1+r)^n - 1), r = annual_rate / 12 / 100."""
    if principal <= 0 or months <= 0 or annual_rate_pct < 0:
        raise ValueError("principal and months must be > 0 and rate >= 0")
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return round(principal / months, 2)
    growth = (1 + r) ** months
    return round(principal * r * growth / (growth - 1), 2)


def obligation_ratio(new_emi: float, existing_emis: float, net_income: float) -> float:
    """Fixed-obligation-to-income ratio (FOIR) = (new EMI + existing EMIs) / net income."""
    if net_income <= 0:
        raise ValueError("net income must be > 0")
    return round((new_emi + existing_emis) / net_income, 4)


def max_affordable_emi(net_income: float, existing_emis: float, cap: float) -> float:
    """Largest new EMI that keeps total obligations within cap * income (never negative)."""
    return round(max(0.0, cap * net_income - existing_emis), 2)


def max_principal_for_emi(max_emi: float, annual_rate_pct: float, months: int) -> float:
    """Inverse of the EMI formula: P = EMI * ((1+r)^n - 1) / (r * (1+r)^n)."""
    if max_emi <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return max_emi * months
    growth = (1 + r) ** months
    return max_emi * (growth - 1) / (r * growth)


def round_down(value: float, step: int) -> float:
    return float(math.floor(value / step) * step)


def min_tenure_for_amount(principal: float, annual_rate_pct: float, max_emi: float,
                          start_months: int, max_months: int) -> Optional[int]:
    """Shortest tenure in [start, max] whose EMI fits under max_emi (None if none does)."""
    for n in range(start_months, max_months + 1):
        if reducing_balance_emi(principal, annual_rate_pct, n) <= max_emi:
            return n
    return None
