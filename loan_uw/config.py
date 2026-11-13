"""All business rules live here so they can be changed (and explained) in one place."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS_PATH = ROOT / "data" / "prior_records.json"
DEFAULT_LOG_PATH = ROOT / "logs" / "run_log.jsonl"
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class PolicyConfig:
    foir_cap: float = 0.40              # (new EMI + existing EMIs) must be <= 40% of net income
    default_annual_rate_pct: float = 10.5
    min_tenure_months: int = 6
    max_tenure_months: int = 84         # longest tenure the bank offers on this product
    amount_step: int = 1_000            # counter-offer amounts are rounded DOWN to this step
    min_offer_amount: int = 10_000      # below this a counter-offer is not worth making
    hard_min_credit_score: int = 600    # below -> decline
    review_credit_score: int = 650      # 600..649 -> refer to human
    late_payments_review: int = 3       # >= 3 late payments in 12 months -> refer


POLICY = PolicyConfig()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


def inr(x: float) -> str:
    return f"Rs {x:,.2f}"
