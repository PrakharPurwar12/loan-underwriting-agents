"""Runnable demo: python demo.py [scenario|all]   (no API key needed - tools are deterministic)."""
import argparse
import asyncio
import sys

from loan_uw.agents.underwriting_manager import UnderwritingManager
from loan_uw.config import DEFAULT_LOG_PATH, inr
from loan_uw.runlog import AppendOnlyRunLog
from loan_uw.samples import SAMPLES
from loan_uw.service import underwrite
from loan_uw.trace import format_event


def show(name: str, text: str, log: AppendOnlyRunLog) -> None:
    print("=" * 110, f"\nSCENARIO: {name}\nINPUT   : {text}\n" + "-" * 110)
    manager = UnderwritingManager(log)
    rec = asyncio.run(manager.process(text))
    for event in manager.last_trace:
        print(format_event(event))
    print("-" * 110, "\nFINAL OUTPUT")
    print(f"  Decision      : {rec.decision.value.upper()}")
    print(f"  Computed EMI  : {inr(rec.emi) if rec.emi is not None else 'n/a (halted before computation)'}")
    if rec.affordability:
        print(f"  Affordability : {'PASS' if rec.affordability.fits else 'FAIL'} "
              f"({rec.affordability.obligation_ratio:.1%} of income vs {rec.affordability.cap:.0%} cap)")
    else:
        print("  Affordability : not evaluated")
    for o in rec.counter_offers:
        print(f"  Counter-offer : {o.kind}: {inr(o.amount)} / {o.tenure_months} months / EMI {inr(o.emi)}")
    print(f"  Justification : {rec.justification}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scenario", nargs="?", default="all", choices=["all", *SAMPLES])
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    args = p.parse_args()
    log = AppendOnlyRunLog(args.log)
    for n, t in SAMPLES.items():
        if args.scenario in ("all", n):
            show(n, t, log)
    ok, bad = log.verify()
    print(f"Run log: {log.path}  | hash-chain intact: {ok}" + ("" if ok else f" (first bad line {bad})"))
    sys.exit(0 if ok else 1)
