"""Scenario tests. Run: pytest -v   (deterministic - no API key or network needed)."""
import asyncio
import json

import httpx

import pytest

from loan_uw import finance
from loan_uw.agents.document_agent import DocumentAgent
from loan_uw.config import POLICY
from loan_uw.memory import SharedMemory
from loan_uw.models import ApplicantRecord, Decision
from loan_uw.runlog import AppendOnlyRunLog
from loan_uw.samples import SAMPLES
from loan_uw.service import underwrite


@pytest.fixture
def log(tmp_path):
    return AppendOnlyRunLog(tmp_path / "run.jsonl")


def _mock_groq_response(monkeypatch, content):
    response_requested = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 30.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, headers, json):
            assert url == "https://api.groq.com/openai/v1/chat/completions"
            assert headers["Authorization"] == "Bearer test-key"
            assert json["temperature"] == 0
            assert json["response_format"] == {"type": "json_object"}
            response_requested.append(True)
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return response_requested


def test_scenario_1_approve(log):
    rec = underwrite(SAMPLES["approve"], log)
    assert rec.decision == Decision.SANCTION
    assert rec.emi == finance.reducing_balance_emi(300_000, 11.0, 36)
    assert rec.affordability.fits and rec.affordability.obligation_ratio <= POLICY.foir_cap
    assert "R5" in rec.justification and not rec.counter_offers


def test_scenario_2_decline_on_credit(log):
    rec = underwrite(SAMPLES["decline"], log)
    assert rec.decision == Decision.DECLINE
    assert rec.affordability.fits                      # affordable, yet declined...
    assert rec.credit.has_hard                         # ...because of the credit red flags
    assert not rec.counter_offers                      # no counter-offer for a credit decline


def test_scenario_3_counter_offer(log):
    rec = underwrite(SAMPLES["counter_offer"], log)
    assert rec.decision == Decision.DECLINE and not rec.affordability.fits
    kinds = {o.kind: o for o in rec.counter_offers}
    assert set(kinds) == {"reduce_amount", "extend_tenure"}
    assert kinds["reduce_amount"].amount < rec.requested_amount
    assert kinds["extend_tenure"].amount == rec.requested_amount
    assert kinds["extend_tenure"].tenure_months == POLICY.max_tenure_months
    for o in rec.counter_offers:                       # every offer must truly pass the rule
        emi = finance.reducing_balance_emi(o.amount, rec.annual_rate_pct, o.tenure_months)
        assert finance.obligation_ratio(emi, rec.existing_emis, rec.monthly_net_income) <= POLICY.foir_cap
    # "largest": one more step of Rs 1,000 must break the rule
    emi_up = finance.reducing_balance_emi(kinds["reduce_amount"].amount + POLICY.amount_step,
                                          rec.annual_rate_pct, rec.tenure_months)
    assert finance.obligation_ratio(emi_up, rec.existing_emis, rec.monthly_net_income) > POLICY.foir_cap


def test_missing_income_halts_and_refers(log, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    response_requested = _mock_groq_response(monkeypatch, json.dumps({
        "monthly_net_income": None,
        "existing_emis": None,
        "requested_amount": 200_000,
        "tenure_months": 24,
        "annual_rate_pct": None,
        "applicant_id": "APP-1004",
        "applicant_name": "Rohan Verma",
    }))
    rec = underwrite(SAMPLES["missing_income"], log)
    assert rec.decision == Decision.REFER and rec.status == "halted_missing_data"
    assert rec.monthly_net_income is None and rec.emi is None and rec.affordability is None
    events = [json.loads(l) for l in log.path.read_text().splitlines()]
    assert any(e["kind"] == "HALT" for e in events)
    assert not any(e["agent"] == "EMIAgent" for e in events)   # downstream workers never ran
    assert response_requested
    assert any(e["kind"] == "EXTRACTOR_USED" and e["detail"]["extractor"] == "groq"
               for e in events)


def test_document_agent_uses_regex_when_groq_key_is_unset(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    memory = SharedMemory(ApplicantRecord(application_id="TEST", raw_text=SAMPLES["approve"]))

    asyncio.run(DocumentAgent().run(memory))

    extractor_event = next(e for e in memory.events if e["kind"] == "EXTRACTOR_USED")
    assert extractor_event["detail"]["extractor"] == "regex"
    assert any(e.get("detail", {}).get("tool") == "document.extract_fields"
               for e in memory.events if e["kind"] == "TOOL_CALL")
    assert memory.snapshot().monthly_net_income == 90_000


def test_document_agent_falls_back_on_invalid_groq_json(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    response_requested = _mock_groq_response(monkeypatch, "not valid JSON")

    memory = SharedMemory(ApplicantRecord(application_id="TEST", raw_text=SAMPLES["approve"]))
    asyncio.run(DocumentAgent().run(memory))

    extractor_event = next(e for e in memory.events if e["kind"] == "EXTRACTOR_USED")
    assert extractor_event["detail"]["extractor"] == "regex"
    assert extractor_event["detail"]["fallback_reason"]
    assert response_requested
    assert any(e.get("detail", {}).get("tool") == "document.extract_fields"
               for e in memory.events if e["kind"] == "TOOL_CALL")
    assert memory.snapshot().monthly_net_income == 90_000


def test_credit_review_refers(log):
    assert underwrite(SAMPLES["credit_review"], log).decision == Decision.REFER


def test_emi_formula_known_value():
    assert finance.reducing_balance_emi(100_000, 12, 12) == 8884.88   # textbook value
    assert finance.reducing_balance_emi(120_000, 0, 12) == 10_000.0   # zero-rate edge case


def test_worker_failure_is_referred_not_crashed(log, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")                                 # valid store, but make a tool explode:
    from loan_uw.agents import credit_history_agent as cha
    orig = cha.CreditTools.lookup_prior_record
    cha.CreditTools.lookup_prior_record = lambda self, applicant_id: 1 / 0
    try:
        rec = underwrite(SAMPLES["approve"], log, records_path=bad)
    finally:
        cha.CreditTools.lookup_prior_record = orig
    assert rec.decision == Decision.REFER and rec.status == "failed_referred"


def test_run_log_is_append_only_and_tamper_evident(log):
    underwrite(SAMPLES["approve"], log)
    assert log.verify() == (True, None)
    n = len(log.path.read_text().splitlines())
    underwrite(SAMPLES["decline"], log)                  # appends, never rewrites
    assert len(log.path.read_text().splitlines()) > n and log.verify()[0]
    lines = log.path.read_text().splitlines()
    lines[2] = lines[2].replace("DocumentAgent", "Attacker", 1)
    log.path.write_text("\n".join(lines) + "\n")
    assert log.verify() == (False, 3)


def test_agents_cannot_write_foreign_fields():
    from loan_uw.memory import SharedMemory
    from loan_uw.models import ApplicantRecord
    mem = SharedMemory(ApplicantRecord(application_id="X", raw_text=""))
    with pytest.raises(PermissionError):
        mem.write("EMIAgent", decision=Decision.SANCTION)
