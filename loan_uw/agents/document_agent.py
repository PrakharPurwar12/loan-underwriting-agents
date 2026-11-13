"""Document Agent - turns raw application text into structured fields.

Tools (plain Python, exposed to Semantic Kernel with @kernel_function):
  1. extract_fields   - regex-based extraction of income, EMIs, amount, tenure, rate, id, name
  2. validate_fields  - lists mandatory fields that are missing (it NEVER guesses a value)
  3. llm_extract_fields - optional Groq extraction; missing or invalid values are not guessed
"""
from __future__ import annotations

import json
import math
import re
from typing import Annotated, Optional

from semantic_kernel.functions import kernel_function

from loan_uw.agents.base import BaseAgent
from loan_uw.config import GROQ_MODEL, POLICY, get_groq_api_key, inr
from loan_uw.memory import SharedMemory
from loan_uw.models import AgentReport

_FIELD_NAMES = ("monthly_net_income", "existing_emis", "requested_amount", "tenure_months",
                "annual_rate_pct", "applicant_id", "applicant_name")
_NUMERIC_FIELDS = ("monthly_net_income", "existing_emis", "requested_amount", "annual_rate_pct")
_AMT = re.compile(r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k)?\b", re.I)
_MULT = {"k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
         "cr": 1e7, "crore": 1e7, "crores": 1e7}
_SEGMENT_END = re.compile(r",\s|;|\n|\.\s+(?=[A-Z])")   # commas inside 5,00,000 are NOT followed by a space


def _to_amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * _MULT.get((m.group(2) or "").lower(), 1)


def _validate_llm_fields(fields: object) -> dict:
    if not isinstance(fields, dict) or set(fields) != set(_FIELD_NAMES):
        raise ValueError("LLM response must be an object with exactly the expected extraction fields")
    for field in _NUMERIC_FIELDS:
        value = fields[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"LLM response field {field} must be a number or null")
        if value is not None and (not math.isfinite(value) or
                                  (field in ("monthly_net_income", "requested_amount") and value <= 0) or
                                  (field in ("existing_emis", "annual_rate_pct") and value < 0)):
            raise ValueError(f"LLM response field {field} is outside the valid range")
    tenure = fields["tenure_months"]
    if tenure is not None and (isinstance(tenure, bool) or not isinstance(tenure, int) or tenure <= 0):
        raise ValueError("LLM response field tenure_months must be a positive integer or null")
    for field in ("applicant_id", "applicant_name"):
        if fields[field] is not None and not isinstance(fields[field], str):
            raise ValueError(f"LLM response field {field} must be a string or null")
    return fields


def _amount_after(text: str, keyword_pattern: str) -> tuple[Optional[float], Optional[tuple[int, int]]]:
    """Find `keyword`, look only inside the same clause, return the first amount (and its span)."""
    for kw in re.finditer(keyword_pattern, text, re.I):
        clause = _SEGMENT_END.split(text[kw.end(): kw.end() + 90])[0]
        amt = _AMT.search(clause)
        if amt:
            return _to_amount(amt), (kw.start(), kw.end() + len(clause))
    return None, None


class DocumentTools:
    @kernel_function(name="extract_fields",
                     description="Extract income, existing EMIs, loan amount, tenure and rate from application text")
    def extract_fields(self, text: Annotated[str, "raw application text"]) -> dict:
        work = text
        income, span = _amount_after(work, r"\b(?:net\s+)?(?:monthly\s+)?(?:income|salary|take[- ]home)\b")
        if span:
            work = work[: span[0]] + " " * (span[1] - span[0]) + work[span[1]:]
        emis, span = _amount_after(work, r"\b(?:existing|current|ongoing|other)\s+(?:loan\s+)?(?:emis?|obligations?|repayments?)\b")
        if span:
            work = work[: span[0]] + " " * (span[1] - span[0]) + work[span[1]:]
        amount, _ = _amount_after(work, r"\b(?:loan(?:\s+amount)?|borrow|requests?|requesting|requested(?:\s+amount)?|need)\b")

        tenure = None
        if m := re.search(r"(\d+)\s*(months?|mos?|years?|yrs?)\b", text, re.I):
            tenure = int(m.group(1)) * (12 if m.group(2).lower().startswith(("y")) else 1)
        rate = None
        if m := re.search(r"(\d+(?:\.\d+)?)\s*%", text):
            rate = float(m.group(1))
        app_id = m.group(1) if (m := re.search(r"\b([A-Z]{2,4}-\d{3,})\b", text)) else None
        name = m.group(1) if (m := re.search(r"(?:Applicant|Name)\s*[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)) else None
        return {"monthly_net_income": income, "existing_emis": emis, "requested_amount": amount,
                "tenure_months": tenure, "annual_rate_pct": rate,
                "applicant_id": app_id, "applicant_name": name}

    @kernel_function(name="llm_extract_fields",
                     description="Extract application fields with Groq as JSON with null for unstated values; never guess income")
    async def llm_extract_fields(self, text: Annotated[str, "raw application text"]) -> dict:
        api_key = get_groq_api_key()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        import httpx

        system_message = (
            "Extract only facts explicitly stated in the application text. Never guess or infer income "
            "or any other field. Return JSON only, with exactly these keys: "
            "monthly_net_income, existing_emis, requested_amount, tenure_months, annual_rate_pct, "
            "applicant_id, applicant_name. Use JSON numbers for numeric values, strings for id/name, "
            "and null for every value not stated."
        )
        request_body = {
            "model": GROQ_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text},
            ],
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_body,
            )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content:
            raise ValueError("LLM response did not contain JSON text")
        try:
            fields = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc
        return _validate_llm_fields(fields)

    @kernel_function(name="validate_fields",
                     description="Return the list of mandatory fields that are missing or non-positive")
    def validate_fields(self, fields: Annotated[dict, "output of extract_fields"]) -> dict:
        missing = [f for f in ("monthly_net_income", "requested_amount", "tenure_months")
                   if not fields.get(f) or fields[f] <= 0]
        return {"missing": missing}


class DocumentAgent(BaseAgent):
    name = "DocumentAgent"
    plugin_name = "document"
    role = "Extracts income, existing EMIs and requested amount/tenure from the application"

    def build_plugin(self):
        return DocumentTools()

    async def run(self, memory: SharedMemory) -> AgentReport:
        text = memory.read(self.name, "raw_text")["raw_text"]
        extractor = "regex"
        fallback_reason = None
        if get_groq_api_key():
            try:
                fields = await self.call_tool(memory, "llm_extract_fields", text=text)
                extractor = "groq"
            except Exception as exc:
                fallback_reason = repr(exc)
                api_key = get_groq_api_key()
                if api_key:
                    fallback_reason = fallback_reason.replace(api_key, "[REDACTED]")
                fields = await self.call_tool(memory, "extract_fields", text=text)
        else:
            fallback_reason = "GROQ_API_KEY is not set"
            fields = await self.call_tool(memory, "extract_fields", text=text)
        memory.log(self.name, "EXTRACTOR_USED", extractor=extractor, fallback_reason=fallback_reason)
        check = await self.call_tool(memory, "validate_fields", fields=fields)

        assumptions = []
        existing = fields["existing_emis"]
        if existing is None:
            existing = 0.0
            assumptions.append("existing EMIs not stated; treated as Rs 0")
        rate = fields["annual_rate_pct"]
        if rate is None:
            rate = POLICY.default_annual_rate_pct
            assumptions.append(f"interest rate not stated; bank default {rate}% used")

        memory.write(self.name,
                     applicant_id=fields["applicant_id"], applicant_name=fields["applicant_name"],
                     monthly_net_income=fields["monthly_net_income"], existing_emis=existing,
                     requested_amount=fields["requested_amount"], tenure_months=fields["tenure_months"],
                     annual_rate_pct=rate, missing_fields=check["missing"], assumptions=assumptions)
        if check["missing"]:
            return AgentReport(agent=self.name, ok=False,
                               summary=f"missing mandatory field(s): {', '.join(check['missing'])}")
        return AgentReport(agent=self.name, ok=True,
                           summary=f"income {inr(fields['monthly_net_income'])}, requested "
                                   f"{inr(fields['requested_amount'])} over {fields['tenure_months']} months")
