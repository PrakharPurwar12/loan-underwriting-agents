# Multi-Agent Loan Underwriting System — CSE476 CA2 (Q2)

Decides a loan application as **sanction / decline / refer** and reports the EMI, the affordability verdict and a one-paragraph justification. Built on **Semantic Kernel (Python)**; runs offline with deterministic regex extraction, or can optionally use Groq to extract document fields when `GROQ_API_KEY` is set. EMI, affordability, credit rules, decisions and counter-offers always remain deterministic Python.

## 1. Agent roster (one module each, `loan_uw/agents/`)
| Agent | Role | Tools (plain Python, exposed via `@kernel_function`) |
|---|---|---|
| **Document Agent** | Extracts income, existing EMIs, amount, tenure, rate, id | `extract_fields`, `validate_fields`, `llm_extract_fields` (optional Groq) |
| **EMI Agent** | Reducing-balance EMI (no LLM) | `compute_emi`, `amortisation_summary` |
| **Affordability Agent** | Rule: new EMI + existing EMIs ≤ 40% of net income | `compute_obligation_ratio`, `max_affordable_emi`, `check_rule` |
| **Credit-History Agent** | Red flags from the prior-record store | `lookup_prior_record`, `evaluate_red_flags` |
| **Counter-Offer Agent** (bonus) | Passing alternative if affordability fails | `max_principal_for_emi`, `verify_offer` |
| **Underwriting Manager** | Delegates, decides, writes justification | `apply_policy`, `compose_justification` |

## 2. Architecture: manager–worker delegation (`docs/architecture.md`)
The Manager owns control flow: Document → (halt check) → EMI → Affordability → Credit-History → (Counter-Offer only if affordability failed and credit is clean) → decision. Workers never call each other; they only read/write the shared record and report back. **Why it fits:** the steps have real dependencies (EMI needs the extracted amount; affordability needs the EMI), and one accountable agent must own the final decision and its audit trail. Each agent has its **own Kernel** holding only its own plugin, so tool access is isolated.

Decision table (`apply_policy`): R1 missing data → **refer**; R2 hard credit flag (default, bankruptcy, score < 600) → **decline**; R3 review flag (score 600–649, ≥ 3 late payments/12m) → **refer**; R4 affordability fails → **decline** (counter-offers attached); R5 otherwise → **sanction**.

## 3. Tools
All money maths is in `loan_uw/finance.py`: `EMI = P·r·(1+r)ⁿ / ((1+r)ⁿ − 1)`, r = annual rate / 12 / 100 (zero-rate handled). Counter-offer uses the inverse formula. Tools are called through `kernel.invoke(...)` and each call (args + result) is logged.

## 4. Memory
`ApplicantRecord` (Pydantic v2, `validate_assignment=True`) is the shared record. `SharedMemory` enforces **field ownership** (e.g. only EMI Agent writes `emi`; only the Manager writes `decision`) and logs every read/write. The prior-record store is `data/prior_records.json`, loaded in memory by the Credit-History tools.

## 5. Failures handled
1. **Missing income** (or amount/tenure): Manager **halts** before any calculation and returns **refer**; the value is never assumed.
2. **Any worker/tool exception**: caught by the Manager, logged as `FAILURE`, application **referred** to a human (tested).
Also: unstated existing EMIs → treated as Rs 0 and recorded in `assumptions`; unknown applicant → "thin file" info flag only.

## 6. Counter-offer (Group-of-3)
If affordability fails (and credit is clean), **reduce_amount** is the largest amount (rounded down to Rs 1,000) passing at the requested tenure. **extend_tenure** offers the same requested amount at the longest allowed tenure (84 months), only when `verify_offer` confirms it passes the 40% rule. Offers are attached to a *decline* of the original request.

## 7. Run log
`logs/run_log.jsonl`: JSON Lines, opened in append mode, no update/delete API, each line carries the SHA-256 of the previous line (`verify()` detects tampering).

## 8. Run
```
pip install -r requirements.txt
python demo.py            # all scenarios, prints trace   (or: python demo.py counter_offer)
pytest -v                 # tests: approve, decline, counter-offer, missing income, credit review, failure, log, ownership, regex fallback
```
Set `GROQ_API_KEY` (and optionally `GROQ_MODEL`; default `llama-3.1-8b-instant`) in the environment or a local `.env` file to enable optional Groq extraction through Semantic Kernel's OpenAI-compatible connector. The LLM is prompted for JSON only with temperature 0; on a failure, timeout, or invalid response, the Document Agent logs the fallback and uses regex extraction. Without the key, regex extraction is used. Missing income is never guessed, and the Manager still halts and refers applications with missing mandatory data. Copy `.env.example` to `.env` and add your key; `.env` is ignored by Git.
