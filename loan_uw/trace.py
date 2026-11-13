"""Human-readable rendering of the multi-agent trace."""
import json


def _short(obj, limit=150) -> str:
    text = json.dumps(obj, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_event(e: dict) -> str:
    d = e["detail"]
    kind = e["kind"]
    if kind == "TOOL_CALL":
        body = f"{d['tool']}({_short(d['args'], 70)}) -> {_short(d['result'], 90)}"
    elif kind == "DELEGATE":
        body = f"-> {d['to']}: {d['task']}"
    elif kind == "REPORT":
        body = f"{'OK' if d['ok'] else 'FAIL'} | {d['summary']}"
    elif kind == "WRITE":
        body = "fields=" + _short(d["fields"], 110)
    elif kind == "READ":
        body = "fields=" + ", ".join(d["fields"])
    else:
        body = _short(d, 130)
    return f"[{e['seq']:>3}] {e['agent']:<20} {kind:<10} {body}"
