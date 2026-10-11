from __future__ import annotations

import json
import os
import re

import requests

from data_analyzer import GROUP_OPS, validate_op

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # ilk yükləmə / ümumi limit (saniyə)
PARSE_TIMEOUT = int(os.getenv("OLLAMA_PARSE_TIMEOUT", "30"))  # sual -> JSON (qısa: yavaşdırsa ehtiyat rejim)
EXPLAIN_TIMEOUT = int(os.getenv("OLLAMA_EXPLAIN_TIMEOUT", "60"))  # nəticə -> izah

PARSE_SYSTEM = """You convert a user's data question into ONE JSON operation. Return ONLY JSON, no text.
Allowed "op": sum, mean, min, max, count, group_sum, group_mean, group_count, trend.
Fields: op, metric (numeric column), group_by (column, for group_* ops), top_n (int, optional),
ascending (bool, optional), date_col (for trend), freq ("M" for monthly, for trend), filters (optional list of
{{"column":..,"operator":"==|!=|>|<|>=|<=|contains","value":..}}).
Do NOT include fields that are not needed (no null values).
Dataset columns (name: type):
{columns_info}
Default metric: {metric}. Default date column: {date}.
Use ONLY the column names listed above, spelled exactly.
For filter values use the exact spelling from the dataset values if they are listed.
Examples (adapt to the real columns):
Q: Which {dim} has the highest {metric}? -> {{"op":"group_sum","metric":"{metric}","group_by":"{dim}","top_n":3}}
Q: Ən çox {metric} hansı {dim}-dədir? -> {{"op":"group_sum","metric":"{metric}","group_by":"{dim}","top_n":1}}
Q: How did {metric} change by month? -> {{"op":"trend","metric":"{metric}","date_col":"{date}","freq":"M"}}
Q: What is the total {metric}? -> {{"op":"sum","metric":"{metric}"}}
Q: How many rows are there? -> {{"op":"count"}}"""

EXPLAIN_SYSTEM = """You are a business data analyst. Explain the RESULT to the user in the same language as the
question (Azerbaijani if the question is Azerbaijani). Use ONLY numbers present in RESULT. NEVER invent
numbers and NEVER claim a cause that the data does not show. Maximum 3 short sentences, then ONE short
practical recommendation phrased as something to check or try."""

_parse_cache: dict = {}
_explain_cache: dict = {}


# ---------------------------------------------------------------- Ollama
def _chat(system: str, user: str, json_mode: bool = False, timeout: int | None = None,
          max_tokens: int = 1024) -> str:
    payload = {
        "model": MODEL, "stream": False, "keep_alive": "2h", "think": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 2048},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout or TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_available() -> bool:
    """Ollama serveri işləyir VƏ MODEL yüklənib?"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if not r.ok:
            return False
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return MODEL in names or f"{MODEL}:latest" in names
    except Exception:
        return False


def warm_up() -> None:
    """Modeli əvvəlcədən yaddaşa yükləyir (demo başlamazdan əvvəl çağır)."""
    try:
        _chat("Reply with OK.", "OK", max_tokens=5)
    except Exception:
        pass


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def normalize_op(op):
    """Modelin tez-tez etdiyi kiçik səhvləri düzəldir: null sahələri və əməliyyata
    aid olmayan açarları silir, top_n-i int-ə çevirir. Sonra validate_op yoxlayır."""
    if not isinstance(op, dict):
        return op
    kind = op.get("op")
    if kind in GROUP_OPS:
        allowed = {"op", "metric", "group_by", "top_n", "ascending", "filters"}
    elif kind == "trend":
        allowed = {"op", "metric", "date_col", "freq", "filters"}
    else:
        allowed = {"op", "metric", "filters"}
    clean = {k: v for k, v in op.items() if k in allowed and v is not None and v != []}
    if "top_n" in clean:
        try:
            clean["top_n"] = int(clean["top_n"])
        except (TypeError, ValueError):
            clean.pop("top_n")
    return clean


# ---------------------------------------------------------------- sual -> JSON
def _fallback_parse(question: str, columns: list, ctx: dict) -> dict:
    """Model işləməyəndə sütun adlarından asılı olmayan sadə ehtiyat parser."""
    q = question.lower()
    metric = ctx.get("metric") or next(iter(ctx.get("numeric", [])), None)
    cats = ctx.get("categorical") or [c for c in columns if c != metric]
    if re.search(r"\b(ay|ayda|aylar|aylıq)\b|month|trend|zaman", q) and metric and ctx.get("date"):
        return {"op": "trend", "metric": metric, "date_col": ctx["date"], "freq": "M"}
    named = next((c for c in columns if c.lower() in q and c != metric), None)
    if named and metric:
        return {"op": "group_sum", "metric": metric, "group_by": named, "top_n": 5}
    if metric and cats and re.search(r"hansı|which|ən çox|most|top|highest|best", q):
        return {"op": "group_sum", "metric": metric, "group_by": cats[0], "top_n": 5}
    if not metric or re.search(r"\bsay\b|count|neçə|how many", q):
        return {"op": "count"}
    return {"op": "sum", "metric": metric}


def parse_question(question: str, columns: list, ctx: dict | None = None, retries: int = 2) -> dict:
    """sual -> doğrulanmış JSON əməliyyat. Model uğursuz olarsa ehtiyat parser.
    ctx (hamısı ixtiyari): {"dtypes": {col: tip}, "numeric": [...], "categorical": [...],
    "metric": str, "date": str, "values": {col: [nümunə dəyərlər]}}"""
    ctx = ctx or {}
    key = (question.strip().lower(), tuple(columns), ctx.get("metric"), ctx.get("date"))
    if key in _parse_cache:
        return dict(_parse_cache[key])

    cats = ctx.get("categorical") or []
    metric = ctx.get("metric") or next(iter(ctx.get("numeric", [])), "VALUE")
    values = ctx.get("values") or {}
    info = "\n".join(
        f"- {c}: {ctx.get('dtypes', {}).get(c, '?')}"
        + (f" (values: {', '.join(map(str, values[c][:12]))})" if c in values else "")
        for c in columns)
    sys_prompt = PARSE_SYSTEM.format(columns_info=info, metric=metric, date=ctx.get("date") or "none",
                                     dim=cats[0] if cats else columns[0])

    op = None
    for _ in range(retries):
        try:
            raw = _extract_json(_chat(sys_prompt, question, json_mode=True,
                                      max_tokens=300, timeout=PARSE_TIMEOUT))
            op = validate_op(normalize_op(raw), columns)
            break
        except (requests.Timeout, requests.ConnectionError):
            break  # model əlçatmazdır/çox yavaşdır: təkrar gözləmə, dərhal ehtiyat rejim
        except requests.HTTPError:
            break  # məs. model yüklənməyib (404): təkrar cəhd mənasızdır
        except Exception:
            continue  # kəsilmiş/boş/yanlış JSON: bir də cəhd et
    if op is None:
        op = validate_op(_fallback_parse(question, columns, ctx), columns)
    _parse_cache[key] = dict(op)
    return op


# ---------------------------------------------------------------- nəticə -> izah
def explain_result(question: str, result: dict) -> str:
    payload = f"QUESTION: {question}\nRESULT: {json.dumps(result, ensure_ascii=False, default=str)}"
    if payload in _explain_cache:
        return _explain_cache[payload]
    try:
        text = _chat(EXPLAIN_SYSTEM, payload, max_tokens=200, timeout=EXPLAIN_TIMEOUT).strip()
    except Exception:
        return _template_explain(result)
    text = text or _template_explain(result)
    _explain_cache[payload] = text
    return text


def _template_explain(result: dict) -> str:
    """Model yoxdursa və ya boş cavab veribsə: yalnız hesablanmış rəqəmlər, uydurma yoxdur."""
    if "value" in result:
        return f"Nəticə: {result['value']} ({result['op']}, {result.get('metric')})."
    t = result.get("table") or []
    if t:
        return f"Birinci nəticə: {t[0]}. Cədvəldə cəmi {len(t)} sətir var. (AI izahı əlçatan deyil.)"
    return "Nəticə tapılmadı."
