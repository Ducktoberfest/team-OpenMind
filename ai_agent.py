"""Üzv 3: Ollama (Gemma) ilə əlaqə. Model yalnız JSON əməliyyat seçir və nəticəni izah edir.
Model adını dəyişmək üçün:  set OLLAMA_MODEL=gemma3:4b   (sponsorun dediyi Gemma 4 tag-ı ilə əvəz edin)
"""
from __future__ import annotations

import json
import os
import re

import requests

from data_analyzer import ALLOWED_OPS, validate_op

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")  # TODO: Gemma 4 tag-ı ilə əvəz et

PARSE_SYSTEM = """You convert a user's data question into ONE JSON operation. Return ONLY JSON, no text.
Allowed "op": sum, mean, min, max, count, group_sum, group_mean, group_count, trend.
Fields: op, metric (numeric column), group_by (column, for group_* ops), top_n (int, optional),
ascending (bool, optional), freq ("M" for monthly, for trend), filters (optional list of
{{"column":..,"operator":"==|!=|>|<|>=|<=|contains","value":..}}).
Use ONLY these columns: {columns}
Examples:
Q: Hansı məhsul daha çox gəlir gətirib? -> {{"op":"group_sum","metric":"Revenue","group_by":"Product","top_n":3}}
Q: Which region sold the most? -> {{"op":"group_sum","metric":"Revenue","group_by":"Region","top_n":1}}
Q: Hansı ayda satışlar yüksək olub? -> {{"op":"trend","metric":"Revenue","freq":"M"}}
Q: Bakıda ümumi gəlir nə qədərdir? -> {{"op":"sum","metric":"Revenue","filters":[{{"column":"Region","operator":"==","value":"Baku"}}]}}
Q: Neçə sətir var? -> {{"op":"count"}}"""

EXPLAIN_SYSTEM = """You are a business data analyst. Explain the RESULT to the user in the same language as the
question (Azerbaijani if the question is Azerbaijani). Use ONLY numbers present in RESULT. NEVER invent
numbers. Be short (3-5 sentences), then give 1-2 practical recommendations."""


def _chat(system: str, user: str, json_mode: bool = False, timeout: int = 90) -> str:
    payload = {"model": MODEL, "stream": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "options": {"temperature": 0.1}}
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except Exception:
        return False


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def _fallback_parse(question: str, columns: list, numeric_cols: list) -> dict:
    """Model işləməsə, sadə açar sözlərlə ehtiyat parser (demo sığortası)."""
    q = question.lower()
    metric = next((c for c in numeric_cols if "revenue" in c.lower() or "gəlir" in c.lower()),
                  numeric_cols[0] if numeric_cols else None)
    if any(w in q for w in ("ay", "month", "trend", "zaman")):
        return {"op": "trend", "metric": metric, "freq": "M"}
    for c in columns:
        if c.lower() in q or (c.lower() == "product" and "məhsul" in q) or (c.lower() == "region" and "region" in q):
            if c != metric:
                return {"op": "group_sum", "metric": metric, "group_by": c, "top_n": 5}
    if "say" in q or "count" in q or "neçə" in q:
        return {"op": "count"}
    return {"op": "sum", "metric": metric}


def parse_question(question: str, columns: list, numeric_cols: list | None = None, retries: int = 2) -> dict:
    """sual -> doğrulanmış JSON əməliyyat. Uğursuz olarsa ehtiyat parser."""
    numeric_cols = numeric_cols or []
    sys_prompt = PARSE_SYSTEM.format(columns=", ".join(columns))
    for _ in range(retries):
        try:
            op = _extract_json(_chat(sys_prompt, question, json_mode=True))
            return validate_op(op, columns)
        except Exception:
            continue
    return validate_op(_fallback_parse(question, columns, numeric_cols), columns)


def explain_result(question: str, result: dict) -> str:
    payload = f"QUESTION: {question}\nRESULT: {json.dumps(result, ensure_ascii=False, default=str)}"
    try:
        return _chat(EXPLAIN_SYSTEM, payload).strip()
    except Exception:
        return _template_explain(result)


def _template_explain(result: dict) -> str:
    """Model yoxdursa, rəqəmləri sadəcə təqdim edir (uydurma yoxdur)."""
    if "value" in result:
        return f"Nəticə: {result['value']} ({result['op']}, {result.get('metric')})."
    t = result.get("table") or []
    if t:
        first = t[0]
        return f"Birinci nəticə: {first}. Cəmi {len(t)} sətir cədvəldə göstərilib. (AI izahı əlçatan deyil.)"
    return "Nəticə tapılmadı."
