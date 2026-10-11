"""Üzv 1: faylın oxunması, keyfiyyət yoxlaması, təhlükəsiz təmizləmə."""
from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
NON_NEGATIVE_HINTS = ("quantity", "revenue", "price", "amount", "say", "gelir")


def load_file(file) -> pd.DataFrame:
    """CSV/Excel oxuyur. `file` yol (str) və ya Streamlit UploadedFile ola bilər."""
    name = (file if isinstance(file, str) else getattr(file, "name", "")).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".csv") or not name:
        return pd.read_csv(file)
    raise ValueError("Yalnız CSV və Excel (.xlsx/.xls) dəstəklənir.")


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _text_cols(df):
    return [c for c in df.columns if _is_text(df[c])]


def is_date_col(name: str) -> bool:
    n = name.lower()
    return "date" in n or "tarix" in n


def _numeric_like(s: pd.Series) -> bool:
    nn = s.dropna().astype(str).str.strip()
    if nn.empty:
        return False
    conv = pd.to_numeric(nn.str.replace(",", ".", regex=False), errors="coerce")
    return conv.notna().mean() >= 0.9


def _case_groups(s: pd.Series) -> dict:
    """lower-strip açarına görə fərqli yazılışları qruplaşdırır (yalnız >1 variant olanlar)."""
    vals = s.dropna().astype(str)
    norm = vals.str.strip().str.replace(r"\s+", " ", regex=True)
    groups: dict = {}
    for raw, key in zip(norm, norm.str.lower()):
        groups.setdefault(key, set()).add(raw)
    return {k: v for k, v in groups.items() if len(v) > 1}


def quality_report(df: pd.DataFrame) -> dict:
    issues = []

    def add(type_, column, count, desc, safe):
        issues.append({"type": type_, "column": column, "count": int(count),
                       "description": desc, "safe_fix": safe})

    # boş xanalar
    for c in df.columns:
        m = int(df[c].isna().sum())
        if m:
            add("missing", c, m, f"'{c}' sütununda {m} boş xana (avtomatik doldurulmur, istifadəçi qərar verir)", False)
    # dublikatlar
    d = int(df.duplicated().sum())
    if d:
        add("duplicates", "-", d, f"{d} təkrarlanan sətir", True)
    # mətn problemləri
    for c in _text_cols(df):
        if is_date_col(c):
            continue
        s = df[c].dropna().astype(str)
        ws = int((s != s.str.strip().str.replace(r"\s+", " ", regex=True)).sum())
        if ws:
            add("whitespace", c, ws, f"'{c}': {ws} dəyərdə artıq boşluq", True)
        g = _case_groups(df[c])
        if g:
            ex = "; ".join("/".join(sorted(v)) for v in list(g.values())[:2])
            add("spelling", c, len(g), f"'{c}': fərqli yazılışlar ({ex})", True)
        if _numeric_like(df[c]) and not is_date_col(c):
            add("dtype", c, len(s), f"'{c}' rəqəm kimi görünür, amma mətn tipindədir", True)
    # tarix formatları
    for c in df.columns:
        if is_date_col(c):
            s = df[c].dropna()
            if _is_text(s):
                s = s.astype(str).str.strip()
                non_iso = int((~s.str.match(ISO_RE)).sum())
                parsed = pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")
                bad = int(parsed.isna().sum())
                if non_iso:
                    add("date_format", c, non_iso, f"'{c}': {non_iso} tarix ISO (YYYY-MM-DD) formatında deyil", True)
                if bad:
                    add("date_invalid", c, bad, f"'{c}': {bad} tarix oxuna bilmir", False)
    # şübhəli rəqəmlər
    for c in df.select_dtypes("number").columns:
        s = df[c].dropna()
        if any(h in c.lower() for h in NON_NEGATIVE_HINTS):
            neg = int((s < 0).sum())
            if neg:
                add("negative", c, neg, f"'{c}': {neg} mənfi dəyər (şübhəli)", False)
        if len(s) >= 20:
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            if iqr > 0:
                out = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
                if out:
                    add("outlier", c, out, f"'{c}': {out} həddindən artıq kənar dəyər (outlier)", False)

    penalty = sum(min(i["count"] / max(len(df), 1), 0.2) * 100 for i in issues)
    return {"rows": len(df), "cols": len(df.columns), "issues": issues,
            "score": max(0, int(100 - penalty))}


def apply_safe_fixes(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Yalnız təhlükəsiz düzəlişlər. Boş dəyərlər və outlier-lərə TOXUNMUR."""
    out, log = df.copy(), []
    for c in _text_cols(out):
        if is_date_col(c):
            continue
        before = out[c].copy()
        out[c] = out[c].where(out[c].isna(), out[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        n = int((before.fillna("") != out[c].fillna("")).sum())
        if n:
            log.append(f"'{c}': {n} dəyərdə artıq boşluq təmizləndi")
        g = _case_groups(out[c])
        if g:
            counts = out[c].value_counts()
            mapping = {}
            for variants in g.values():
                best = max(variants, key=lambda v: counts.get(v, 0))
                mapping.update({v: best for v in variants})
            changed = int(out[c].isin([v for v in mapping if mapping[v] != v]).sum())
            out[c] = out[c].replace(mapping)
            log.append(f"'{c}': {changed} dəyərdə yazılış vahid formaya salındı")
        if _numeric_like(out[c]):
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
            log.append(f"'{c}': mətn → rəqəm tipinə çevrildi")
    for c in out.columns:
        if is_date_col(c) and _is_text(out[c]):
            parsed = pd.to_datetime(out[c].astype(str).str.strip(), format="mixed", dayfirst=True, errors="coerce")
            parsed[out[c].isna()] = pd.NaT
            bad = int(parsed.isna().sum() - out[c].isna().sum())
            out[c] = parsed
            log.append(f"'{c}': tarixlər vahid formata (YYYY-MM-DD) salındı" + (f", {bad} oxunmadı" if bad else ""))
    dups = int(out.duplicated().sum())
    if dups:
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"{dups} təkrarlanan sətir silindi")
    if not log:
        log.append("Təhlükəsiz düzəliş lazım olmadı")
    return out, log
