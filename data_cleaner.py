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
        # ayırıcı (, ; tab) avtomatik tapılır; kodlaşdırma: UTF-8 -> Windows-1254 -> Windows-1251 -> Latin-1
        last_exc = None
        for enc in ("utf-8-sig", "cp1254", "cp1251", "latin-1"):
            try:
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_csv(file, sep=None, engine="python", encoding=enc)
                if df.shape[1] == 1:  # ayırıcı tapılmadısa vergüllə cəhd et
                    if hasattr(file, "seek"):
                        file.seek(0)
                    df = pd.read_csv(file, encoding=enc)
                return df
            except UnicodeDecodeError as exc:
                last_exc = exc
        raise last_exc
    raise ValueError("Yalnız CSV və Excel (.xlsx/.xls) dəstəklənir.")


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _text_cols(df):
    return [c for c in df.columns if _is_text(df[c])]


# Tarix sütunu adları: Date, order_date, Tarix, Sifariş tarixi, Tarixi, OrderDate ...
DATE_RE = re.compile(
    r"(?i:(^|[_\s\-])(date|tarix\w*)($|[_\s\-]))|[a-z]Date$"
)


def is_date_col(name: str) -> bool:
    return bool(DATE_RE.search(str(name)))


def _looks_like_dates(s: pd.Series) -> bool:
    """Mətn sütununun əksəriyyəti tarix kimi oxunursa True."""
    if not _is_text(s):
        return False
    nn = s.dropna().astype(str).str.strip()
    if nn.empty or _numeric_like(s):
        return False
    if nn.str.contains(r"\d").mean() < 0.8:
        return False
    parsed = pd.to_datetime(nn.head(200), format="mixed", dayfirst=True, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def date_columns(df: pd.DataFrame) -> list:
    """Tarix sütunları: adına görə, datetime tipinə görə və ya məzmununa görə."""
    return [c for c in df.columns
            if is_date_col(c)
            or pd.api.types.is_datetime64_any_dtype(df[c])
            or _looks_like_dates(df[c])]


AMBIG_RE = re.compile(r"^\s*[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*$")
ID_HINTS = ("id", "code", "phone", "zip", "postal", "sku", "kod")


def _numeric_like(s: pd.Series) -> bool:
    nn = s.dropna().astype(str).str.strip()
    if nn.empty:
        return False
    if nn.str.match(r"^0\d+").any():
        return False
    if nn.str.match(AMBIG_RE).any():
        return False
    converted = pd.to_numeric(nn.str.replace(",", ".", regex=False), errors="coerce")
    return bool(converted.notna().all())


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
    dcols = set(date_columns(df))

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
        if c in dcols:
            continue
        s = df[c].dropna().astype(str)
        ws = int((s != s.str.strip().str.replace(r"\s+", " ", regex=True)).sum())
        if ws:
            add("whitespace", c, ws, f"'{c}': {ws} dəyərdə artıq boşluq", True)
        g = _case_groups(df[c])
        if g:
            ex = "; ".join("/".join(sorted(v)) for v in list(g.values())[:2])
            add("spelling", c, len(g), f"'{c}': fərqli yazılışlar ({ex})", True)
        if _numeric_like(df[c]) and c not in dcols:
            add("dtype", c, len(s), f"'{c}' rəqəm kimi görünür, amma mətn tipindədir", True)

    # tarix formatları
    for c in df.columns:
        if c in dcols:
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
    dcols = set(date_columns(out))
    for c in _text_cols(out):
        if c in dcols:
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
        if (
            not any(h in str(c).lower() for h in ID_HINTS)
            and _numeric_like(out[c])
        ):
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
            log.append(f"'{c}': mətn → rəqəm tipinə çevrildi")

    for c in out.columns:
        if c in dcols and _is_text(out[c]):
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


# ===== AI CLEANING PLAN =====
ALLOWED_CLEANING_ACTIONS = {
    "trim_whitespace",
    "fill_missing",
    "remove_duplicates",
}

ACTION_FIELDS = {
    "trim_whitespace": {"action", "column"},
    "fill_missing": {"action", "column", "method"},
    "remove_duplicates": {"action"},
}


def validate_cleaning_plan(plan: dict, df: pd.DataFrame) -> list:
    """AI planını yoxlayır, amma datanı dəyişmir."""
    if not isinstance(plan, dict):
        raise ValueError("Cleaning plan JSON obyekt olmalıdır.")
    if set(plan) != {"operations"}:
        raise ValueError("Plan yalnız 'operations' açarını qəbul edir.")
    operations = plan["operations"]
    if not isinstance(operations, list):
        raise ValueError("'operations' siyahı olmalıdır.")
    if len(operations) > 20:
        raise ValueError("Maksimum 20 əməliyyata icazə verilir.")
    if not df.columns.is_unique:
        raise ValueError("Datasetdə təkrarlanan sütun adları var.")

    validated = []
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("Hər əməliyyat JSON obyekt olmalıdır.")
        action = op.get("action")
        if action not in ALLOWED_CLEANING_ACTIONS:
            raise ValueError(f"Qadağan edilmiş əməliyyat: {action}")
        if set(op) != ACTION_FIELDS[action]:
            raise ValueError(f"Yanlış parametrlər: {action}")
        column = op.get("column")
        if action != "remove_duplicates":
            if not isinstance(column, str) or column not in df.columns:
                raise ValueError(f"Sütun tapılmadı: {column}")
        if action == "trim_whitespace":
            if not _is_text(df[column]):
                raise ValueError("Whitespace yalnız mətn sütununa tətbiq edilir.")
        if action == "fill_missing":
            if op["method"] != "median":
                raise ValueError("Yalnız median metodu dəstəklənir.")
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("Median yalnız rəqəmsal sütuna tətbiq edilir.")
            if not pd.api.types.is_bool_dtype(df[column]) and not pd.api.types.is_complex_dtype(df[column]):
                if not pd.to_numeric(df[column], errors="coerce").notna().any():
                    raise ValueError("Median hesablamaq üçün rəqəm yoxdur.")
            else:
                raise ValueError("Boolean və complex sütunlar dəstəklənmir.")
        validated.append({
            **op,
            "requires_approval": action in {"fill_missing", "remove_duplicates"},
        })
    return validated


def apply_cleaning_plan(df: pd.DataFrame, plan: dict, approved: bool = False) -> tuple:
    """Planı yoxlayır və icazəli əməliyyatları surət üzərində icra edir."""
    operations = validate_cleaning_plan(plan, df)
    if any(op["requires_approval"] for op in operations) and not approved:
        raise PermissionError("Bu cleaning plan istifadəçi təsdiqi tələb edir.")

    result = df.copy(deep=True)
    log = []
    for op in operations:
        action = op["action"]
        if action == "trim_whitespace":
            column = op["column"]
            before = result[column].copy()
            result[column] = result[column].map(
                lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x
            )
            changed = int((before.fillna("") != result[column].fillna("")).sum())
            log.append(f"{column}: {changed} mətn təmizləndi")
        elif action == "fill_missing":
            column = op["column"]
            count = int(result[column].isna().sum())
            median = result[column].median()
            if pd.isna(median):
                raise ValueError("Median hesablana bilmir.")
            result[column] = result[column].fillna(median)
            log.append(f"{column}: {count} boş xana median ilə dolduruldu")
        elif action == "remove_duplicates":
            count = int(result.duplicated().sum())
            result = result.drop_duplicates().reset_index(drop=True)
            log.append(f"{count} dublikat sətir silindi")
    return result, log


def preview_cleaning_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Təklif edilən dəyişiklikləri orijinal dataya toxunmadan göstərir."""
    operations = validate_cleaning_plan(plan, df)
    # Bu yalnız simulyasiyadır, istifadəçi təsdiqi deyil.
    preview_df, log = apply_cleaning_plan(df, plan, approved=True)
    return {
        "original_rows": len(df),
        "preview_rows": len(preview_df),
        "rows_removed": len(df) - len(preview_df),
        "requires_approval": any(op["requires_approval"] for op in operations),
        "operations": operations,
        "log": log,
        "preview_data": preview_df,
    }
