"""Üzv 1: faylın oxunması, keyfiyyət yoxlaması, təhlükəsiz təmizləmə."""
from __future__ import annotations

import csv
import io
import math
import os
import re
from datetime import datetime
from typing import Tuple

import pandas as pd

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?$")
NON_NEGATIVE_HINTS = ("quantity", "revenue", "price", "amount", "say", "gelir")


def load_file(file) -> pd.DataFrame:
    """Read uploads without discarding identifier zeroes or guessing numeric separators."""
    name = str(os.fspath(file) if isinstance(file, (str, os.PathLike))
               else getattr(file, "name", "")).lower()
    if name and not name.endswith((".csv", ".xlsx", ".xls")):
        raise ValueError("Yalnız CSV və Excel (.xlsx/.xls) dəstəklənir.")
    if name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(file, dtype=object, keep_default_na=False, na_values=[""])
    else:
        if isinstance(file, (str, os.PathLike)):
            with open(file, "rb") as stream:
                contents = stream.read()
        elif hasattr(file, "getvalue"):
            contents = file.getvalue()
        else:
            position = file.tell() if hasattr(file, "tell") else None
            if hasattr(file, "seek"):
                file.seek(0)
            contents = file.read()
            if position is not None and hasattr(file, "seek"):
                file.seek(position)
        try:
            text = contents.decode("utf-8-sig") if isinstance(contents, bytes) else contents.lstrip("\ufeff")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV UTF-8 formatında olmalıdır; faylı UTF-8 kimi yadda saxlayın.") from exc
        if not text.strip():
            raise ValueError("CSV faylı boşdur.")
        try:
            separator = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            first_line = text.splitlines()[0]
            separator = max(",;\t|", key=first_line.count) if any(x in first_line for x in ",;\t|") else ","
        try:
            reader = csv.reader(io.StringIO(text), delimiter=separator, strict=True)
            header = next(row for row in reader if row and any(cell.strip() for cell in row))
            if len(header) != len(set(header)):
                raise ValueError("Datasetdə təkrarlanan sütun adları var.")
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                if len(row) != len(header):
                    raise ValueError("CSV sətirlərində sütun sayı başlıqla uyğun deyil.")
            # Parse text first: pandas' default inference otherwise drops leading zeroes.
            frame = pd.read_csv(io.StringIO(text), sep=separator, dtype=str,
                                keep_default_na=False, na_values=[""])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, StopIteration) as exc:
            raise ValueError("CSV oxuna bilmir; sütun ayırıcılarını və sətirləri yoxlayın.") from exc
    if frame.empty or not len(frame.columns):
        raise ValueError("Dataset boşdur; ən azı bir məlumat sətri lazımdır.")
    for column in _text_cols(frame):
        if not _is_identifier(column) and not is_date_col(column) and _numeric_like(frame[column]):
            frame[column] = _convert_numeric_text(frame[column])
    return frame


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _text_cols(df):
    return [c for c in df.columns if _is_text(df[c])]


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


DATE_RE = re.compile(
    r"(^|[_\s\-])(date|tarix)($|[_\s\-])",
    re.IGNORECASE
)

def is_date_col(name: str) -> bool:
    return bool(DATE_RE.search(str(name)))


AMBIG_RE = re.compile(r"^\s*[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*$")

ID_HINTS = (
    "id", "code", "phone", "zip",
    "postal", "sku", "kod"
)


def _is_identifier(column) -> bool:
    return any(hint in str(column).lower() for hint in ID_HINTS)

def _numeric_like(s: pd.Series) -> bool:
    nn = s.dropna().astype(str).str.strip()

    if nn.empty:
        return False

    if nn.str.match(r"^[+-]?0\d+").any():
        return False

    # A comma can mean either a decimal or a thousands separator; never guess.
    if nn.str.contains(",", regex=False).any():
        return False
    if not nn.str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?").all():
        return False
    try:
        if nn.str.fullmatch(r"[+-]?\d+").all():
            return all(-(2 ** 63) <= int(value) < 2 ** 63 for value in nn)
        return all(math.isfinite(float(value)) for value in nn)
    except (ValueError, OverflowError):
        return False


def _convert_numeric_text(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    integers = values.str.fullmatch(r"[+-]?\d+").all()
    parsed = [pd.NA if pd.isna(value) else (int(str(value).strip()) if integers else float(str(value).strip()))
              for value in series]
    return pd.Series(parsed, index=series.index, name=series.name,
                     dtype="Int64" if integers else "Float64")


def _date_status(value) -> tuple:
    """Classify dates conservatively. Locale-dependent dates remain unchanged."""
    if pd.isna(value):
        return "missing", None
    if not isinstance(value, str):
        if isinstance(value, (datetime, pd.Timestamp)):
            return "valid", pd.Timestamp(value)
        return "invalid", None
    text = value.strip()
    if ISO_RE.fullmatch(text):
        try:
            return "valid", pd.Timestamp(text)
        except (ValueError, OverflowError):
            return "invalid", None
    numeric = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{4})", text)
    if numeric:
        first, second, year = int(numeric[1]), int(numeric[3]), int(numeric[4])
        if 1 <= first <= 12 and 1 <= second <= 12 and first != second:
            return "ambiguous", None
        day, month = (second, first) if second > 12 else (first, second)
        try:
            return "valid", pd.Timestamp(datetime(year, month, day))
        except ValueError:
            return "invalid", None
    for format_ in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
        try:
            return "valid", pd.Timestamp(datetime.strptime(text, format_))
        except ValueError:
            pass
    return "invalid", None


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
        m = int(_missing_mask(df[c]).sum())
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
            add("spelling", c, len(g), f"'{c}': fərqli yazılışlar ({ex}); istifadəçi qərarı tələb olunur", False)
        if not _is_identifier(c) and _numeric_like(df[c]):
            add("dtype", c, len(s), f"'{c}' rəqəm kimi görünür, amma mətn tipindədir", True)
        if not _is_identifier(c):
            numeric_format = s.str.strip().str.fullmatch(r"[+-]?\d[\d., ]*") & s.str.contains(",", regex=False)
            if numeric_format.any():
                add("numeric_format", c, int(numeric_format.sum()),
                    f"'{c}': vergüllü rəqəmlər qeyri-müəyyəndir; avtomatik çevrilmir", False)
    # tarix formatları
    for c in df.columns:
        if is_date_col(c):
            s = df[c].dropna()
            if _is_text(s):
                s = s.astype(str).str.strip()
                statuses = s.map(lambda value: _date_status(value)[0])
                non_iso = int(((statuses == "valid") & ~s.str.match(ISO_RE)).sum())
                bad = int((statuses == "invalid").sum())
                ambiguous = int((statuses == "ambiguous").sum())
                if non_iso:
                    add("date_format", c, non_iso, f"'{c}': {non_iso} tarix ISO (YYYY-MM-DD) formatında deyil", True)
                if bad:
                    add("date_invalid", c, bad, f"'{c}': {bad} tarix oxuna bilmir", False)
                if ambiguous:
                    add("date_ambiguous", c, ambiguous,
                        f"'{c}': {ambiguous} tarixdə gün/ay sırası qeyri-müəyyəndir; avtomatik dəyişdirilmir", False)
    # şübhəli rəqəmlər
    for c in df.select_dtypes("number").columns:
        s = df[c].dropna()
        if any(h in str(c).lower() for h in NON_NEGATIVE_HINTS):
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


def apply_safe_fixes(df: pd.DataFrame, approved: bool = False) -> Tuple[pd.DataFrame, list]:
    """Apply approved, conservative fixes to a copy; uncertain values stay intact."""
    if approved is not True:
        raise PermissionError("Data dəyişiklikləri istifadəçi təsdiqi tələb edir.")
    out, log = df.copy(deep=True), []
    for c in _text_cols(out):
        if is_date_col(c):
            continue
        before = out[c].copy()
        out[c] = out[c].map(lambda value: re.sub(r"\s+", " ", value.strip())
                            if isinstance(value, str) else value)
        n = int((before.astype("object").fillna("") != out[c].astype("object").fillna("")).sum())
        if n:
            log.append(f"'{c}': {n} dəyərdə artıq boşluq təmizləndi")
        if (
            not _is_identifier(c)
            and _numeric_like(out[c])
        ):
            out[c] = _convert_numeric_text(out[c])
            log.append(f"'{c}': mətn → rəqəm tipinə çevrildi")
    for c in out.columns:
        if is_date_col(c) and _is_text(out[c]):
            def normalize_date(value):
                status, parsed = _date_status(value)
                if status != "valid" or not isinstance(value, str):
                    return value
                # Date-only values remain date-only; timestamps retain their time/zone.
                return parsed.isoformat() if ("T" in value or re.search(r"\d{2}:\d{2}", value)) else parsed.date().isoformat()
            before = out[c].copy()
            out[c] = out[c].map(normalize_date)
            count = int((before.astype("object").fillna("") != out[c].astype("object").fillna("")).sum())
            if count:
                log.append(f"'{c}': {count} dəqiq tarix ISO formatına çevrildi; qeyri-müəyyən tarixlər saxlanıldı")
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

        if not isinstance(action, str) or action not in ALLOWED_CLEANING_ACTIONS:
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
            if not isinstance(op["method"], str) or op["method"] != "median":
                raise ValueError("Yalnız median metodu dəstəklənir.")

            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("Median yalnız rəqəmsal sütuna tətbiq edilir.")

            if not pd.api.types.is_bool_dtype(df[column]) and not pd.api.types.is_complex_dtype(df[column]):
                if not pd.to_numeric(df[column], errors="coerce").notna().any():
                    raise ValueError("Median hesablamaq üçün rəqəm yoxdur.")
            else:
                raise ValueError("Boolean və complex sütunlar dəstəklənmir.")
            median = df[column].median()
            if pd.isna(median) or not math.isfinite(float(median)):
                raise ValueError("Sonlu median hesablamaq üçün rəqəm yoxdur.")

        validated.append({
            **op,
            "requires_approval": True,
        })

    return validated


def apply_cleaning_plan(
    df: pd.DataFrame,
    plan: dict,
    approved: bool = False
) -> tuple:
    """Planı yoxlayır və icazəli əməliyyatları surət üzərində icra edir."""

    operations = validate_cleaning_plan(plan, df)

    if operations and approved is not True:
        raise PermissionError(
            "Bu cleaning plan istifadəçi təsdiqi tələb edir."
        )

    result = df.copy(deep=True)
    log = []

    for op in operations:
        action = op["action"]

        if action == "trim_whitespace":
            column = op["column"]
            before = result[column].copy()

            result[column] = result[column].map(
                lambda x: re.sub(r"\s+", " ", x.strip())
                if isinstance(x, str) else x
            )

            changed = int(
                (before.astype("object").fillna("") != result[column].astype("object").fillna("")).sum()
            )
            log.append(f"{column}: {changed} mətn təmizləndi")

        elif action == "fill_missing":
            column = op["column"]
            count = int(result[column].isna().sum())
            median = result[column].median()

            if pd.isna(median) or not math.isfinite(float(median)):
                raise ValueError("Median hesablana bilmir.")

            if pd.api.types.is_integer_dtype(result[column].dtype) and float(median) % 1:
                # Nullable integer arrays reject a fractional median. Promote the copy.
                result[column] = result[column].astype("Float64")
            result[column] = result[column].fillna(median)
            log.append(
                f"{column}: {count} boş xana median ({median}) ilə dolduruldu"
            )

        elif action == "remove_duplicates":
            count = int(result.duplicated().sum())
            result = result.drop_duplicates().reset_index(drop=True)
            log.append(f"{count} dublikat sətir silindi")

    return result, log


def preview_cleaning_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Təklif edilən dəyişiklikləri orijinal dataya toxunmadan göstərir."""

    operations = validate_cleaning_plan(plan, df)

    # Bu yalnız simulyasiyadır, istifadəçi təsdiqi deyil.
    preview_df, log = apply_cleaning_plan(
        df, plan, approved=True
    )

    return {
        "original_rows": len(df),
        "preview_rows": len(preview_df),
        "rows_removed": len(df) - len(preview_df),
        "requires_approval": any(
            op["requires_approval"] for op in operations
        ),
        "operations": operations,
        "log": log,
        "preview_data": preview_df,
        "before_report": quality_report(df),
        "after_report": quality_report(preview_df),
    }

