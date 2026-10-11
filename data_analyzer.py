
"""
DataPilot - Data Analyzer

Üzv 1: statistik analiz və Gemma tərəfindən təklif edilən
JSON əməliyyatlarının təhlükəsiz icrası.

Gemma Python kodu göndərmir.
Yalnız ALLOWED_OPS siyahısındakı əməliyyatları seçir.

Nümunə:
{
    "op": "group_sum",
    "metric": "Revenue",
    "group_by": "Product",
    "top_n": 5,
    "filters": [
        {
            "column": "Region",
            "operator": "==",
            "value": "Baku"
        }
    ]
}

Trend:
{
    "op": "trend",
    "metric": "Revenue",
    "date_col": "Order_Date",
    "freq": "M"
}
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ALLOWED_OPS = {
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "group_sum",
    "group_mean",
    "group_count",
    "trend",
}

GROUP_OPS = {
    "group_sum",
    "group_mean",
    "group_count",
}

FILTER_OPS = {
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "contains",
}

ALLOWED_FREQ = {
    "D": "D",
    "W": "W-MON",
    "M": "MS",
    "Q": "QS",
    "Y": "YS",
}

MAX_FILTERS = 10
MAX_TOP_N = 100
DEFAULT_TOP_N = 10

DATE_RE = re.compile(
    r"(^|[_\s-])(date|tarix)($|[_\s-])",
    re.IGNORECASE,
)

SCALAR_OPS = {
    "sum",
    "mean",
    "min",
    "max",
    "count",
}

NUMERIC_OPS = {
    "sum",
    "mean",
    "group_sum",
    "group_mean",
    "trend",
}


# ============================================================
# 2. JSON-SAFE HELPERS
# ============================================================

def _safe_number(value: Any):
    """
    NaN və sonsuzluq kimi JSON üçün problemli
    nəticələri None-a çevirir.
    """

    if pd.isna(value):
        return None

    number = float(value)

    if not math.isfinite(number):
        return None

    return round(number, 2)


def _json_value(value: Any):
    """Pandas nəticələrini JSON-a uyğun formaya çevirir."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): _json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (pd.Timedelta,)):
        return str(value)

    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return bool(value)

        if not math.isfinite(float(value)):
            return None

        return value

    if hasattr(value, "item"):
        return _json_value(value.item())

    return value


def _check_dataframe(df: pd.DataFrame):
    """Dataset strukturunu yoxlayır."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Dataset pandas DataFrame olmalıdır."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Datasetdə təkrarlanan sütun adları var."
        )

    if not all(isinstance(c, str) for c in df.columns):
        raise ValueError(
            "Sütun adları mətn (string) olmalıdır."
        )


def _is_numeric_metric(series: pd.Series) -> bool:
    """Boolean və complex istisna olmaqla rəqəmsal sütun."""

    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and not pd.api.types.is_complex_dtype(series)
    )


# ============================================================
# 3. SUMMARY STATISTICS
# ============================================================

def summary_stats(df: pd.DataFrame) -> dict:
    """
    Datasetin ümumi statistik hesabatını qaytarır.

    Nəticə Gemma promptuna və Streamlit interfeysinə
    göndərilə bilər.
    """

    _check_dataframe(df)

    numeric_columns = [
        c for c in df.columns
        if _is_numeric_metric(df[c])
    ]

    missing = df.isna().sum()

    numeric_stats = {}

    for col in numeric_columns:
        series = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = series.dropna()

        numeric_stats[col] = {
            "count": int(valid.count()),
            "missing": int(series.isna().sum()),
            "mean": _safe_number(valid.mean()),
            "median": _safe_number(valid.median()),
            "min": _safe_number(valid.min()),
            "max": _safe_number(valid.max()),
            "sum": _safe_number(valid.sum()) if len(valid) else None,
            "std": (
                _safe_number(valid.std())
                if len(valid) > 1
                else None
            ),
        }

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "missing_total": int(missing.sum()),
        "missing_by_col": {
            col: int(value)
            for col, value in missing.items()
            if value > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "numeric": numeric_stats,
    }


# ============================================================
# 4. OPERATION VALIDATION
# ============================================================

def validate_op(op: dict, columns: list) -> dict:
    """
    Gemma-nın JSON əməliyyatını yoxlayır.

    Burada yalnız JSON strukturu və sütun adları
    yoxlanır. Dataset tipləri run_operation
    daxilində ayrıca yoxlanılır.
    """

    if not isinstance(op, dict):
        raise ValueError(
            "AI əməliyyatı JSON obyekt olmalıdır."
        )

    kind = op.get("op")

    if not isinstance(kind, str) or kind not in ALLOWED_OPS:
        raise ValueError(
            f"Naməlum əməliyyat: {kind}"
        )

    if not isinstance(columns, list):
        raise ValueError(
            "columns siyahı olmalıdır."
        )

    if not all(isinstance(column, str) for column in columns):
        raise ValueError("Sütun adları mətn (string) olmalıdır.")

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Təkrarlanan sütun adları dəstəklənmir."
        )

    allowed_fields = {
        "op",
        "metric",
        "group_by",
        "top_n",
        "ascending",
        "filters",
        "date_col",
        "freq",
    }

    if set(op) - allowed_fields:
        raise ValueError(
            f"İcazəsiz JSON parametrləri: {set(op) - allowed_fields}"
        )

    metric = op.get("metric")

    # count və group_count üçün metric məcburi deyil.
    if kind not in {"count", "group_count"}:
        if not isinstance(metric, str) or metric not in columns:
            raise ValueError(
                f"Metric sütunu tapılmadı: {metric}"
            )

    elif metric is not None:
        if not isinstance(metric, str) or metric not in columns:
            raise ValueError(
                f"Metric sütunu tapılmadı: {metric}"
            )

    # Group əməliyyatları
    if kind in GROUP_OPS:
        group_by = op.get("group_by")

        if not isinstance(group_by, str) or group_by not in columns:
            raise ValueError(
                f"group_by sütunu tapılmadı: {group_by}"
            )

    elif "group_by" in op:
        raise ValueError(
            "group_by yalnız group əməliyyatları üçündür."
        )

    # top_n yalnız group əməliyyatlarında istifadə edilir.
    if "top_n" in op:
        if kind not in GROUP_OPS:
            raise ValueError(
                "top_n yalnız group əməliyyatlarında işləyir."
            )

        top_n = op["top_n"]

        if (
            type(top_n) is not int
            or not 1 <= top_n <= MAX_TOP_N
        ):
            raise ValueError(
                f"top_n 1 ilə {MAX_TOP_N} arasında tam ədəd olmalıdır."
            )

    # ascending
    if "ascending" in op:
        if kind not in GROUP_OPS:
            raise ValueError(
                "ascending yalnız group əməliyyatlarında işləyir."
            )

        if type(op["ascending"]) is not bool:
            raise ValueError(
                "ascending true və ya false olmalıdır."
            )

    # Trend parametrləri
    if kind == "trend":
        freq = op.get("freq", "M")

        if not isinstance(freq, str) or freq not in ALLOWED_FREQ:
            raise ValueError(
                "freq yalnız D, W, M, Q və Y ola bilər."
            )

        date_col = op.get("date_col")

        if date_col is not None:
            if not isinstance(date_col, str) or date_col not in columns:
                raise ValueError(
                    f"Tarix sütunu tapılmadı: {date_col}"
                )

    elif "freq" in op or "date_col" in op:
        raise ValueError(
            "freq və date_col yalnız trend əməliyyatı üçündür."
        )

    # Filters
    filters = op.get("filters", [])

    if filters is None:
        filters = []

    if not isinstance(filters, list):
        raise ValueError(
            "filters siyahı olmalıdır."
        )

    if len(filters) > MAX_FILTERS:
        raise ValueError(
            f"Maksimum {MAX_FILTERS} filter dəstəklənir."
        )

    for f in filters:
        if not isinstance(f, dict):
            raise ValueError(
                "Hər filter JSON obyekt olmalıdır."
            )

        if set(f) != {"column", "operator", "value"}:
            raise ValueError(
                f"Yanlış filter strukturu: {f}"
            )

        col = f["column"]
        operator = f["operator"]
        value = f["value"]

        if not isinstance(col, str) or col not in columns:
            raise ValueError(
                f"Filter sütunu tapılmadı: {col}"
            )

        if (
            not isinstance(operator, str)
            or operator not in FILTER_OPS
        ):
            raise ValueError(
                f"İcazəsiz filter operatoru: {operator}"
            )

        if value is None:
            raise ValueError(
                "Filter value boş ola bilməz."
            )

        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                "Filter value sadə JSON dəyəri olmalıdır."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                "Filter value sonsuz və ya NaN ola bilməz."
            )

    return op


# ============================================================
# 5. FILTER EXECUTION
# ============================================================

def _apply_filters(
    df: pd.DataFrame,
    filters: list | None,
) -> pd.DataFrame:
    """
    Filtrləri ardıcıl tətbiq edir.

    Orijinal DataFrame dəyişdirilmir.
    """

    result = df.copy(deep=True)

    for f in filters or []:
        column = f["column"]
        operator = f["operator"]
        value = f["value"]

        series = result[column]

        if operator == "contains":
            mask = (
                series.astype("string")
                .str.contains(
                    str(value),
                    case=False,
                    regex=False,
                    na=False,
                )
            )

            result = result.loc[mask].copy()
            continue

        if _is_numeric_metric(series):
            if isinstance(value, bool):
                raise ValueError(
                    f"'{column}' üçün rəqəmsal filter lazımdır."
                )

            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"'{column}' üçün rəqəm tələb olunur."
                )

            if not math.isfinite(value):
                raise ValueError(
                    "Filter dəyəri sonlu rəqəm olmalıdır."
                )

        elif pd.api.types.is_datetime64_any_dtype(series):
            if not isinstance(value, str):
                raise ValueError("Tarix filteri aydın mətn formatında olmalıdır.")
            date_text = value
            parsed = _parse_trend_dates(pd.Series([date_text])).iloc[0]
            if pd.isna(parsed):
                raise ValueError(
                    f"Yanlış və ya qeyri-müəyyən tarix filteri: {value}"
                )
            value = parsed
            if series.dt.tz is not None:
                if re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", date_text, re.IGNORECASE):
                    value = value.tz_localize("UTC").tz_convert(series.dt.tz)
                else:
                    value = value.tz_localize(series.dt.tz)

        else:
            if operator not in {"==", "!="}:
                raise ValueError(
                    f"'{column}' üçün yalnız == və != dəstəklənir."
                )

            series = series.astype("string").str.casefold()
            value = str(value).casefold()

        if operator == "==":
            mask = series == value
        elif operator == "!=":
            mask = series != value
        elif operator == ">":
            mask = series > value
        elif operator == "<":
            mask = series < value
        elif operator == ">=":
            mask = series >= value
        elif operator == "<=":
            mask = series <= value
        else:
            raise ValueError(
                f"Naməlum filter operatoru: {operator}"
            )

        result = result.loc[mask.fillna(False)].copy()

    return result


# ============================================================
# 6. DATE COLUMN DETECTION
# ============================================================

def _find_date_column(
    df: pd.DataFrame,
    requested: str | None = None,
) -> str:
    """Trend üçün uyğun tarix sütununu müəyyən edir."""

    if requested is not None:
        return requested

    datetime_cols = [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    if datetime_cols:
        return datetime_cols[0]

    date_cols = [
        col for col in df.columns
        if DATE_RE.search(col)
    ]

    if date_cols:
        return date_cols[0]

    raise ValueError(
        "Trend üçün tarix sütunu tapılmadı. "
        "date_col parametrini göndərin."
    )


def _parse_trend_dates(series: pd.Series) -> pd.Series:
    """
    Tarixləri oxuyur.

    Qeyri-müəyyən və oxunmayan tarixlər
    trend hesablamasına daxil edilmir.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(
            series,
            errors="coerce",
        )

    values = series.astype("string").str.strip()

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    # Use explicit formats; never guess between month/day and day/month.
    for pattern, date_format in (
        (r"\d{4}-\d{1,2}-\d{1,2}", "%Y-%m-%d"),
        (r"\d{4}/\d{1,2}/\d{1,2}", "%Y/%m/%d"),
        (r"[A-Za-z]+ \d{1,2}, \d{4}", "%b %d, %Y"),
        (r"[A-Za-z]+ \d{1,2}, \d{4}", "%B %d, %Y"),
        (r"\d{1,2} [A-Za-z]+ \d{4}", "%d %b %Y"),
        (r"\d{1,2} [A-Za-z]+ \d{4}", "%d %B %Y"),
    ):
        mask = values.str.fullmatch(pattern, na=False) & parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(
                values.loc[mask], format=date_format, errors="coerce"
            )

    for separator in ("/", ".", "-"):
        parts = values.str.extract(
            rf"^(\d{{1,2}}){re.escape(separator)}(\d{{1,2}})"
            rf"{re.escape(separator)}(\d{{4}})$"
        )
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        day_first = (first > 12) & second.between(1, 12)
        month_first = first.between(1, 12) & (second > 12)
        equal = first.eq(second) & first.between(1, 12)
        for mask, date_format in (
            (day_first | equal, f"%d{separator}%m{separator}%Y"),
            (month_first, f"%m{separator}%d{separator}%Y"),
        ):
            if mask.any():
                parsed.loc[mask] = pd.to_datetime(
                    values.loc[mask], format=date_format, errors="coerce"
                )

    # ISO timestamps specify their calendar order explicitly. Normalize a
    # timezone offset to UTC so mixed offsets can still form one trend index.
    iso_time = values.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*", na=False
    )
    if iso_time.any():
        parsed.loc[iso_time] = pd.to_datetime(
            values.loc[iso_time], format="ISO8601", errors="coerce", utc=True
        ).dt.tz_localize(None)

    return parsed


# ============================================================
# 7. RUN AI OPERATION
# ============================================================

def run_operation(
    df: pd.DataFrame,
    op: dict,
) -> dict:
    """
    Gemma-nın JSON əməliyyatını təhlükəsiz icra edir.

    Nəticələr:
      Scalar: {"op", "metric", "value", ...}
      Group:  {"op", "metric", "table", ...}
      Trend:  {"op", "metric", "table", ...}
    """

    _check_dataframe(df)

    op = validate_op(op, list(df.columns))

    kind = op["op"]
    metric = op.get("metric")

    # Numeric metric validation
    if kind in NUMERIC_OPS:
        if not _is_numeric_metric(df[metric]):
            raise ValueError(
                f"'{metric}' rəqəmsal sütun olmalıdır."
            )

    if kind in {"min", "max"}:
        if (
            pd.api.types.is_complex_dtype(df[metric])
            or pd.api.types.is_bool_dtype(df[metric])
        ):
            raise ValueError(
                f"'{metric}' bu əməliyyat üçün uyğun deyil."
            )

    filtered = _apply_filters(
        df,
        op.get("filters"),
    )

    result = {
        "op": kind,
        "metric": metric,
        "filters": op.get("filters") or [],
        "rows_used": int(len(filtered)),
    }

    # --------------------------------------------------------
    # Scalar operations
    # --------------------------------------------------------

    if kind in {"sum", "mean", "min", "max"}:
        series = filtered[metric]

        if series.dropna().empty:
            result["value"] = None
        else:
            value = getattr(series, kind)()

            if _is_numeric_metric(series):
                result["value"] = _safe_number(value)
            else:
                result["value"] = _json_value(value)

    elif kind == "count":
        # count burada filterdən sonra qalan sətir sayıdır.
        result["value"] = int(len(filtered))

    # --------------------------------------------------------
    # Group operations
    # --------------------------------------------------------

    elif kind in GROUP_OPS:
        group_col = op["group_by"]

        if kind == "group_count":
            grouped = (
                filtered.groupby(
                    group_col,
                    dropna=False,
                )
                .size()
                .rename("count")
            )

            value_name = "count"

        else:
            method = kind.split("_")[1]

            aggregate = filtered.groupby(
                group_col, dropna=False,
            )[metric]
            grouped = (
                aggregate.sum(min_count=1)
                if method == "sum"
                else aggregate.mean()
            )

            value_name = metric

        ascending = op.get("ascending", False)

        grouped = grouped.sort_values(
            ascending=ascending,
            kind="stable",
        )

        top_n = op.get("top_n", DEFAULT_TOP_N)

        grouped = grouped.head(top_n)

        # The grouped index and aggregate must have distinct names, including
        # group_count grouped by a column literally named "count".
        if value_name == group_col:
            value_name = f"{value_name}_{kind.removeprefix('group_')}"
        table = grouped.rename(value_name).reset_index()

        result["group_by"] = group_col
        result["value_column"] = value_name
        result["table"] = [
            {
                str(k): _json_value(v)
                for k, v in record.items()
            }
            for record in table.to_dict("records")
        ]

    # --------------------------------------------------------
    # Trend operation
    # --------------------------------------------------------

    elif kind == "trend":
        value_name = "period_sum" if metric == "period" else metric
        date_col = _find_date_column(
            filtered,
            op.get("date_col"),
        )

        freq = op.get("freq", "M")
        pandas_freq = ALLOWED_FREQ[freq]

        dates = _parse_trend_dates(
            filtered[date_col]
        )

        trend_df = pd.DataFrame({
            "_date": dates,
            "_metric": filtered[metric],
        })

        invalid_dates = int(
            trend_df["_date"].isna().sum()
        )

        trend_df = trend_df.dropna(
            subset=["_date"]
        )

        if trend_df.empty:
            table = []

        else:
            trend_df = trend_df.set_index("_date")

            aggregated = (
                trend_df["_metric"]
                .resample(pandas_freq)
                .sum(min_count=1)
            )

            table = []

            for date, value in aggregated.items():
                table.append({
                    "period": date.strftime("%Y-%m-%d"),
                    value_name: _safe_number(value),
                })

        result["date_col"] = date_col
        result["freq"] = freq
        result["invalid_dates"] = invalid_dates
        result["value_column"] = value_name
        result["rows_used"] = int(len(trend_df))
        result["table"] = table

    return result


# ============================================================
# 8. OPTIONAL: MULTIPLE OPERATIONS
# ============================================================

def run_operations(
    df: pd.DataFrame,
    operations: list,
) -> list:
    """
    Bir neçə Gemma əməliyyatını ardıcıl icra edir.

    Maksimum 20 əməliyyat qəbul edilir.
    """

    _check_dataframe(df)

    if not isinstance(operations, list):
        raise ValueError(
            "operations siyahı olmalıdır."
        )

    if len(operations) > 20:
        raise ValueError(
            "Maksimum 20 əməliyyat icra edilə bilər."
        )

    results = []

    for operation in operations:
        results.append(
            run_operation(df, operation)
        )

    return results

