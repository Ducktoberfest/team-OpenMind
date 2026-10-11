"""Üzv 1: statistika və Gemma-nın JSON əməliyyatlarını icra edən funksiyalar.
Model heç vaxt kod yazmır, yalnız aşağıdakı ALLOWED_OPS-dan birini seçir.
op formatı:
  {"op": "group_sum", "metric": "Revenue", "group_by": "Product", "top_n": 5,
   "filters": [{"column": "Region", "operator": "==", "value": "Baku"}]}
trend üçün: {"op": "trend", "metric": "Revenue", "freq": "M"}
"""
from __future__ import annotations

import pandas as pd

from data_cleaner import date_columns

ALLOWED_OPS = {"sum", "mean", "min", "max", "count",
               "group_sum", "group_mean", "group_count", "trend"}
GROUP_OPS = {"group_sum", "group_mean", "group_count"}
FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "contains"}


def summary_stats(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number")
    return {
        "rows": len(df), "cols": len(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": {c: int(v) for c, v in df.isna().sum().items() if v},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric": {c: {"mean": round(float(num[c].mean()), 2), "min": float(num[c].min()),
                        "max": float(num[c].max()), "sum": round(float(num[c].sum()), 2)}
                    for c in num.columns},
    }


def validate_op(op: dict, columns: list) -> dict:
    if not isinstance(op, dict) or op.get("op") not in ALLOWED_OPS:
        raise ValueError(f"Naməlum əməliyyat: {op.get('op') if isinstance(op, dict) else op}")
    kind = op["op"]
    if kind != "count" and kind != "group_count" and op.get("metric") not in columns:
        raise ValueError(f"Metric sütunu tapılmadı: {op.get('metric')}")
    if kind in GROUP_OPS and op.get("group_by") not in columns:
        raise ValueError(f"group_by sütunu tapılmadı: {op.get('group_by')}")
    if op.get("date_col") not in (None, *columns):
        raise ValueError(f"date_col tapılmadı: {op.get('date_col')}")
    for f in op.get("filters") or []:
        if f.get("column") not in columns or f.get("operator") not in FILTER_OPS:
            raise ValueError(f"Yanlış filter: {f}")
    return op


def _apply_filters(df, filters):
    for f in filters or []:
        col, o, v = f["column"], f["operator"], f["value"]
        s = df[col]
        if o == "contains":
            df = df[s.astype(str).str.contains(str(v), case=False, na=False)]
        else:
            if pd.api.types.is_numeric_dtype(s):
                v = float(v)
            elif pd.api.types.is_datetime64_any_dtype(s):
                v = pd.to_datetime(v)
            elif o in ("==", "!="):
                s, v = s.astype(str).str.lower(), str(v).lower()
            df = df[{"==": s == v, "!=": s != v, ">": s > v, "<": s < v,
                     ">=": s >= v, "<=": s <= v}[o]]
    return df


def run_operation(df: pd.DataFrame, op: dict) -> dict:
    """Qaytarır: {"op", "metric", "value"} və ya {"op", "table": [...]}"""
    op = validate_op(op, list(df.columns))
    kind, metric = op["op"], op.get("metric")
    d = _apply_filters(df, op.get("filters"))
    res = {"op": kind, "metric": metric, "filters": op.get("filters") or [], "rows_used": len(d)}

    if kind in {"sum", "mean", "min", "max"}:
        res["value"] = round(float(getattr(d[metric], kind)()), 2)
    elif kind == "count":
        res["value"] = int(len(d))
    elif kind in GROUP_OPS:
        g = op["group_by"]
        if kind == "group_count":
            t = d.groupby(g).size().rename("count")
        else:
            t = getattr(d.groupby(g)[metric], kind.split("_")[1])().rename(metric)
        t = t.sort_values(ascending=bool(op.get("ascending", False))).round(2)
        if op.get("top_n"):
            t = t.head(int(op["top_n"]))
        res["group_by"] = g
        res["table"] = t.reset_index().to_dict("records")
    elif kind == "trend":
        dc = op.get("date_col") or next(iter(date_columns(df)), None)
        if dc is None:
            raise ValueError("Tarix sütunu tapılmadı")
        dd = d.assign(_d=pd.to_datetime(d[dc], errors="coerce", format="mixed", dayfirst=True)).dropna(subset=["_d"])
        t = dd.set_index("_d")[metric].resample("MS" if op.get("freq", "M") == "M" else op["freq"]).sum().round(2)
        res["table"] = [{"period": i.strftime("%Y-%m"), metric: float(v)} for i, v in t.items()]
    return res
