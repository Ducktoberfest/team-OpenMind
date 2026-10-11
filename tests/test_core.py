import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import pytest
from generate_sample_data import make_clean, make_dirty
from data_cleaner import quality_report, apply_safe_fixes
from data_analyzer import run_operation, summary_stats
import ai_agent


@pytest.fixture(scope="module")
def clean():
    return make_clean()


@pytest.fixture(scope="module")
def dirty(clean):
    return make_dirty(clean)


def test_dirty_issues_detected(dirty):
    types = {i["type"] for i in quality_report(dirty)["issues"]}
    assert {"missing", "duplicates", "whitespace", "spelling", "date_format", "negative", "outlier"} <= types


def test_safe_fixes(dirty):
    fixed, log = apply_safe_fixes(dirty)
    assert fixed.duplicated().sum() == 0
    assert fixed["Product"].nunique() == 6
    assert pd.api.types.is_datetime64_any_dtype(fixed["Date"])
    assert fixed["Revenue"].isna().sum() == dirty["Revenue"].isna().sum()  # kor-koranə doldurmur
    assert log


def test_run_operation(clean):
    r = run_operation(clean, {"op": "group_sum", "metric": "Revenue", "group_by": "Product", "top_n": 3})
    assert len(r["table"]) == 3
    assert r["table"][0]["Revenue"] >= r["table"][1]["Revenue"]
    assert run_operation(clean, {"op": "count"})["value"] == len(clean)
    assert len(run_operation(clean, {"op": "trend", "metric": "Revenue", "freq": "M"})["table"]) >= 11
    f = run_operation(clean, {"op": "sum", "metric": "Revenue", "filters": [{"column": "Region", "operator": "==", "value": "baku"}]})
    assert f["value"] > 0


def test_rejects_bad_ops(clean):
    with pytest.raises(ValueError):
        run_operation(clean, {"op": "exec", "metric": "Revenue"})
    with pytest.raises(ValueError):
        run_operation(clean, {"op": "sum", "metric": "Nonexistent"})


def test_fallback_parser(clean):
    ai_agent.OLLAMA_URL = "http://127.0.0.1:9"  # model yoxdur
    ctx = {"numeric": ["Quantity", "Revenue"], "categorical": ["Product", "Region"], "metric": "Revenue", "date": "Date"}
    op = ai_agent.parse_question("Hansı Product daha çox gəlir gətirib?", list(clean.columns), ctx, retries=1)
    assert op["op"] == "group_sum" and op["group_by"] == "Product"
    assert "Nəticə" in ai_agent.explain_result("x", {"op": "sum", "metric": "Revenue", "value": 5})


def test_other_schema_works(clean):
    """Tamam fərqli sütun adları olan dataset də işləməlidir."""
    from data_cleaner import date_columns
    other = clean.rename(columns={"Date": "Order Day", "Product": "Item", "Revenue": "Sales", "Region": "Store"})
    other["Order Day"] = other["Order Day"].dt.strftime("%d.%m.%Y")
    assert date_columns(other) == ["Order Day"]
    fixed, _ = apply_safe_fixes(other)
    assert pd.api.types.is_datetime64_any_dtype(fixed["Order Day"])
    r = run_operation(fixed, {"op": "trend", "metric": "Sales"})
    assert len(r["table"]) >= 11
    ai_agent.OLLAMA_URL = "http://127.0.0.1:9"
    ctx = {"numeric": ["Quantity", "Sales"], "categorical": ["Store", "Item"], "metric": "Sales", "date": "Order Day"}
    op = ai_agent.parse_question("Which one sold the most?", list(fixed.columns), ctx, retries=1)
    assert op["op"] == "group_sum" and op["metric"] == "Sales"
