from pathlib import Path

import pandas as pd
import pytest
import requests
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
    original = dirty.copy(deep=True)
    before = quality_report(dirty)
    fixed, log = apply_safe_fixes(dirty, approved=True)
    assert fixed.duplicated().sum() == 0
    assert fixed["Product"].nunique() > 6  # Casing may have meaning; do not silently merge it.
    assert not pd.api.types.is_datetime64_any_dtype(fixed["Date"])
    date_parts = dirty["Date"].str.extract(r"^(\d{1,2})/(\d{1,2})/\d{4}$")
    first = pd.to_numeric(date_parts[0], errors="coerce")
    second = pd.to_numeric(date_parts[1], errors="coerce")
    ambiguous = first.between(1, 12) & second.between(1, 12) & first.ne(second)
    assert set(dirty.loc[ambiguous, "Date"]) <= set(fixed["Date"])
    assert fixed["Revenue"].isna().sum() == dirty["Revenue"].isna().sum()  # kor-koranə doldurmur
    assert quality_report(fixed)["score"] >= before["score"]
    pd.testing.assert_frame_equal(dirty, original)
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


def test_ai_unavailable_is_explicit_and_does_not_change_configuration(clean, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("Ollama unavailable for this test")

    previous_url = ai_agent.OLLAMA_URL
    monkeypatch.setattr(ai_agent.requests, "post", unavailable)
    with pytest.raises(ai_agent.AIError):
        ai_agent.parse_question(
            "Which product has the highest revenue?", list(clean.columns),
            {"metric": "Revenue", "numeric": ["Quantity", "Revenue"], "categorical": ["Product"]},
            retries=1,
        )
    with pytest.raises(ai_agent.AIError):
        ai_agent.explain_result("Total revenue?", {"op": "sum", "metric": "Revenue", "value": 5})
    assert ai_agent.OLLAMA_URL == previous_url


def test_mocked_ollama_answer_runs_on_sample_without_changing_original(clean, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"op":"group_sum","metric":"Revenue","group_by":"Product","top_n":3}'}}

    monkeypatch.setattr(ai_agent.requests, "post", lambda *args, **kwargs: Response())
    original = clean.copy(deep=True)
    operation = ai_agent.parse_question(
        "Which product has the highest revenue?", list(clean.columns),
        {"metric": "Revenue", "numeric": ["Quantity", "Revenue"], "categorical": ["Product"]},
    )
    result = run_operation(clean, operation)
    expected = clean.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(3)
    assert result["table"] == [
        {"Product": product, "Revenue": round(float(revenue), 2)}
        for product, revenue in expected.items()
    ]
    pd.testing.assert_frame_equal(clean, original)


@pytest.mark.parametrize("filename", ["sample_data.csv", "sample_data_dirty.csv"])
def test_bundled_sample_csv_quality_and_chart_data(filename):
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / filename)
    original = frame.copy(deep=True)
    report = quality_report(frame)
    assert 0 <= report["score"] <= 100
    assert summary_stats(frame)["rows"] == len(frame)
    by_product = run_operation(frame, {"op": "group_sum", "metric": "Revenue", "group_by": "Product"})
    trend = run_operation(frame, {"op": "trend", "metric": "Revenue", "date_col": "Date"})
    assert by_product["table"]
    assert trend["table"]
    assert all("period" in row and "Revenue" in row for row in trend["table"])
    pd.testing.assert_frame_equal(frame, original)
