"""Run the real Streamlit UI; only external Ollama calls are mocked.

These tests do not establish that a real Gemma model is installed or works.
Cleaning, validation, quality scoring and analyst computations run normally.
"""
import json
import io
from pathlib import Path

import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs

import ai_agent
from data_cleaner import load_file, quality_report


ROOT = Path(__file__).resolve().parents[1]
BASIC_PREVIEW = "🔎 Preview Basic Fixes"
BASIC_APPLY = "✅ Apply Basic Fixes"
AI_GENERATE = "🤖 Generate AI Cleaning Plan"
AI_APPLY = "✅ Apply AI Cleaning Plan"
AI_REJECT = "❌ Reject AI Cleaning Plan"
RESET = "🔄 Reset Data (orijinal fayla qayıt)"
PLAN = {"operations": [
    {"action": "trim_whitespace", "column": "Product"},
    {"action": "fill_missing", "column": "Revenue", "method": "median"},
    {"action": "remove_duplicates"},
]}


@pytest.fixture(autouse=True)
def isolate_ollama(monkeypatch):
    """Never contact a user's local model while running the automated UI suite."""
    st.cache_data.clear()
    monkeypatch.setattr(ai_agent, "ollama_status", lambda: {
        "available": False, "models": [], "model": ai_agent.MODEL,
        "message": "Ollama is unavailable. Start Ollama with 'ollama serve'.",
    })
    yield
    st.cache_data.clear()


def available_model(monkeypatch):
    monkeypatch.setattr(ai_agent, "ollama_status", lambda: {
        "available": True, "models": [ai_agent.MODEL], "model": ai_agent.MODEL,
        "message": f"Ollama is ready with '{ai_agent.MODEL}'.",
    })
    st.cache_data.clear()


class ChatResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}}


def chat_responses(monkeypatch, *responses):
    """Exercise real ai_agent HTTP payloads, JSON parsing and validation."""
    outputs = iter(responses)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        output = next(outputs)
        if isinstance(output, Exception):
            raise output
        return ChatResponse(output)

    monkeypatch.setattr(ai_agent.requests, "post", post)
    return calls


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def assert_runs(app):
    assert not app.exception, [exc.message for exc in app.exception]
    return app


def app_with_sample(filename="sample_data_dirty.csv"):
    app = assert_runs(AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run())
    return assert_runs(app.selectbox[0].select(filename).run())


def upload(name, contents):
    return UploadedFile(UploadedFileRec("test-file", name, "text/csv", contents), FileURLs())


def assert_pending_cleared(app):
    for key in ("ai_plan", "ai_preview", "safe_preview"):
        assert key not in app.session_state


def test_initial_screen_starts_without_model_or_dataset():
    app = assert_runs(AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run())
    assert "df" not in app.session_state
    assert any("CSV/Excel" in entry.value for entry in app.info)
    assert any("ollama serve" in entry.value for entry in app.caption)


@pytest.mark.parametrize("filename", ["sample_data.csv", "sample_data_dirty.csv"])
def test_bundled_samples_render_data_quality_and_charts_without_ai(filename):
    app = app_with_sample(filename)
    expected = load_file(ROOT / filename)
    pd.testing.assert_frame_equal(app.session_state["df"], expected)
    pd.testing.assert_frame_equal(app.session_state["original_df"], expected)
    assert app.session_state["original_bytes"] == (ROOT / filename).read_bytes()
    assert app.session_state["original_name"] == filename
    assert len(app.get("plotly_chart")) >= 3
    quality = next(metric for metric in app.metric if metric.label == "Keyfiyyət balı")
    assert quality.value == f"{quality_report(expected)['score']}/100"
    assert button(app, AI_GENERATE).disabled
    assert app.chat_input[0].disabled
    assert app.session_state["history"] == []
    assert any("demo/fallback" in entry.value for entry in app.info)


def test_basic_preview_requires_approval_preserves_original_and_reset_restores_data():
    app = app_with_sample()
    original = app.session_state["df"].copy(deep=True)
    before_score = quality_report(original)["score"]
    assert_runs(button(app, BASIC_PREVIEW).click().run())
    preview = app.session_state["safe_preview"]["preview_data"]
    assert quality_report(preview)["score"] > before_score
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert button(app, BASIC_APPLY).disabled
    assert not app.checkbox[0].value

    assert_runs(app.checkbox[0].check().run())
    assert not button(app, BASIC_APPLY).disabled
    assert_runs(button(app, BASIC_APPLY).click().run())
    assert quality_report(app.session_state["df"])["score"] > before_score
    assert app.session_state["df"].duplicated().sum() == 0
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert_pending_cleared(app)
    assert app.session_state["log"]
    scores = {metric.label: metric.value for metric in app.metric}
    assert scores["Before Quality"] == f"{before_score}/100"
    assert scores["After Quality"] == f"{quality_report(app.session_state['df'])['score']}/100"

    assert_runs(button(app, RESET).click().run())
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert app.session_state["log"] == []
    assert app.session_state["history"] == []
    assert_pending_cleared(app)


def test_model_json_plan_is_previewed_then_applied_only_after_approval(monkeypatch):
    available_model(monkeypatch)
    calls = chat_responses(monkeypatch, json.dumps(PLAN))
    app = app_with_sample()
    original = app.session_state["df"].copy(deep=True)
    assert_runs(button(app, AI_GENERATE).click().run())
    assert app.session_state["ai_plan"] == PLAN
    assert len(calls) == 1
    assert calls[0][0].endswith("/api/chat")
    assert calls[0][1]["json"]["format"] == "json"
    assert calls[0][1]["json"]["model"] == ai_agent.MODEL
    assert button(app, AI_APPLY).disabled
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)

    approval = next(box for box in app.checkbox if "AI cleaning plan" in box.label)
    assert_runs(approval.check().run())
    assert not button(app, AI_APPLY).disabled
    assert_runs(button(app, AI_APPLY).click().run())
    cleaned = app.session_state["df"]
    assert cleaned["Revenue"].isna().sum() == 0
    assert cleaned.duplicated().sum() == 0
    assert quality_report(cleaned)["score"] > quality_report(original)["score"]
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert_pending_cleared(app)
    assert app.session_state["log"]


def test_rejecting_plan_clears_preview_and_never_changes_data(monkeypatch):
    available_model(monkeypatch)
    chat_responses(monkeypatch, json.dumps(PLAN))
    app = app_with_sample()
    original = app.session_state["df"].copy(deep=True)
    assert_runs(button(app, AI_GENERATE).click().run())
    assert_runs(button(app, AI_REJECT).click().run())
    assert_pending_cleared(app)
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    assert app.session_state["log"] == []


def test_replacement_ai_plan_does_not_reuse_previous_approval(monkeypatch):
    available_model(monkeypatch)
    chat_responses(monkeypatch, json.dumps(PLAN), json.dumps({"operations": [{"action": "remove_duplicates"}]}))
    app = app_with_sample()
    assert_runs(button(app, AI_GENERATE).click().run())
    assert_runs(app.checkbox[0].check().run())
    assert not button(app, AI_APPLY).disabled
    assert_runs(button(app, AI_GENERATE).click().run())
    assert button(app, AI_APPLY).disabled
    assert not app.checkbox[0].value
    assert app.session_state["log"] == []


@pytest.mark.parametrize("response", [
    json.dumps({"operations": [{"action": "execute_python", "code": "raise RuntimeError('unsafe')"}]}),
    requests.ConnectionError("Ollama disconnected during request"),
])
def test_failed_or_unsafe_ai_plan_is_explicit_and_keeps_data(monkeypatch, response):
    available_model(monkeypatch)
    chat_responses(monkeypatch, response)
    app = app_with_sample()
    original = app.session_state["df"].copy(deep=True)
    assert_runs(button(app, AI_GENERATE).click().run())
    assert app.error
    assert "data dəyişmədi" in app.error[0].value
    assert_pending_cleared(app)
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert app.session_state["log"] == []


def test_analyst_executes_validated_json_and_shows_actual_model_explanation(monkeypatch):
    available_model(monkeypatch)
    explanation = "Mocked model explanation for the computed revenue, solely for this automated test."
    calls = chat_responses(monkeypatch, '{"op":"sum","metric":"Revenue"}', explanation)
    app = app_with_sample("sample_data.csv")
    original = app.session_state["df"].copy(deep=True)
    assert_runs(app.chat_input[0].set_value("What is total Revenue?").run())
    history = app.session_state["history"]
    assert len(history) == 2
    assert history[0] == {"role": "user", "text": "What is total Revenue?"}
    assert history[1]["role"] == "assistant"
    assert explanation in history[1]["text"]
    assert "Validated operation" in history[1]["text"]
    assert history[1]["result"]["value"] == pytest.approx(round(float(original["Revenue"].sum()), 2))
    assert len(calls) == 2
    assert calls[0][1]["json"]["format"] == "json"
    assert "format" not in calls[1][1]["json"]
    pd.testing.assert_frame_equal(app.session_state["df"], original)


def test_analyst_failure_is_honest_and_has_no_fake_result(monkeypatch):
    available_model(monkeypatch)
    chat_responses(monkeypatch, requests.ConnectionError("offline"))
    app = app_with_sample("sample_data.csv")
    assert_runs(app.chat_input[0].set_value("What is total Revenue?").run())
    response = app.session_state["history"][-1]
    assert "AI sualı icra edilmədi" in response["text"]
    assert "Ollama is unavailable" in response["text"]
    assert "result" not in response
    assert "table" not in response


def test_switching_dataset_expires_plans_approval_and_chat_history(monkeypatch):
    available_model(monkeypatch)
    chat_responses(monkeypatch, '{"op":"count"}', "Mocked row count explanation.", json.dumps(PLAN))
    app = app_with_sample()
    assert_runs(app.chat_input[0].set_value("How many rows?").run())
    assert len(app.session_state["history"]) == 2
    assert_runs(button(app, AI_GENERATE).click().run())
    assert_runs(app.checkbox[0].check().run())
    assert_runs(button(app, BASIC_PREVIEW).click().run())
    assert "ai_plan" in app.session_state
    assert "safe_preview" in app.session_state
    version = app.session_state["data_version"]

    source_selector = next(box for box in app.selectbox if box.label == "və ya demo fayl")
    assert_runs(source_selector.select("sample_data.csv").run())
    expected = load_file(ROOT / "sample_data.csv")
    pd.testing.assert_frame_equal(app.session_state["df"], expected)
    pd.testing.assert_frame_equal(app.session_state["original_df"], expected)
    assert app.session_state["data_version"] > version
    assert app.session_state["history"] == []
    assert app.session_state["log"] == []
    assert_pending_cleared(app)
    assert not app.checkbox


@pytest.mark.parametrize("suffix", ["csv", "xlsx"])
def test_uploaded_file_runs_real_loader_and_keeps_exact_original_bytes(monkeypatch, suffix):
    """Inject the upload transport: AppTest does not expose a file_uploader setter."""
    csv = ("Date,Product,CustomerID,Revenue\n"
           "2026-01-01, Alpha ,0001,10\n"
           "2026-01-02,Beta,0002,\n"
           "2026-01-02,Beta,0002,\n"
           "01/02/2026,Gamma,0003,30\n").encode()
    if suffix == "csv":
        raw = csv
    else:
        excel = io.BytesIO()
        pd.read_csv(io.BytesIO(csv), dtype=str, keep_default_na=False).to_excel(excel, index=False)
        raw = excel.getvalue()
    source = upload(f"uploaded.{suffix}", raw)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: source)
    app = assert_runs(AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run())
    original = app.session_state["df"].copy(deep=True)
    assert original["CustomerID"].tolist() == ["0001", "0002", "0002", "0003"]
    assert original["Revenue"].isna().sum() == 2
    assert original.duplicated().sum() == 1
    assert original["Date"].iloc[-1] == "01/02/2026"
    issues = {issue["type"] for issue in quality_report(original)["issues"]}
    assert {"missing", "duplicates", "whitespace", "date_ambiguous"} <= issues
    assert app.session_state["original_bytes"] == raw
    assert app.session_state["original_name"] == source.name
    assert app.get("plotly_chart")

    assert_runs(button(app, BASIC_PREVIEW).click().run())
    assert button(app, BASIC_APPLY).disabled
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    assert_runs(app.checkbox[0].check().run())
    assert_runs(button(app, BASIC_APPLY).click().run())
    assert app.session_state["df"].duplicated().sum() == 0
    assert app.session_state["df"]["Revenue"].isna().sum() == 1
    assert app.session_state["df"]["Date"].iloc[-1] == "01/02/2026"
    assert app.session_state["original_bytes"] == raw
    pd.testing.assert_frame_equal(app.session_state["original_df"], original)
    assert_runs(button(app, RESET).click().run())
    pd.testing.assert_frame_equal(app.session_state["df"], original)
    assert app.session_state["original_bytes"] == raw


def test_malformed_uploaded_csv_is_a_user_error_without_partial_data(monkeypatch):
    source = upload("malformed.csv", b"Product,Revenue\nAlpha,10,extra\n")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: source)
    app = assert_runs(AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run())
    assert app.error
    assert "Fayl oxuna bilmədi" in app.error[0].value
    assert "df" not in app.session_state
    assert "original_df" not in app.session_state


def test_reupload_same_name_with_changed_contents_expires_pending_plan(monkeypatch):
    available_model(monkeypatch)
    chat_responses(monkeypatch, json.dumps(PLAN))
    current = {"file": upload("dataset.csv", (ROOT / "sample_data_dirty.csv").read_bytes())}
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: current["file"])
    app = assert_runs(AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run())
    assert_runs(button(app, AI_GENERATE).click().run())
    assert_runs(app.checkbox[0].check().run())
    previous_version = app.session_state["data_version"]

    raw = (ROOT / "sample_data.csv").read_bytes()
    current["file"] = upload("dataset.csv", raw)
    assert_runs(app.run())
    assert_pending_cleared(app)
    assert app.session_state["data_version"] > previous_version
    assert app.session_state["original_bytes"] == raw
    pd.testing.assert_frame_equal(app.session_state["df"], load_file(current["file"]))
    assert not app.checkbox
