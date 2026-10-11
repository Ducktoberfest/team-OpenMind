"""Contract tests use mocked Ollama HTTP responses, never pretend a model ran."""
import json

import pandas as pd
import pytest
import requests

import ai_agent


class Response:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.body


def mock_chat(monkeypatch, content):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response({"message": {"content": content}})

    monkeypatch.setattr(ai_agent.requests, "post", post)
    return calls


@pytest.fixture
def dirty():
    return pd.DataFrame({"Product": [" Mouse ", "Keyboard", "Keyboard"], "Revenue": [None, 40.0, 40.0]})


def test_real_cleaning_request_is_validated_and_does_not_modify_data(monkeypatch, dirty):
    plan = {"operations": [
        {"action": "trim_whitespace", "column": "Product"},
        {"action": "fill_missing", "column": "Revenue", "method": "median"},
        {"action": "remove_duplicates"},
    ]}
    calls = mock_chat(monkeypatch, json.dumps(plan))
    original = dirty.copy(deep=True)
    assert ai_agent.get_cleaning_plan(dirty) == plan
    pd.testing.assert_frame_equal(dirty, original)
    url, kwargs = calls[0]
    assert url.endswith("/api/chat")
    assert kwargs["json"]["model"] == ai_agent.MODEL
    assert kwargs["json"]["format"] == "json"
    assert kwargs["json"]["stream"] is False
    profile = json.loads(kwargs["json"]["messages"][1]["content"].split("\n", 1)[1])
    assert profile["summary"]["missing_by_col"] == {"Revenue": 1}
    assert profile["summary"]["duplicate_rows"] == 1


@pytest.mark.parametrize("plan", [
    {"operations": [{"action": "execute_python", "code": "__import__('os').system('echo unsafe')"}]},
    {"operations": [{"action": "trim_whitespace", "column": "Product", "code": "print(1)"}]},
    {"operations": [{"action": "fill_missing", "column": "Product", "method": "median"}]},
    {"operations": [{"action": "trim_whitespace", "column": "Nonexistent"}]},
    {"operations": [], "python": "print(1)"},
])
def test_unsafe_or_invalid_cleaning_is_rejected(monkeypatch, dirty, plan):
    mock_chat(monkeypatch, json.dumps(plan))
    original = dirty.copy(deep=True)
    with pytest.raises(ai_agent.AIError, match="unsafe or invalid cleaning plan"):
        ai_agent.get_cleaning_plan(dirty)
    pd.testing.assert_frame_equal(dirty, original)


def test_model_can_report_no_supported_cleaning_actions(monkeypatch, dirty):
    mock_chat(monkeypatch, '{"operations": []}')
    assert ai_agent.get_cleaning_plan(dirty) == {"operations": []}


@pytest.mark.parametrize("content", [
    "__import__('os').system('echo unsafe')",
    'Here is your plan: {"operations": []}',
    '{"operations": [], "operations": [{"action": "remove_duplicates"}]}',
    '{"operations": NaN}',
    '[]',
])
def test_non_json_and_ambiguous_json_are_rejected(monkeypatch, dirty, content):
    mock_chat(monkeypatch, content)
    with pytest.raises(ai_agent.AIError):
        ai_agent.get_cleaning_plan(dirty)


def test_single_json_fence_is_accepted(monkeypatch, dirty):
    mock_chat(monkeypatch, '```json\n{"operations": []}\n```')
    assert ai_agent.get_cleaning_plan(dirty) == {"operations": []}


def test_unavailable_model_is_explicit_for_every_ai_api(monkeypatch, dirty):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(ai_agent.requests, "post", fail)
    with pytest.raises(ai_agent.AIError, match="ollama serve"):
        ai_agent.get_cleaning_plan(dirty)
    with pytest.raises(ai_agent.AIError, match="ollama serve"):
        ai_agent.parse_question("What is total Revenue?", list(dirty.columns), {"numeric": ["Revenue"]})
    with pytest.raises(ai_agent.AIError, match="ollama serve"):
        ai_agent.explain_result("Revenue?", {"op": "sum", "value": 80})


def test_missing_model_http_error_has_pull_instructions(monkeypatch):
    monkeypatch.setattr(ai_agent.requests, "post", lambda *a, **kw: Response({}, 404))
    with pytest.raises(ai_agent.AIError, match="ollama pull"):
        ai_agent.parse_question("How many rows?", ["Revenue"])


def test_timeout_has_actionable_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(ai_agent.requests, "post", timeout)
    with pytest.raises(ai_agent.AIError, match="OLLAMA_TIMEOUT"):
        ai_agent.explain_result("Count?", {"value": 10})


@pytest.mark.parametrize("body", [{}, {"message": {}}, {"message": {"content": ""}}, {"message": {"content": 10}}])
def test_invalid_ollama_response_is_explicit(monkeypatch, body):
    monkeypatch.setattr(ai_agent.requests, "post", lambda *a, **kw: Response(body))
    with pytest.raises(ai_agent.AIError, match="empty or invalid chat response"):
        ai_agent.explain_result("Count?", {"value": 10})


def test_valid_analyst_operation_from_http(monkeypatch):
    operation = {"op": "group_sum", "metric": "Revenue", "group_by": "Product", "top_n": 3}
    calls = mock_chat(monkeypatch, json.dumps(operation))
    assert ai_agent.parse_question("Which product has highest Revenue?", ["Product", "Revenue"], ["Revenue"]) == operation
    assert "Revenue" in calls[0][1]["json"]["messages"][0]["content"]


def test_invalid_analyst_response_does_not_become_fallback(monkeypatch):
    calls = mock_chat(monkeypatch, '{"op": "execute_python", "code": "print(1)"}')
    with pytest.raises(ai_agent.AIError, match="valid analyst operation"):
        ai_agent.parse_question("How many rows?", ["Revenue"], retries=2)
    assert len(calls) == 2


def test_malformed_operation_can_retry_to_valid_json(monkeypatch):
    outputs = iter(["not JSON", '{"op":"count"}'])
    monkeypatch.setattr(ai_agent.requests, "post", lambda *a, **kw: Response({"message": {"content": next(outputs)}}))
    assert ai_agent.parse_question("How many rows?", ["Revenue"], retries=2) == {"op": "count"}


def test_explanation_is_actual_model_text(monkeypatch):
    text = "The total Revenue is 80. Check the underlying transactions."
    calls = mock_chat(monkeypatch, text)
    assert ai_agent.explain_result("What is Revenue?", {"value": 80}) == text
    assert '"value": 80' in calls[0][1]["json"]["messages"][1]["content"]


def test_status_requires_the_selected_installed_model(monkeypatch):
    monkeypatch.setattr(ai_agent, "MODEL", "gemma4:e2b")
    monkeypatch.setattr(ai_agent.requests, "get", lambda *a, **kw: Response({"models": [{"name": "gemma3:4b"}]}))
    status = ai_agent.ollama_status()
    assert status["available"] is False
    assert status["models"] == ["gemma3:4b"]
    assert "not installed" in status["message"]
    assert not ai_agent.ollama_available()


def test_status_matches_installed_selected_model(monkeypatch):
    monkeypatch.setattr(ai_agent, "MODEL", "gemma4:e2b")
    monkeypatch.setattr(ai_agent.requests, "get", lambda *a, **kw: Response({"models": [{"name": "gemma4:e2b"}]}))
    assert ai_agent.ollama_available()


def test_status_matches_default_latest_tag(monkeypatch):
    monkeypatch.setattr(ai_agent, "MODEL", "gemma4")
    monkeypatch.setattr(ai_agent.requests, "get", lambda *a, **kw: Response({"models": [{"name": "gemma4:latest"}]}))
    assert ai_agent.ollama_available()


@pytest.mark.parametrize("body", [{}, {"models": None}, {"models": "gemma4:e2b"}])
def test_status_rejects_malformed_model_listing(monkeypatch, body):
    monkeypatch.setattr(ai_agent.requests, "get", lambda *a, **kw: Response(body))
    status = ai_agent.ollama_status()
    assert status["available"] is False
    assert "invalid model list" in status["message"]


def test_status_handles_unavailable_service(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(ai_agent.requests, "get", fail)
    status = ai_agent.ollama_status()
    assert status["available"] is False
    assert "ollama serve" in status["message"]
