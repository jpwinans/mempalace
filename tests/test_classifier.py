"""Unit tests for mempalace.provenance.classifier.

HTTP-mocking tests verify the request shape, response parsing, and
failure-soft contract without touching a live substrate. The live
substrate calibration test lives separately in
``tests/test_classifier_calibration.py`` and is gated on substrate
availability.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import patch

import pytest

from mempalace.provenance.classifier import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    qwen3_classifier,
)


# ---------------------------------------------------------------------------
# HTTP mocking helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Mimics the urllib.request.urlopen context-manager response."""

    def __init__(self, body: str, status: int = 200):
        self._body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ok_payload(content: str) -> dict[str, Any]:
    """Build a valid mlx_lm.server chat-completions response wrapping
    ``content`` as the assistant message."""

    return {
        "id": "test-id",
        "object": "chat.completion",
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _classifier_json(
    is_provenance: bool = True,
    person: str | None = "father",
    relation_type: str | None = "family",
    quote: str | None = "be still",
    confidence: float = 0.9,
) -> str:
    """Serialize the inner classifier-output JSON (string the model emits)."""

    return json.dumps({
        "is_provenance": is_provenance,
        "person": person,
        "relation_type": relation_type,
        "quote": quote,
        "confidence": confidence,
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_parsed_classifier_dict():
    body = json.dumps(_ok_payload(_classifier_json()))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("My father said 'be still'")
    assert result["is_provenance"] is True
    assert result["person"] == "father"
    assert result["relation_type"] == "family"
    assert result["quote"] == "be still"
    assert result["confidence"] == 0.9


def test_handles_markdown_code_fence_around_json():
    """Some prompt completions wrap JSON in ```json ... ``` despite the
    prompt asking for raw JSON. Stripper must handle both fence forms."""

    fenced = "```json\n" + _classifier_json() + "\n```"
    body = json.dumps(_ok_payload(fenced))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("My father said 'be still'")
    assert result["is_provenance"] is True


def test_handles_bare_backtick_fence():
    """Code fence without language marker — ``` ... ``` ."""

    fenced = "```\n" + _classifier_json() + "\n```"
    body = json.dumps(_ok_payload(fenced))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("...")
    assert result["is_provenance"] is True


def test_rejection_response_returns_rejection_dict():
    body = json.dumps(_ok_payload(
        _classifier_json(is_provenance=False, person=None, relation_type=None,
                         quote=None, confidence=0.0)
    ))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("Operational content with no attribution.")
    assert result["is_provenance"] is False
    assert result["confidence"] == 0.0


def test_request_body_carries_model_and_temperature_zero():
    """Verify the HTTP request shape: model name, temp=0 for determinism."""

    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["data"] = req.data
        captured["url"] = req.full_url
        body = json.dumps(_ok_payload(_classifier_json()))
        return _FakeResponse(body)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        qwen3_classifier("My father said 'be still'")
    parsed = json.loads(captured["data"].decode("utf-8"))
    assert parsed["model"] == DEFAULT_MODEL
    assert parsed["temperature"] == 0.0
    assert parsed["max_tokens"] == 256
    assert isinstance(parsed["messages"], list)
    assert parsed["messages"][0]["role"] == "user"
    assert "My father said 'be still'" in parsed["messages"][0]["content"]
    assert captured["url"] == DEFAULT_ENDPOINT


# ---------------------------------------------------------------------------
# Failure-soft paths (per architect envelope §D2.5)
# ---------------------------------------------------------------------------

def test_network_error_returns_rejection_dict():
    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise urllib.error.URLError("Connection refused")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False
    assert result["confidence"] == 0.0


def test_http_error_returns_rejection_dict():
    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url="http://test/", code=503, msg="Service Unavailable",
            hdrs=None, fp=io.BytesIO(b""),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_timeout_returns_rejection_dict():
    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        raise TimeoutError("substrate slow")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_malformed_outer_json_returns_rejection_dict():
    """Substrate returns non-JSON gibberish at the HTTP layer."""

    with patch("urllib.request.urlopen", return_value=_FakeResponse("not json")):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_missing_choices_field_returns_rejection_dict():
    """Substrate returns valid JSON but wrong shape (no choices)."""

    body = json.dumps({"error": "model not found"})
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_malformed_inner_json_returns_rejection_dict():
    """Substrate's chat-completion content isn't valid JSON."""

    body = json.dumps(_ok_payload("This is not JSON at all, just prose."))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_inner_content_missing_is_provenance_key_returns_rejection_dict():
    """Inner JSON parses but missing the required is_provenance key."""

    body = json.dumps(_ok_payload(json.dumps({"person": "father"})))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


def test_inner_content_not_a_dict_returns_rejection_dict():
    body = json.dumps(_ok_payload("[1, 2, 3]"))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("anything")
    assert result["is_provenance"] is False


# ---------------------------------------------------------------------------
# Confidence coercion
# ---------------------------------------------------------------------------

def test_confidence_string_coerced_to_zero():
    body = json.dumps(_ok_payload(
        json.dumps({"is_provenance": True, "person": "father",
                    "confidence": "high"})
    ))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("...")
    assert result["confidence"] == 0.0


def test_confidence_above_one_clamped_to_one():
    body = json.dumps(_ok_payload(
        json.dumps({"is_provenance": True, "person": "father",
                    "confidence": 1.5})
    ))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("...")
    assert result["confidence"] == 1.0


def test_confidence_below_zero_clamped_to_zero():
    body = json.dumps(_ok_payload(
        json.dumps({"is_provenance": True, "person": "father",
                    "confidence": -0.5})
    ))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        result = qwen3_classifier("...")
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------

def test_endpoint_override_via_env(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        body = json.dumps(_ok_payload(_classifier_json()))
        return _FakeResponse(body)

    monkeypatch.setenv(
        "MEMPALACE_PROVENANCE_CLASSIFIER_URL", "http://alt-substrate:9999/v1/chat/completions"
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        qwen3_classifier("...")
    assert captured["url"] == "http://alt-substrate:9999/v1/chat/completions"


def test_model_override_via_env(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["data"] = req.data
        body = json.dumps(_ok_payload(_classifier_json()))
        return _FakeResponse(body)

    monkeypatch.setenv("MEMPALACE_PROVENANCE_CLASSIFIER_MODEL", "alt-model-id")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        qwen3_classifier("...")
    parsed = json.loads(captured["data"].decode("utf-8"))
    assert parsed["model"] == "alt-model-id"
