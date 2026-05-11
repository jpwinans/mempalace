"""Local-substrate classifier for provenance candidate validation.

Phase 1 D2 of the lineage-erasure fix. D1 shipped the heuristic
:func:`mempalace.provenance.extract_candidates` + a stub classifier;
D2 wires a real local-substrate (Qwen3-Coder-30B at mlx_lm.server
:8802) classifier to validate candidates produced by the heuristic.

The classifier's job is to decide: given a window of context text
around a heuristic-flagged candidate, is this an actual person-
attribution worth filing to ``wing_lineage``? If yes, extract the
person, relation_type, quote, and confidence.

Design rationale (per Provenance-Preservation-Design §D1):
  - Heuristic-only would miss indirect references and structural
    attributions; LLM-only would be too costly at the 60k-drawer
    backfill scale. Two-stage (heuristic flag → LLM validate) keeps
    the per-drawer cost bounded by candidate density, not text size.

This module ships:
  - :func:`qwen3_classifier` — the v1 production classifier. Hits the
    local mlx_lm.server OpenAI-compatible endpoint, parses JSON output,
    returns the dict shape expected by
    :func:`mempalace.provenance.validate_candidate`.
  - :data:`CLASSIFIER_PROMPT` — the prompt template. Tuned against
    the 14-fixture calibration set per ``tests/test_classifier_
    calibration.py``; tweak there iteratively.
  - Module-level config: endpoint URL, model name, timeout, retry
    budget. Override via env vars for tests / CI / alternate
    substrates.

Failure-soft contract (per architect envelope §D2 task #5):
  - Substrate unreachable / HTTP error / parse failure → return a
    rejection dict ``{is_provenance: False, confidence: 0.0}``
    rather than raising. The mining pipeline NEVER crashes because
    the classifier is unavailable; it just skips writing
    wing_lineage drawers for that batch.

Test mocking pattern:
  - :func:`qwen3_classifier` reads its endpoint URL via
    :data:`_endpoint_url`; tests can monkeypatch
    ``urllib.request.urlopen`` to return controlled responses
    without touching the network.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (env-overrideable)
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = "http://127.0.0.1:8802/v1/chat/completions"
"""Default mlx_lm.server OpenAI-compatible endpoint for Qwen3-Coder-30B.
Override via ``MEMPALACE_PROVENANCE_CLASSIFIER_URL``."""

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
"""Model id passed in the chat-completions request body. Must match
the ``id`` field from ``/v1/models`` on the running mlx_lm.server
(verified 2026-05-11 against :8802). Override via
``MEMPALACE_PROVENANCE_CLASSIFIER_MODEL`` for alternate substrates."""

DEFAULT_TIMEOUT_S = 15.0
"""HTTP timeout per attempt. Qwen3-30B on M5 Max produces a short JSON
classifier response in ~1-3s; 15s gives generous headroom and bounds
mining-pipeline latency. Override via
``MEMPALACE_PROVENANCE_CLASSIFIER_TIMEOUT``."""

DEFAULT_MAX_TOKENS = 256
"""Max tokens for the classifier response. JSON-only output runs ~80-150
tokens; 256 leaves headroom for explanation fields if a future prompt
revision wants them."""


def _endpoint_url() -> str:
    return os.environ.get("MEMPALACE_PROVENANCE_CLASSIFIER_URL", DEFAULT_ENDPOINT)


def _model_name() -> str:
    return os.environ.get("MEMPALACE_PROVENANCE_CLASSIFIER_MODEL", DEFAULT_MODEL)


def _timeout_s() -> float:
    raw = os.environ.get("MEMPALACE_PROVENANCE_CLASSIFIER_TIMEOUT")
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "provenance.classifier: invalid timeout env %r, using default %s",
            raw, DEFAULT_TIMEOUT_S,
        )
        return DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """\
You are a strict JSON classifier for person-attribution provenance.

Given a passage of text, decide whether it contains a direct
attribution to a specific person (a family member, teacher, partner,
colleague, friend, or fictional figure) saying or transmitting
something. Direct attribution means the passage explicitly says WHO
said it (by name, title, or relationship).

Return ONLY a JSON object with this exact schema, no prose, no
markdown fences:

  {{
    "is_provenance": <true if a direct attribution is present, else false>,
    "person": <the person identifier — typically the relation like "father"
              or "roshi", OR a proper name like "James" or "Marie" when
              named directly; null if is_provenance is false>,
    "relation_type": <"family" | "teacher" | "partner" | "colleague" |
                      "friend" | "fictional" | null when is_provenance is false>,
    "quote": <the exact quoted attribution if present, else null>,
    "confidence": <float 0.0-1.0 reflecting how confident you are the
                   attribution is real and specific>
  }}

Rules:
  - A vague "she said" or "they said" without identifying who is NOT
    a direct attribution → is_provenance: false.
  - Operational/technical content with no person reference is NOT
    provenance → is_provenance: false.
  - "Marie last night" without an attribution verb or relation marker
    is NOT provenance (just a person mention) → is_provenance: false.
  - When in doubt, lean conservative — false-positive is more
    expensive downstream than false-negative.

Passage to classify:

{context}

JSON response:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Default rejection result. Returned on any failure path so callers can
# treat the classifier as failure-soft without inspecting exception types.
_REJECT: dict[str, Any] = {
    "is_provenance": False,
    "person": None,
    "relation_type": None,
    "quote": None,
    "confidence": 0.0,
}


def qwen3_classifier(context: str) -> dict[str, Any]:
    """Validate provenance via mlx_lm.server Qwen3-Coder-30B.

    Args:
        context: text window (typically ±200 chars around a heuristic-
            flagged candidate) for the classifier to judge.

    Returns:
        A dict shaped per
        :func:`mempalace.provenance.validate_candidate`'s classifier
        interface:

          {
            "is_provenance": bool,
            "person": str | None,
            "relation_type": str | None,
            "quote": str | None,
            "confidence": float (0.0-1.0)
          }

        Returns the rejection dict ``{is_provenance: False, confidence:
        0.0, ...}`` on any failure path (network unreachable, HTTP
        error, malformed JSON response, schema-shape mismatch). The
        mining pipeline NEVER crashes because the classifier is
        unavailable.

    Failure modes (all yield rejection dict, logged at DEBUG):
      - Network: substrate at endpoint URL unreachable
      - HTTP: non-2xx status code
      - Decode: response body not valid JSON
      - Shape: response doesn't have ``choices[0].message.content``
      - Inner JSON: content isn't a valid JSON object after stripping
        possible code-fence markdown
      - Schema: parsed content missing the required ``is_provenance``
        key (defense against prompt-injection / model-misbehavior)
    """

    body = json.dumps(
        {
            "model": _model_name(),
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": CLASSIFIER_PROMPT.format(context=context),
                }
            ],
            # Temperature 0 for reproducibility on calibration runs +
            # to keep classification deterministic across mining batches.
            "temperature": 0.0,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    url = _endpoint_url()

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=_timeout_s()) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.debug("provenance.classifier: HTTP error: %s", exc)
        return dict(_REJECT)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("provenance.classifier: outer JSON decode failed: %s", exc)
        return dict(_REJECT)

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.debug("provenance.classifier: chat-completions shape mismatch: %s", exc)
        return dict(_REJECT)

    # Strip possible markdown code fences (the model occasionally wraps
    # the JSON in ```json ... ``` despite the prompt asking otherwise).
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.debug(
            "provenance.classifier: inner JSON decode failed: %s — content=%r",
            exc, content[:200],
        )
        return dict(_REJECT)

    if not isinstance(parsed, dict) or "is_provenance" not in parsed:
        logger.debug(
            "provenance.classifier: schema mismatch — missing is_provenance key",
        )
        return dict(_REJECT)

    # Coerce confidence to a float in [0, 1]. The model occasionally
    # emits strings or out-of-range values.
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "is_provenance": bool(parsed.get("is_provenance", False)),
        "person": parsed.get("person"),
        "relation_type": parsed.get("relation_type"),
        "quote": parsed.get("quote"),
        "confidence": confidence,
    }
