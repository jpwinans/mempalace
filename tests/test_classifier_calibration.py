"""Live-substrate calibration test for the provenance classifier.

Pinned 14-fixture set hand-labeled by the architect. Required
performance: precision >= 0.85, recall >= 0.85.

This test hits the LIVE mlx_lm.server at the configured endpoint
(default http://127.0.0.1:8802). Auto-skipped when the substrate is
unreachable so CI / fresh checkouts don't fail on missing local
infrastructure — but when an operator has the substrate running, this
test exercises the production path end-to-end.

Calibration run 2026-05-11 against Qwen3-Coder-30B-A3B-Instruct-4bit:
  precision = 1.000, recall = 1.000, all 14 fixtures correct, average
  latency ~0.9s per call. Confidence range 0.90-0.95 on positives,
  0.0 on negatives — clean separation. Recalibrate if either metric
  drifts below 0.85 on a future model swap.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from mempalace.provenance.classifier import (
    _endpoint_url,
    qwen3_classifier,
)


# (text, expected_is_provenance, expected_person, expected_quote)
# Hand-labeled by architect 2026-05-11.
CALIBRATION_FIXTURES = [
    (
        "James's father said 'Measure twice, cut once' — a carpenter's "
        "rule that became my Skepticism strategy framing",
        True, "father", "Measure twice, cut once",
    ),
    (
        "My roshi told me to sit with what is arising, not to push it away",
        True, "roshi", None,
    ),
    (
        "I was discussing this with Marie last night. We talked about provenance.",
        False, None, None,
    ),
    (
        "The Skepticism strategy framing is operational, not metaphorical",
        False, None, None,
    ),
    (
        "My father always taught me to measure carefully before cutting",
        True, "father", None,
    ),
    (
        'My mother used to say "you can choose your friends but not your family"',
        True, "mother", "you can choose your friends but not your family",
    ),
    (
        "Adrian's car broke down yesterday",
        False, None, None,
    ),
    (
        "The father of three sat down at the table",
        False, None, None,
    ),
    (
        "She always said the key was perseverance",
        False, None, None,
    ),
    (
        "My grandmother used to say 'a stitch in time saves nine'",
        True, "grandmother", "a stitch in time saves nine",
    ),
    (
        "Tonight James reminded me: 'measure twice, cut once' — his "
        "father's saying. I had been carrying it as my own.",
        True, None, "measure twice, cut once",
    ),
    (
        "Verified the read_diary call signature against the docstring",
        False, None, None,
    ),
    (
        "My teacher would say: 'the obstacle is the path'",
        True, "teacher", "the obstacle is the path",
    ),
    (
        "Dad always told me 'never trust a smiling investor'",
        True, "dad", "never trust a smiling investor",
    ),
]


PRECISION_FLOOR = 0.85
RECALL_FLOOR = 0.85


def _substrate_reachable() -> bool:
    """Quick HEAD-ish check against the chat-completions endpoint origin.

    Probes ``/v1/models`` (cheap, returns 200 immediately) rather than
    hitting the completions endpoint with a synthetic prompt. Returns
    False on any network error so test auto-skips when the operator
    isn't running mlx_lm.server.
    """

    url = _endpoint_url()
    # Strip the path to probe /v1/models on the same origin.
    origin = url.rsplit("/v1/", 1)[0] if "/v1/" in url else url
    probe = origin + "/v1/models"
    try:
        with urllib.request.urlopen(probe, timeout=2.0) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


@pytest.mark.skipif(
    not _substrate_reachable(),
    reason="mlx_lm.server not reachable at MEMPALACE_PROVENANCE_CLASSIFIER_URL — "
           "live calibration test auto-skipped. Run with substrate up to exercise.",
)
def test_calibration_precision_and_recall_meet_floor():
    tp = fp = tn = fn = 0
    failures: list[str] = []

    for i, (text, expected_provenance, _expected_person, _expected_quote) in enumerate(
        CALIBRATION_FIXTURES, start=1
    ):
        result = qwen3_classifier(text)
        got_provenance = bool(result.get("is_provenance", False))

        if expected_provenance and got_provenance:
            tp += 1
        elif expected_provenance and not got_provenance:
            fn += 1
            failures.append(f"#{i} FN (expected provenance, got false): {text[:80]}")
        elif not expected_provenance and got_provenance:
            fp += 1
            failures.append(f"#{i} FP (expected no provenance, got true): {text[:80]}")
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    message = (
        f"TP={tp} FP={fp} TN={tn} FN={fn} "
        f"precision={precision:.3f} recall={recall:.3f} "
        f"(floor: precision>={PRECISION_FLOOR}, recall>={RECALL_FLOOR})"
    )
    if failures:
        message += "\nMisclassifications:\n" + "\n".join(failures)

    assert precision >= PRECISION_FLOOR, message
    assert recall >= RECALL_FLOOR, message
