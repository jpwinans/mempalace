"""Tests for mempalace.provenance — Phase 1 D1.

Covers:
  - extract_candidates: each regex pattern + composite cases + the
    smoke fixtures from the architect's envelope.
  - validate_candidate: stub path (no classifier), custom classifier
    path, rejection path, exception-soft path.
  - The wing_lineage schema doc string is present (regression cover
    against accidental deletion).
"""

from __future__ import annotations

import pytest

from mempalace.provenance import (
    HEURISTIC_CONFIDENCE_QUOTE,
    HEURISTIC_CONFIDENCE_RELATION_ONLY,
    WING_LINEAGE_SCHEMA_DOC,
    ProvenanceCandidate,
    ProvenanceRecord,
    extract_candidates,
    validate_candidate,
)


# ---------------------------------------------------------------------------
# extract_candidates — Pass 1 (relation + attribution + quote)
# ---------------------------------------------------------------------------

def test_extract_pass1_relation_attribution_quote_straight_quotes():
    text = "James's father said 'Measure twice, cut once'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.person_hint == "father"
    assert c.relation_hint == "father"
    assert c.quote == "Measure twice, cut once"
    assert c.confidence_floor == HEURISTIC_CONFIDENCE_QUOTE


def test_extract_pass1_relation_attribution_quote_double_quotes():
    text = 'My mother always said "be patient with yourself"'
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "mother"
    assert candidates[0].quote == "be patient with yourself"


def test_extract_pass1_curly_quotes():
    text = "His grandfather used to say “kindness travels” every winter."
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "grandfather"
    assert candidates[0].quote == "kindness travels"


def test_extract_pass1_told_me_variant():
    text = "My teacher told me \"the breath is enough\""
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "teacher"
    assert candidates[0].quote == "the breath is enough"


def test_extract_pass1_taught_me_variant():
    text = 'My roshi taught me "just sit"'
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "roshi"
    assert candidates[0].quote == "just sit"


def test_extract_pass1_used_to_say_variant():
    text = "My dad used to say 'measure what matters'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "dad"


def test_extract_pass1_named_possessive_marie():
    text = "Marie's brother said \"the work is the practice\""
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "brother"
    assert candidates[0].quote == "the work is the practice"


# ---------------------------------------------------------------------------
# extract_candidates — Pass 2 (relation marker alone)
# ---------------------------------------------------------------------------

def test_extract_pass2_roshi_told_no_quote():
    """Architect smoke fixture: "My roshi told me to sit with what is arising"

    Pass-1 requires a quote and won't match; Pass-2 catches the bare
    relation marker. Quote is None, confidence_floor is the lower value.
    """

    text = "My roshi told me to sit with what is arising"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.person_hint == "roshi"
    assert c.relation_hint == "roshi"
    assert c.quote is None
    assert c.confidence_floor == HEURISTIC_CONFIDENCE_RELATION_ONLY


def test_extract_pass2_bare_my_father():
    text = "I called my father last night."
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "father"
    assert candidates[0].quote is None


def test_extract_pass2_word_boundary_avoids_fatherland():
    """`\\b` boundary must prevent matching "my fatherland" as "my father"."""

    text = "I returned to my fatherland after a long absence."
    candidates = extract_candidates(text)
    assert len(candidates) == 0


# ---------------------------------------------------------------------------
# extract_candidates — Pass 3 (Capitalized bare-relation as subject)
# ---------------------------------------------------------------------------

def test_extract_pass3_dad_always_told_me():
    """Calibration fixture #14: "Dad always told me 'never trust a
    smiling investor'". Pass-1 misses (no possessive prefix); Pass-2
    misses (also requires possessive). Pass-3 catches capitalized-
    bare-relation + attribution + quote."""

    text = "Dad always told me 'never trust a smiling investor'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.person_hint == "dad"
    assert c.relation_hint == "dad"
    assert c.quote == "never trust a smiling investor"
    assert c.confidence_floor == HEURISTIC_CONFIDENCE_QUOTE


def test_extract_pass3_mom_used_to_say():
    text = "Mom used to say 'be kind'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "mom"
    assert candidates[0].quote == "be kind"


def test_extract_pass3_roshi_said():
    text = "Roshi taught me: 'sit with what is arising'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].person_hint == "roshi"


def test_extract_pass3_requires_capitalization():
    """Lowercase "dad" mid-sentence without possessive prefix must NOT
    match Pass-3. The capitalize-only constraint keeps false-positives
    manageable; the D2 classifier catches lowercase cases via context."""

    text = "Some old dad told someone to slow down."
    candidates = extract_candidates(text)
    assert candidates == []


# ---------------------------------------------------------------------------
# extract_candidates — Pass-1 deduplication of Pass-2 overlaps
# ---------------------------------------------------------------------------

def test_pass1_match_suppresses_overlapping_pass2():
    """When Pass-1 matches relation+quote, the Pass-2 relation-alone
    match at the same position must not be reported as a duplicate
    candidate."""

    text = "My father said 'be still'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    assert candidates[0].quote == "be still"
    assert candidates[0].confidence_floor == HEURISTIC_CONFIDENCE_QUOTE


# ---------------------------------------------------------------------------
# extract_candidates — Negative cases (smoke fixtures)
# ---------------------------------------------------------------------------

def test_extract_marie_without_relation_marker_zero_candidates():
    """Architect smoke fixture: "I was discussing this with Marie last night"

    Marie is a proper name but there's no relation marker ("my X" /
    "X's family") and no quote markers. v1 heuristic does not match
    this case — Marie-only references are out of scope until D2's
    classifier sees them.
    """

    text = "I was discussing this with Marie last night."
    candidates = extract_candidates(text)
    assert candidates == []


def test_extract_operational_content_zero_candidates():
    """Architect smoke fixture: "The Skepticism strategy framing"
    (operational content with no attribution)."""

    text = "The Skepticism strategy framing was added to Core Identity v8.3."
    candidates = extract_candidates(text)
    assert candidates == []


def test_extract_empty_text_returns_empty_list():
    assert extract_candidates("") == []


# ---------------------------------------------------------------------------
# extract_candidates — Multi-candidate ordering
# ---------------------------------------------------------------------------

def test_extract_multiple_candidates_sorted_by_position():
    text = (
        "My father always said 'measure twice'. "
        "Later, my roshi told me to be patient."
    )
    candidates = extract_candidates(text)
    assert len(candidates) == 2
    # Sorted by position ascending.
    assert candidates[0].position < candidates[1].position
    assert candidates[0].person_hint == "father"
    assert candidates[1].person_hint == "roshi"


# ---------------------------------------------------------------------------
# validate_candidate — stub classifier (D1 default)
# ---------------------------------------------------------------------------

def test_validate_with_stub_accepts_candidate():
    text = "My father said 'measure twice, cut once'"
    candidates = extract_candidates(text)
    assert len(candidates) == 1
    record = validate_candidate(candidates[0], ctx=text)
    assert record is not None
    assert record.person == "father"
    assert record.relation_type == "father"
    assert record.quote == "measure twice, cut once"
    assert record.context == text
    assert record.confidence == 0.5
    assert record.extracted_by == "heuristic_v1+stub_classifier_v1"


def test_validate_stub_handles_candidate_without_quote():
    text = "My roshi told me to sit with what is arising"
    candidates = extract_candidates(text)
    record = validate_candidate(candidates[0], ctx=text)
    assert record is not None
    assert record.person == "roshi"
    assert record.relation_type == "roshi"
    assert record.quote == ""  # ProvenanceRecord normalizes None -> empty string


# ---------------------------------------------------------------------------
# validate_candidate — custom classifier
# ---------------------------------------------------------------------------

def test_validate_with_custom_classifier_accepts():
    text = "My father said 'be still'"
    candidates = extract_candidates(text)
    ctx = text

    def classifier(context_text: str) -> dict:
        assert context_text == ctx  # context passed through correctly
        return {
            "is_provenance": True,
            "person": "James's father",
            "relation_type": "family",
            "quote": "be still",
            "confidence": 0.92,
        }

    record = validate_candidate(candidates[0], ctx=ctx, classifier=classifier)
    assert record is not None
    assert record.person == "James's father"
    assert record.relation_type == "family"  # classifier overrides heuristic
    assert record.quote == "be still"
    assert record.confidence == 0.92
    assert record.extracted_by == "heuristic_v1+custom_classifier"


def test_validate_with_custom_classifier_rejects():
    text = "My father said 'be still'"
    candidates = extract_candidates(text)

    def classifier(context_text: str) -> dict:
        return {"is_provenance": False, "confidence": 0.1}

    record = validate_candidate(candidates[0], ctx=text, classifier=classifier)
    assert record is None


def test_validate_custom_classifier_label_override():
    text = "My father said 'be still'"
    candidates = extract_candidates(text)

    def classifier(context_text: str) -> dict:
        return {
            "is_provenance": True,
            "person": "father",
            "relation_type": "family",
            "quote": "be still",
            "confidence": 0.8,
        }

    record = validate_candidate(
        candidates[0],
        ctx=text,
        classifier=classifier,
        extractor_label="heuristic_v1+qwen3_classifier_v1",
    )
    assert record is not None
    assert record.extracted_by == "heuristic_v1+qwen3_classifier_v1"


def test_validate_classifier_exception_yields_none():
    """Classifier failure must not crash the mining pipeline."""

    text = "My father said 'be still'"
    candidates = extract_candidates(text)

    def broken_classifier(context_text: str) -> dict:
        raise RuntimeError("simulated classifier failure")

    record = validate_candidate(
        candidates[0], ctx=text, classifier=broken_classifier,
    )
    assert record is None


def test_validate_classifier_returns_non_dict_yields_none():
    """Defense against contract drift from custom classifiers."""

    text = "My father said 'be still'"
    candidates = extract_candidates(text)

    record = validate_candidate(
        candidates[0], ctx=text, classifier=lambda _ctx: "not a dict",  # type: ignore[arg-type]
    )
    assert record is None


def test_validate_rejects_when_person_missing():
    """If classifier accepts but person can't be resolved from result
    or candidate, reject — wing_lineage drawer needs the room key."""

    candidate = ProvenanceCandidate(
        text="some span",
        person_hint=None,
        relation_hint=None,
        quote=None,
        position=0,
        confidence_floor=0.4,
    )

    def classifier(context_text: str) -> dict:
        return {"is_provenance": True, "confidence": 0.8}

    record = validate_candidate(candidate, ctx="ctx", classifier=classifier)
    assert record is None


def test_validate_non_float_confidence_defaults_to_zero():
    text = "My father said 'be still'"
    candidates = extract_candidates(text)

    def classifier(context_text: str) -> dict:
        return {
            "is_provenance": True,
            "person": "father",
            "relation_type": "family",
            "quote": "be still",
            "confidence": "not-a-number",
        }

    record = validate_candidate(candidates[0], ctx=text, classifier=classifier)
    assert record is not None
    assert record.confidence == 0.0


# ---------------------------------------------------------------------------
# Schema doc presence (regression cover)
# ---------------------------------------------------------------------------

def test_wing_lineage_schema_doc_present():
    """Regression cover: the schema doc string must not be accidentally
    deleted. Downstream D3 implementation references it for drawer
    construction; design-doc drift between code and doc is hard to
    catch otherwise."""

    assert "wing_lineage" in WING_LINEAGE_SCHEMA_DOC
    assert "room: <person_label>" in WING_LINEAGE_SCHEMA_DOC
    assert "PROVENANCE:" in WING_LINEAGE_SCHEMA_DOC


# ---------------------------------------------------------------------------
# End-to-end integration — real-shape diary fixture
# ---------------------------------------------------------------------------

def test_integration_realistic_diary_chunk_yields_validated_record():
    """End-to-end on a sample text shape that real diary drawers produce.

    Goes through both extract_candidates and validate_candidate to
    confirm the pipeline composes cleanly without external state.
    """

    sample = (
        "SESSION:2026-05-10|v1.0.0-cutover+skepticism|"
        "I shipped Vestige Control v1.0.0 today. "
        "James shared his father's saying — 'Measure twice, cut once' — "
        "and we used that as the framing for the Skepticism strategy."
    )
    candidates = extract_candidates(sample)
    # Should pick up "father's saying" via Pass-1 (named possessive variant).
    assert len(candidates) >= 1
    record = validate_candidate(candidates[0], ctx=sample)
    assert record is not None
    assert record.person == "father"
    assert record.context == sample
    assert record.extracted_by == "heuristic_v1+stub_classifier_v1"
