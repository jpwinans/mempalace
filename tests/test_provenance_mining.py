"""Tests for mempalace.provenance.mining — Phase 1 D3.

Covers:
  - mine_chunk_for_provenance happy path (mock classifier above
    threshold -> wing_lineage drawer written).
  - Confidence threshold (below threshold -> no drawer).
  - Dedup (same chunk + source twice -> one drawer).
  - Disabled mode (MEMPALACE_PROVENANCE_DISABLED=1 -> no extraction).
  - Failure-soft (classifier raising -> 0 drawers, no exception).
  - Transitive-attribution rewrite (classifier returns 'James' for
    "his father's saying" text -> drawer filed under room='father').
  - convo_miner integration: chunk filing produces both operational
    AND wing_lineage drawers.
"""

from __future__ import annotations

from typing import Any

import pytest

from mempalace.provenance.mining import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    mine_chunk_for_provenance,
    _rewrite_speaker_to_source,
)


# ---------------------------------------------------------------------------
# Fake chromadb collection — captures upserts + supports get for dedup
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Just enough chromadb surface for mine_chunk_for_provenance."""

    def __init__(self):
        self.upserts: list[dict[str, Any]] = []
        self._ids: set[str] = set()

    def upsert(self, *, documents, ids, metadatas):
        assert len(documents) == len(ids) == len(metadatas)
        for doc, _id, meta in zip(documents, ids, metadatas):
            self.upserts.append({"id": _id, "doc": doc, "meta": meta})
            self._ids.add(_id)

    def get(self, ids=None, include=None):  # noqa: ARG002
        if ids is None:
            return {"ids": list(self._ids)}
        present = [i for i in ids if i in self._ids]
        return {"ids": present}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _accept_classifier(person: str = "father", confidence: float = 0.92):
    """Build a custom classifier that accepts with the given fields."""

    def classifier(ctx: str) -> dict:
        return {
            "is_provenance": True,
            "person": person,
            "relation_type": "family",
            "quote": "measure twice, cut once",
            "confidence": confidence,
        }

    return classifier


def test_writes_wing_lineage_drawer_on_accept(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    written = mine_chunk_for_provenance(
        col,
        chunk_content="My father said 'measure twice, cut once'",
        source_file="/tmp/session-A.jsonl",
        classifier=_accept_classifier(),
    )
    assert written == 1
    assert len(col.upserts) == 1
    drawer = col.upserts[0]
    assert drawer["meta"]["wing"] == "wing_lineage"
    assert drawer["meta"]["room"] == "father"
    assert drawer["meta"]["relation_type"] == "family"
    assert drawer["meta"]["confidence"] == 0.92
    assert drawer["meta"]["source_file"] == "/tmp/session-A.jsonl"
    # Content rendered per design doc §D3 schema.
    assert "PROVENANCE:" in drawer["doc"]
    assert "Person: father" in drawer["doc"]
    assert "Relation: family" in drawer["doc"]
    assert "measure twice, cut once" in drawer["doc"]


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

def test_below_threshold_no_drawer(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    written = mine_chunk_for_provenance(
        col,
        chunk_content="My father said 'measure twice, cut once'",
        source_file="/tmp/x.jsonl",
        # 0.5 is below the 0.7 default threshold.
        classifier=_accept_classifier(confidence=0.5),
    )
    assert written == 0
    assert col.upserts == []


def test_custom_threshold(monkeypatch):
    """Lowering the threshold lets a 0.6 result through."""

    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    written = mine_chunk_for_provenance(
        col,
        chunk_content="My father said 'measure twice, cut once'",
        source_file="/tmp/x.jsonl",
        classifier=_accept_classifier(confidence=0.6),
        confidence_threshold=0.5,
    )
    assert written == 1


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_dedup_same_chunk_same_source_twice(monkeypatch):
    """Re-mining the same source file shouldn't produce duplicate
    wing_lineage drawers — the dedupe key is
    (person, quote, source_file) hashed into the drawer_id."""

    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    chunk = "My father said 'measure twice, cut once'"
    written1 = mine_chunk_for_provenance(
        col, chunk_content=chunk, source_file="/tmp/A.jsonl",
        classifier=_accept_classifier(),
    )
    written2 = mine_chunk_for_provenance(
        col, chunk_content=chunk, source_file="/tmp/A.jsonl",
        classifier=_accept_classifier(),
    )
    assert written1 == 1
    assert written2 == 0
    assert len(col.upserts) == 1


def test_different_sources_produce_distinct_drawers(monkeypatch):
    """Same attribution found in two different source files SHOULD
    produce two drawers — distinct events to track separately."""

    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    chunk = "My father said 'measure twice, cut once'"
    mine_chunk_for_provenance(
        col, chunk_content=chunk, source_file="/tmp/A.jsonl",
        classifier=_accept_classifier(),
    )
    mine_chunk_for_provenance(
        col, chunk_content=chunk, source_file="/tmp/B.jsonl",
        classifier=_accept_classifier(),
    )
    assert len(col.upserts) == 2
    assert col.upserts[0]["id"] != col.upserts[1]["id"]


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------

def test_disabled_via_env_returns_zero(monkeypatch):
    monkeypatch.setenv("MEMPALACE_PROVENANCE_DISABLED", "1")
    col = _FakeCollection()
    written = mine_chunk_for_provenance(
        col,
        chunk_content="My father said 'measure twice, cut once'",
        source_file="/tmp/x.jsonl",
        classifier=_accept_classifier(),  # would otherwise accept
    )
    assert written == 0
    assert col.upserts == []


def test_disabled_via_env_truthy_variants(monkeypatch):
    for value in ("true", "True", "yes", "1"):
        monkeypatch.setenv("MEMPALACE_PROVENANCE_DISABLED", value)
        col = _FakeCollection()
        written = mine_chunk_for_provenance(
            col,
            chunk_content="My father said 'be still'",
            source_file="/tmp/x.jsonl",
            classifier=_accept_classifier(),
        )
        assert written == 0, f"value={value!r} should disable"


# ---------------------------------------------------------------------------
# No candidates / no extraction
# ---------------------------------------------------------------------------

def test_no_candidates_returns_zero(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()
    written = mine_chunk_for_provenance(
        col,
        chunk_content="Operational content with no attribution markers.",
        source_file="/tmp/x.jsonl",
        classifier=_accept_classifier(),
    )
    assert written == 0
    assert col.upserts == []


# ---------------------------------------------------------------------------
# Failure-soft on classifier exception
# ---------------------------------------------------------------------------

def test_classifier_exception_yields_zero_no_crash(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()

    def broken(ctx: str) -> dict:
        raise RuntimeError("simulated classifier failure")

    written = mine_chunk_for_provenance(
        col,
        chunk_content="My father said 'measure twice'",
        source_file="/tmp/x.jsonl",
        classifier=broken,
    )
    assert written == 0
    assert col.upserts == []


# ---------------------------------------------------------------------------
# Transitive-attribution rewrite
# ---------------------------------------------------------------------------

def test_transitive_attribution_rewrites_speaker_to_source(monkeypatch):
    """Architect-flagged case #11: classifier returns person='James' for
    "Tonight James reminded me: 'measure twice' — his father's saying";
    rewrite must redirect to room='father'."""

    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)
    col = _FakeCollection()

    def classifier(ctx: str) -> dict:
        return {
            "is_provenance": True,
            "person": "James",
            "relation_type": "family",
            "quote": "measure twice, cut once",
            "confidence": 0.9,
        }

    # The text mentions James (the speaker) but the actual source is
    # his father — surfaces as "his father's saying" in the text.
    chunk = (
        "Tonight James reminded me: 'measure twice, cut once' "
        "— his father's saying. I had been carrying it as my own."
    )
    written = mine_chunk_for_provenance(
        col, chunk_content=chunk, source_file="/tmp/x.jsonl",
        classifier=classifier,
    )
    assert written == 1
    drawer = col.upserts[0]
    # Rewrite should redirect James -> father.
    assert drawer["meta"]["room"] == "father", (
        f"Expected room=father after transitive rewrite, got {drawer['meta']['room']!r}"
    )
    assert "Person: father" in drawer["doc"]


def test_rewrite_helper_returns_relation_when_possessive_source_present():
    """Unit test on _rewrite_speaker_to_source directly."""

    assert _rewrite_speaker_to_source(
        "James", "James reminded me of it",
        "Tonight James reminded me: 'X' — his father's saying.",
    ) == "father"


def test_rewrite_helper_returns_input_when_no_possessive_source():
    assert _rewrite_speaker_to_source(
        "father", "my father said 'X'", "my father said 'X'",
    ) == "father"


def test_rewrite_helper_preserves_none():
    assert _rewrite_speaker_to_source(
        None, "some text", "some context"
    ) is None


# ---------------------------------------------------------------------------
# convo_miner integration: chunk filing produces both wings
# ---------------------------------------------------------------------------

def test_convo_miner_integration_produces_both_operational_and_lineage_drawers(
    monkeypatch, tmp_path,
):
    """End-to-end via convo_miner._file_chunks_locked. Mock the
    classifier (so we don't need substrate) + provide a chunk with
    a provenance-bearing line. Assert both operational drawer (wing
    metadata == 'wing_test') and lineage drawer (wing_lineage)
    are upserted."""

    monkeypatch.delenv("MEMPALACE_PROVENANCE_DISABLED", raising=False)

    # Mock the qwen3 classifier — the production code path imports
    # it lazily, so patching the module attribute reaches the lazy
    # import inside mine_chunk_for_provenance.
    def fake_classifier(ctx: str) -> dict:
        return {
            "is_provenance": True,
            "person": "father",
            "relation_type": "family",
            "quote": "measure twice, cut once",
            "confidence": 0.9,
        }

    monkeypatch.setattr(
        "mempalace.provenance.classifier.qwen3_classifier", fake_classifier
    )

    # Patch mine_lock + file_already_mined since they require a real
    # palace path / state on disk for full integration. We only care
    # that _file_chunks_locked's loop reaches the provenance hook.
    monkeypatch.setattr(
        "mempalace.convo_miner.mine_lock",
        lambda source: _NullContext(),
    )
    monkeypatch.setattr(
        "mempalace.convo_miner.file_already_mined", lambda col, src: False
    )

    from mempalace.convo_miner import _file_chunks_locked

    col = _FakeCollection()
    chunks = [
        {
            "content": "My father said 'measure twice, cut once' yesterday.",
            "chunk_index": 0,
        },
    ]
    drawers_added, _room_delta, skipped = _file_chunks_locked(
        collection=col,
        source_file=str(tmp_path / "session.jsonl"),
        chunks=chunks,
        wing="wing_test",
        room="general",
        agent="mempalace",
        extract_mode="exchange",
    )

    assert drawers_added == 1
    assert skipped is False

    wings = {u["meta"]["wing"] for u in col.upserts}
    assert "wing_test" in wings, f"missing operational drawer: {wings}"
    assert "wing_lineage" in wings, f"missing lineage drawer: {wings}"


class _NullContext:
    """Stand-in for mine_lock's context-manager interface."""

    def __enter__(self): return self
    def __exit__(self, *a): return False
