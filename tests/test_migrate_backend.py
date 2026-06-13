"""
test_migrate_backend.py — TDD for the cross-backend drawer migration tool (B1).

Migrates the drawer collection from one backend to another (chroma -> qdrant)
by re-`add`/`upsert` through BaseCollection (re-embedding via the configured
embedder). Hard guarantees under test:

  - READ-ONLY on source: the source palace is never mutated or deleted.
  - Payload-index verification BY QUERY: after creating the wing/room/
    filed_at_ts payload indexes, the tool re-reads qdrant collection info and
    fails loud if any index is absent — because create_payload_index swallows
    HTTP 400/409 ("called" != "exists").
  - count + metadata parity per wing.
  - idempotent + resumable (re-run does not duplicate).

Integration tests require a live qdrant on localhost:6333 (skipped if absent).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mempalace.migrate_backend import (
    REQUIRED_PAYLOAD_INDEXES,
    MigrationIndexError,
    enumerate_source_ids,
    ensure_and_verify_payload_indexes,
    migrate,
)


# ── pure logic ───────────────────────────────────────────────────────────


def test_required_indexes_are_metadata_nested_with_correct_schemas():
    # qdrant nests drawer metadata under the `metadata` payload key, so filter
    # indexes must target metadata.<field>, matching the backend's filter path.
    assert REQUIRED_PAYLOAD_INDEXES["metadata.wing"] == "keyword"
    assert REQUIRED_PAYLOAD_INDEXES["metadata.room"] == "keyword"
    assert REQUIRED_PAYLOAD_INDEXES["metadata.filed_at_ts"] == "float"


def test_enumerate_source_ids_paginates_then_stops():
    class FakeSource:
        def __init__(self, all_ids, page):
            self.all_ids = all_ids
            self.page = page
            self.calls = []

        def get(self, *, ids=None, where=None, limit=None, offset=None, include=None):
            self.calls.append((limit, offset))
            chunk = self.all_ids[offset : offset + limit]
            return {"ids": chunk}

    src = FakeSource([f"d{i}" for i in range(7)], page=3)
    batches = list(enumerate_source_ids(src, batch_size=3))
    assert [len(b) for b in batches] == [3, 3, 1]
    assert sum(batches, []) == [f"d{i}" for i in range(7)]
    # read-only: only .get was called, never add/upsert/delete
    assert all(call[0] == 3 for call in src.calls)


# ── payload-index verification (the coder-2 finding guard) ───────────────


class _FakeQdrantClient:
    """Stand-in for _QdrantRESTClient: records create calls, returns a
    controllable payload_schema from get_collection_info."""

    def __init__(self, schema_present):
        self.schema_present = set(schema_present)
        self.created = []

    def create_payload_index(self, collection, field_name, field_schema):
        self.created.append((field_name, field_schema))
        # mimic the real client: swallow failures silently (the bug we guard against)

    def get_collection_info(self, collection):
        return {"result": {"payload_schema": {f: {"data_type": "x"} for f in self.schema_present}}}


class _FakeQdrantCollection:
    def __init__(self, client, remote="remote-coll"):
        self._client = client
        self._remote_collection = remote


def test_verify_raises_when_an_index_is_absent_after_creation():
    # create_payload_index "succeeds" (swallowed), but the index never appears
    # in payload_schema → must fail loud, not pass silently.
    client = _FakeQdrantClient(
        schema_present=["metadata.wing", "metadata.room"]
    )  # filed_at_ts missing
    col = _FakeQdrantCollection(client)
    with pytest.raises(MigrationIndexError) as exc:
        ensure_and_verify_payload_indexes(col)
    assert "metadata.filed_at_ts" in str(exc.value)
    # it did attempt to create all three
    assert {f for f, _ in client.created} == set(REQUIRED_PAYLOAD_INDEXES)


def test_verify_passes_when_all_indexes_present():
    client = _FakeQdrantClient(schema_present=list(REQUIRED_PAYLOAD_INDEXES))
    col = _FakeQdrantCollection(client)
    verified = ensure_and_verify_payload_indexes(col)
    assert verified == {f: True for f in REQUIRED_PAYLOAD_INDEXES}


# ── integration: real qdrant + a real chroma source palace ───────────────

QDRANT_URL = os.environ.get("MEMPALACE_QDRANT_URL", "http://localhost:6333")


def _qdrant_live() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_live(), reason="qdrant not running on localhost:6333"
)


def _hnsw_fingerprint(path: str) -> str:
    """Content hash of every HNSW index file (*.bin) under the palace. This is
    the guardrail that matters: the TB-bloat pathology is HNSW link_lists.bin
    growth. Opening a chroma collection writes benign idempotent markers
    (.blob_seq_ids_migrated, .collection_type_fixed, a small sqlite touch), but
    a read must NEVER mutate the HNSW index.
    """
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*.bin")):
        if p.is_file():
            h.update(f"{p.relative_to(path)}|".encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _source_data_snapshot(palace: str):
    """The real read-only guarantee: every drawer's id, document, and metadata
    is unchanged. Returns (count, {id: (document, sorted-metadata-tuple)})."""
    from mempalace.palace import get_collection

    col = get_collection(palace, "mempalace_drawers", create=False, backend="chroma")
    g = col.get(include=["documents", "metadatas"])
    ids = (g["ids"] if isinstance(g, dict) else g.ids) or []
    docs = (g["documents"] if isinstance(g, dict) else g.documents) or []
    metas = (g["metadatas"] if isinstance(g, dict) else g.metadatas) or []
    snap = {i: (d, tuple(sorted((m or {}).items()))) for i, d, m in zip(ids, docs, metas)}
    return col.count(), snap


def _seed_chroma_source(palace_path: str, drawers: list[dict]) -> int:
    from mempalace.palace import get_collection

    col = get_collection(palace_path, "mempalace_drawers", create=True, backend="chroma")
    col.add(
        documents=[d["content"] for d in drawers],
        ids=[d["id"] for d in drawers],
        metadatas=[d["meta"] for d in drawers],
    )
    return col.count()


_SAMPLE = [
    {
        "id": "drawer_ves_sessions_a",
        "content": "alpha content about migration",
        "meta": {"wing": "ves_sessions", "room": "technical", "filed_at_ts": 1000.0},
    },
    {
        "id": "drawer_ves_sessions_b",
        "content": "beta content about palaces",
        "meta": {"wing": "ves_sessions", "room": "planning", "filed_at_ts": 2000.0},
    },
    {
        "id": "drawer_ves_c",
        "content": "gamma content in ves wing",
        "meta": {"wing": "ves", "room": "diary", "filed_at_ts": 3000.0},
    },
]


@requires_qdrant
def test_migrate_one_wing_reproduces_count_and_metadata(tmp_path):
    source = str(tmp_path / "src_palace")
    target = str(tmp_path / "tgt_palace")
    _seed_chroma_source(source, _SAMPLE)
    data_before = _source_data_snapshot(source)
    hnsw_before = _hnsw_fingerprint(source)

    result = migrate(source, target, wings=["ves_sessions"], batch_size=2)

    # count parity for the migrated wing (2 drawers in ves_sessions)
    assert result.per_wing_source["ves_sessions"] == 2
    assert result.per_wing_target["ves_sessions"] == 2
    assert result.migrated == 2
    # metadata preserved verbatim in qdrant
    from mempalace.palace import get_collection

    tgt = get_collection(target, "mempalace_drawers", create=False, backend="qdrant")
    got = tgt.get(ids=["drawer_ves_sessions_a"], include=["metadatas"])
    meta = (got["metadatas"] if isinstance(got, dict) else got.metadatas)[0]
    assert meta["wing"] == "ves_sessions"
    assert meta["room"] == "technical"
    # SOURCE READ-ONLY (hard guardrail): drawer DATA unchanged + HNSW index
    # untouched. (ChromaDB writes benign idempotent open-time markers on read;
    # the guarantee that matters is no data mutation + no HNSW-bloat op.)
    assert _source_data_snapshot(source) == data_before, "source drawer data was mutated"
    assert _hnsw_fingerprint(source) == hnsw_before, "source HNSW index was mutated"


@requires_qdrant
def test_payload_indexes_exist_and_verified_after_migration(tmp_path):
    source = str(tmp_path / "src_palace")
    target = str(tmp_path / "tgt_palace")
    _seed_chroma_source(source, _SAMPLE)

    result = migrate(source, target, batch_size=2)

    assert result.indexes_verified == {f: True for f in REQUIRED_PAYLOAD_INDEXES}


@requires_qdrant
def test_migrate_is_idempotent_on_rerun(tmp_path):
    source = str(tmp_path / "src_palace")
    target = str(tmp_path / "tgt_palace")
    _seed_chroma_source(source, _SAMPLE)

    migrate(source, target, batch_size=2)
    result2 = migrate(source, target, batch_size=2)

    from mempalace.palace import get_collection

    tgt = get_collection(target, "mempalace_drawers", create=False, backend="qdrant")
    assert tgt.count() == 3  # no duplication on re-run
    assert result2.source_count == 3
