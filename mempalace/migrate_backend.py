"""
migrate_backend.py — Cross-backend drawer migration (e.g. ChromaDB -> Qdrant).

No cross-backend migration tool ships in mempalace: the `migrate` CLI only does
ChromaDB *version* migration. Cross-backend means standing up the target empty
and re-ingesting every drawer through ``BaseCollection`` (which re-embeds via
the configured embedder), then validating parity.

Hard guarantees:

* **READ-ONLY on source.** Only ``get`` / ``count`` are ever called on the
  source collection. The source palace is never mutated or deleted. (The live
  ChromaDB palace is the rollback path during the Qdrant cutover.)
* **Payload-index verification by query.** Qdrant filters drawer metadata under
  the ``metadata`` payload key, so fast filtered search at scale needs payload
  indexes on ``metadata.wing`` / ``metadata.room`` / ``metadata.filed_at_ts``.
  The backend's ``create_payload_index`` swallows HTTP 400 *and* 409, so a
  "called" index may be silently absent. After creating them we re-read the
  collection info and **fail loud** if any is missing — discharge by
  verification, not by detection.
* **Batched + resumable.** Designed for millions of drawers: enumeration is
  paginated, writes are batched, and an optional checkpoint file lets a run
  resume. Writes use ``upsert`` (keyed by the preserved drawer id), so re-runs
  are idempotent — no duplication.

KG (``knowledge_graph.sqlite3``), hallways, and tunnels are backend-independent
and are NOT migrated here; verify post-cutover that they still resolve.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

# Qdrant nests drawer metadata under the `metadata` payload key (see
# backends/qdrant.py `_condition`: key = f"{_PAYLOAD_METADATA}.{field}"), so the
# filter indexes must target metadata.<field>. wing/room are keyword; the
# numeric recency field filed_at_ts is float.
REQUIRED_PAYLOAD_INDEXES: dict[str, str] = {
    "metadata.wing": "keyword",
    "metadata.room": "keyword",
    "metadata.filed_at_ts": "float",
}


class MigrationIndexError(RuntimeError):
    """Raised when a required payload index is absent after creation —
    create_payload_index swallowed a failure and the index does not exist."""


@dataclass
class MigrationResult:
    source_count: int
    migrated: int
    per_wing_source: dict[str, int] = field(default_factory=dict)
    per_wing_target: dict[str, int] = field(default_factory=dict)
    indexes_verified: dict[str, bool] = field(default_factory=dict)


# ── result-shape helpers (GetResult may be a dict or a dataclass) ────────


def _field(result, name):
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


# ── enumeration (read-only, paginated for millions) ──────────────────────


def enumerate_source_ids(source_col, batch_size: int = 512) -> Iterator[list[str]]:
    """Yield drawer ids from the source collection in pages, read-only.

    Pagination is offset-based; the source is never written, so its ordering is
    stable across a resumed run.
    """
    offset = 0
    while True:
        result = source_col.get(limit=batch_size, offset=offset, include=[])
        ids = _field(result, "ids") or []
        if not ids:
            return
        yield list(ids)
        offset += len(ids)


# ── payload-index ensure + verify (the load-bearing safety check) ────────


def ensure_and_verify_payload_indexes(qdrant_collection) -> dict[str, bool]:
    """Create the required payload indexes on the qdrant collection, then VERIFY
    each one actually exists by re-reading collection info. Raises
    MigrationIndexError if any is absent.

    ``qdrant_collection`` is the underlying QdrantCollection (unwrapped from any
    EmbeddingCollection), exposing ``_client`` and ``_remote_collection``.
    """
    client = qdrant_collection._client
    remote = qdrant_collection._remote_collection

    for field_name, schema in REQUIRED_PAYLOAD_INDEXES.items():
        client.create_payload_index(remote, field_name, schema)

    info = client.get_collection_info(remote)
    result = _field(info, "result") or info
    payload_schema = (_field(result, "payload_schema") or {}) if result else {}
    verified = {f: (f in payload_schema) for f in REQUIRED_PAYLOAD_INDEXES}
    missing = [f for f, ok in verified.items() if not ok]
    if missing:
        raise MigrationIndexError(
            f"payload indexes absent after creation: {missing} "
            f"(present: {sorted(payload_schema)}). create_payload_index swallows "
            f"400/409, so this is a real rejection — not benign."
        )
    return verified


def _unwrap_qdrant(collection):
    """Return the underlying QdrantCollection from an EmbeddingCollection (or the
    collection itself if not wrapped)."""
    inner = getattr(collection, "_inner", None)
    return inner if inner is not None else collection


# ── checkpoint (optional, for resumable runs over huge palaces) ──────────


def _load_checkpoint(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path) as fh:
            return int(json.load(fh).get("completed_offset", 0))
    except (ValueError, OSError, json.JSONDecodeError):
        return 0


def _save_checkpoint(path: Optional[str], completed_offset: int, source_count: int) -> None:
    if not path:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump({"completed_offset": completed_offset, "source_count": source_count}, fh)
    os.replace(tmp, path)


# ── migration ────────────────────────────────────────────────────────────


def migrate(
    source_palace: str,
    target_palace: str,
    *,
    collection_name: Optional[str] = None,
    batch_size: int = 512,
    wings: Optional[list[str]] = None,
    checkpoint_path: Optional[str] = None,
    progress=None,
) -> MigrationResult:
    """Migrate the drawer collection from the source (chroma) palace to the
    target (qdrant) palace, re-embedding via the configured embedder.

    ``wings`` — when given, only drawers in those wings are migrated (used for
    bounded dry-runs). Counts in ``per_wing_source`` still tally the full source.
    """
    from mempalace.palace import get_collection

    source = get_collection(source_palace, collection_name, create=False, backend="chroma")
    target = get_collection(target_palace, collection_name, create=True, backend="qdrant")
    wing_filter = set(wings) if wings else None

    per_wing_source: dict[str, int] = {}
    per_wing_target: dict[str, int] = {}
    migrated = 0
    source_count = 0
    start_offset = _load_checkpoint(checkpoint_path)
    offset = 0

    for batch_ids in enumerate_source_ids(source, batch_size=batch_size):
        batch_start = offset
        offset += len(batch_ids)
        source_count += len(batch_ids)

        fetched = source.get(ids=batch_ids, include=["documents", "metadatas"])
        ids = _field(fetched, "ids") or []
        documents = _field(fetched, "documents") or []
        metadatas = _field(fetched, "metadatas") or [{} for _ in ids]

        # tally per-wing source counts over the full source
        for meta in metadatas:
            w = (meta or {}).get("wing", "")
            per_wing_source[w] = per_wing_source.get(w, 0) + 1

        # resume: skip batches already completed in a prior run
        if batch_start + len(ids) <= start_offset:
            continue

        # select rows to migrate (wing filter)
        sel_docs, sel_ids, sel_meta = [], [], []
        for d, i, m in zip(documents, ids, metadatas):
            w = (m or {}).get("wing", "")
            if wing_filter is not None and w not in wing_filter:
                continue
            sel_docs.append(d)
            sel_ids.append(i)
            sel_meta.append(m)
            per_wing_target[w] = per_wing_target.get(w, 0) + 1

        if sel_ids:
            # upsert (not add): idempotent on re-run, keyed by preserved drawer id.
            target.upsert(documents=sel_docs, ids=sel_ids, metadatas=sel_meta)
            migrated += len(sel_ids)

        _save_checkpoint(checkpoint_path, offset, source_count)
        if progress:
            progress(migrated, source_count)

    # Create + VERIFY payload indexes once the collection exists (post-write).
    indexes_verified: dict[str, bool] = {}
    if migrated > 0:
        indexes_verified = ensure_and_verify_payload_indexes(_unwrap_qdrant(target))

    return MigrationResult(
        source_count=source_count,
        migrated=migrated,
        per_wing_source=per_wing_source,
        per_wing_target=per_wing_target,
        indexes_verified=indexes_verified,
    )
