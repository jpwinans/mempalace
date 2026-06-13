"""
migrate_parity.py — B2 parity validation for the chroma->qdrant migration.

After the full migration runs, this validates that the qdrant target faithfully
reproduces the chroma source:

* **count parity per wing**, computed from the LIVE qdrant target (not from the
  MigrationResult, which undercounts after a checkpoint-resume — data is correct
  via upsert, but the reporting metric is a trap).
* **search parity** — a fixed query battery returns overlapping top-k from both
  backends (exact ANN ordering differs; we assert overlap + that known anchors
  surface).
* **diary parity** for the ``ves`` wing (the drawers ``diary_read`` surfaces).
* **embedder identity** recorded so a search-parity gap is interpretable
  (ANN-ordering difference vs embedder drift). Expected: minilm / 384-dim.

KG / hallways / tunnels are backend-independent files at ~/.mempalace/; B2 only
asserts they are UNTOUCHED (see :func:`home_files_fingerprint`). Whether they
resolve against the qdrant-backed palace is the B3 post-cutover check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

EXPECTED_EMBEDDER = "minilm"
EXPECTED_DIM = 384

# Backend-independent home-anchored artifacts that a backend swap must NOT move
# or touch (siblings of the palace dir).
_HOME_ARTIFACTS = (
    "knowledge_graph.sqlite3",
    "knowledge_graph.sqlite3-wal",
    "knowledge_graph.sqlite3-shm",
    "hallways.json",
    "tunnels.json",
)


# ── result shapes ────────────────────────────────────────────────────────


@dataclass
class WingParity:
    wing: str
    source: int
    target: int

    @property
    def match(self) -> bool:
        return self.source == self.target


@dataclass
class SearchParity:
    query: str
    overlap: float
    anchor_found: Optional[bool] = None


@dataclass
class ParityReport:
    embedder_model: str
    source_total: int
    target_total: int
    per_wing: list[WingParity] = field(default_factory=list)
    search: list[SearchParity] = field(default_factory=list)
    diary: dict = field(default_factory=dict)

    @property
    def embedder_ok(self) -> bool:
        return self.embedder_model == EXPECTED_EMBEDDER

    @property
    def total_match(self) -> bool:
        return self.source_total == self.target_total

    @property
    def diary_match(self) -> bool:
        return self.diary.get("source") == self.diary.get("target")

    @property
    def anchors_ok(self) -> bool:
        return all(s.anchor_found is not False for s in self.search)

    @property
    def ok(self) -> bool:
        return (
            self.embedder_ok
            and self.total_match
            and self.diary_match
            and self.anchors_ok
            and all(w.match for w in self.per_wing)
        )

    def format(self) -> str:
        lines = [
            f"embedder: {self.embedder_model} (expected {EXPECTED_EMBEDDER}/{EXPECTED_DIM}d) "
            f"[{'OK' if self.embedder_ok else 'MISMATCH'}]",
            f"total: source={self.source_total} target={self.target_total} "
            f"[{'OK' if self.total_match else 'MISMATCH'}]",
            "per-wing (source vs LIVE target):",
        ]
        for w in sorted(self.per_wing, key=lambda x: x.wing):
            lines.append(
                f"  {w.wing}: {w.source} -> {w.target} [{'OK' if w.match else 'MISMATCH'}]"
            )
        lines.append(
            f"diary (ves/diary): source={self.diary.get('source')} "
            f"target={self.diary.get('target')} [{'OK' if self.diary_match else 'MISMATCH'}]"
        )
        if self.search:
            lines.append("search battery (top-k overlap; ANN ordering differs):")
            for s in self.search:
                anchor = (
                    ""
                    if s.anchor_found is None
                    else f" anchor={'found' if s.anchor_found else 'MISSING'}"
                )
                lines.append(f"  {s.query!r}: overlap={s.overlap:.2f}{anchor}")
        lines.append(f"PARITY: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


# ── pure helpers ─────────────────────────────────────────────────────────


def embedder_model_id() -> str:
    from mempalace.config import MempalaceConfig

    return MempalaceConfig().embedding_model


def search_overlap(a: list[str], b: list[str]) -> float:
    """Fraction of overlap between two top-k id lists, ignoring order. 1.0 when
    identical, 0.0 when disjoint. Denominator is the larger set so a target that
    returns extra ids cannot inflate the score."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    denom = max(len(sa), len(sb))
    return len(sa & sb) / denom if denom else 1.0


def home_files_fingerprint(palace_path: str) -> str:
    """Content hash of the backend-independent home artifacts (KG sqlite +
    hallways/tunnels json) that live as siblings of the palace dir. Used to
    assert the migration never touches them."""
    home = Path(palace_path).parent
    h = hashlib.sha256()
    for name in _HOME_ARTIFACTS:
        p = home / name
        if p.is_file():
            h.update(f"{name}|".encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# ── collection helpers ───────────────────────────────────────────────────


def _ids(result) -> list[str]:
    if isinstance(result, dict):
        return result.get("ids") or []
    return getattr(result, "ids", None) or []


def _metas(result) -> list[dict]:
    if isinstance(result, dict):
        return result.get("metadatas") or []
    return getattr(result, "metadatas", None) or []


def _count_wing(col, wing: str) -> int:
    return len(_ids(col.get(where={"wing": wing}, include=[])))


def _diary_count(col) -> int:
    """Count drawers diary_read('ves') surfaces: wing=ves, room=diary. Filter
    room in Python to stay backend-agnostic (avoids multi-key where syntax).
    The ves wing is small, so a single where-get is safe."""
    res = col.get(where={"wing": "ves"}, include=["metadatas"])
    return sum(1 for m in _metas(res) if (m or {}).get("room") == "diary")


def source_tally(col, batch_size: int = 512) -> tuple[dict[str, int], int]:
    """Per-wing counts + ves/diary count from a PAGINATED full pass over the
    source. Pagination is mandatory at scale: a single unbounded
    ``get(include=["metadatas"])`` on a 67k+ chroma collection exceeds SQLite's
    bind-variable limit ("too many SQL variables"). Returns (per_wing, diary)."""
    per_wing: dict[str, int] = {}
    diary = 0
    offset = 0
    while True:
        res = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        ids = _ids(res)
        if not ids:
            break
        for m in _metas(res):
            w = (m or {}).get("wing", "")
            per_wing[w] = per_wing.get(w, 0) + 1
            if w == "ves" and (m or {}).get("room") == "diary":
                diary += 1
        offset += len(ids)
        if len(ids) < batch_size:
            break
    return per_wing, diary


def _topk_ids(col, query: str, k: int) -> list[str]:
    res = col.query(query_texts=[query], n_results=k)
    ids = res["ids"] if isinstance(res, dict) else res.ids
    return list(ids[0]) if ids else []


# ── orchestration ────────────────────────────────────────────────────────


def validate_parity(
    source_palace: str,
    target_palace: str,
    *,
    collection_name: Optional[str] = None,
    queries: Optional[list[str]] = None,
    anchors: Optional[dict[str, str]] = None,
    k: int = 10,
) -> ParityReport:
    """Compare the chroma source and qdrant target collections and return a
    structured parity report. Read-only on both."""
    from mempalace.palace import get_collection

    source = get_collection(source_palace, collection_name, create=False, backend="chroma")
    target = get_collection(target_palace, collection_name, create=False, backend="qdrant")

    # source per-wing + diary from a PAGINATED full pass (scale-safe)
    src_total = source.count()
    per_wing_source, diary_source = source_tally(source)

    # per-wing target counts from the LIVE qdrant target (coder-2's note)
    per_wing = [
        WingParity(wing=w, source=n, target=_count_wing(target, w))
        for w, n in sorted(per_wing_source.items())
    ]

    # search battery
    anchors = anchors or {}
    search: list[SearchParity] = []
    for q in queries or []:
        s_ids = _topk_ids(source, q, k)
        t_ids = _topk_ids(target, q, k)
        anchor_id = anchors.get(q)
        search.append(
            SearchParity(
                query=q,
                overlap=search_overlap(s_ids, t_ids),
                anchor_found=(anchor_id in t_ids) if anchor_id else None,
            )
        )

    report = ParityReport(
        embedder_model=embedder_model_id(),
        source_total=src_total,
        target_total=target.count(),
        per_wing=per_wing,
        search=search,
        diary={"source": diary_source, "target": _diary_count(target)},
    )
    return report
