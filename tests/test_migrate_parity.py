"""
test_migrate_parity.py — TDD for B2 parity validation of the chroma->qdrant migration.

After the full migration runs, parity is validated against the LIVE target
(per coder-2's review note: count from the live qdrant target, NOT from
MigrationResult, which undercounts after a checkpoint-resume). The report also
records the embedder identity so a search-parity gap is interpretable
(ANN-ordering difference vs embedder drift).

KG / hallways / tunnels are backend-independent files at ~/.mempalace/; B2's
check is that they are UNTOUCHED (byte-identical before/after the migration).
Resolve-against-qdrant is the B3 post-cutover check, not B2's.

Integration tests require a live qdrant on localhost:6333 (skipped if absent).
"""

from __future__ import annotations

import os

import pytest

from mempalace.migrate_parity import (
    EXPECTED_DIM,
    EXPECTED_EMBEDDER,
    ParityReport,
    embedder_model_id,
    home_files_fingerprint,
    search_overlap,
    source_tally,
    validate_parity,
)


# ── pure logic ───────────────────────────────────────────────────────────


def test_expected_embedder_is_minilm_384():
    assert EXPECTED_EMBEDDER == "minilm"
    assert EXPECTED_DIM == 384


def test_embedder_model_id_reports_configured_model():
    # Identity comes from config (the EF's name() is "default" and not useful).
    assert embedder_model_id() == "minilm"


def test_search_overlap_identical_is_one():
    assert search_overlap(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_search_overlap_disjoint_is_zero():
    assert search_overlap(["a", "b"], ["c", "d"]) == 0.0


def test_search_overlap_partial():
    # 2 shared of top-3 (exact ordering ignored) -> 2/3
    assert search_overlap(["a", "b", "c"], ["a", "b", "z"]) == pytest.approx(2 / 3)


def test_source_tally_paginates_and_never_does_unbounded_get():
    """Regression guard: a single unbounded get() on a 67k chroma collection
    blows SQLite's bind-variable limit. source_tally MUST paginate."""

    class FakeSource:
        def __init__(self, metas, page):
            self.metas = metas
            self.page = page

        def get(self, *, ids=None, where=None, limit=None, offset=None, include=None):
            if limit is None:
                raise AssertionError("source_tally did an UNBOUNDED get() — must paginate")
            chunk = self.metas[offset : offset + limit]
            return {"ids": [f"d{offset + i}" for i in range(len(chunk))], "metadatas": chunk}

    metas = (
        [{"wing": "ves_sessions", "room": "technical"}] * 5
        + [{"wing": "ves", "room": "diary"}] * 2
        + [{"wing": "ves", "room": "general"}] * 1
    )
    per_wing, diary = source_tally(FakeSource(metas, page=3), batch_size=3)
    assert per_wing == {"ves_sessions": 5, "ves": 3}
    assert diary == 2  # only ves/diary counts


def test_home_files_fingerprint_changes_when_a_file_changes(tmp_path):
    home = tmp_path / ".mempalace"
    home.mkdir()
    (home / "knowledge_graph.sqlite3").write_bytes(b"kg-v1")
    (home / "hallways.json").write_text("[]")
    (home / "tunnels.json").write_text("[]")
    palace = str(home / "palace")

    fp1 = home_files_fingerprint(palace)
    (home / "knowledge_graph.sqlite3").write_bytes(b"kg-v2-MUTATED")
    fp2 = home_files_fingerprint(palace)
    assert fp1 != fp2


def test_home_files_fingerprint_stable_when_unchanged(tmp_path):
    home = tmp_path / ".mempalace"
    home.mkdir()
    (home / "knowledge_graph.sqlite3").write_bytes(b"kg")
    (home / "hallways.json").write_text("[]")
    palace = str(home / "palace")
    assert home_files_fingerprint(palace) == home_files_fingerprint(palace)


# ── integration: real qdrant + synthetic chroma source ───────────────────

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

_SAMPLE = [
    {
        "id": "drawer_ves_sessions_a",
        "content": "alpha content about migration tooling",
        "meta": {"wing": "ves_sessions", "room": "technical", "filed_at_ts": 1000.0},
    },
    {
        "id": "drawer_ves_sessions_b",
        "content": "beta content about vector palaces",
        "meta": {"wing": "ves_sessions", "room": "planning", "filed_at_ts": 2000.0},
    },
    {
        "id": "drawer_ves_diary_c",
        "content": "gamma diary reflection entry",
        "meta": {"wing": "ves", "room": "diary", "filed_at_ts": 3000.0},
    },
]


def _seed(palace, drawers):
    from mempalace.palace import get_collection

    col = get_collection(palace, "mempalace_drawers", create=True, backend="chroma")
    col.add(
        documents=[d["content"] for d in drawers],
        ids=[d["id"] for d in drawers],
        metadatas=[d["meta"] for d in drawers],
    )


@requires_qdrant
def test_full_synthetic_migration_passes_parity(tmp_path):
    from mempalace.migrate_backend import migrate

    # qdrant target is a SEPARATE sibling dir: mempalace's resolver refuses to
    # open a dir with chroma artifacts as qdrant (BackendMismatchError). Same
    # parent (~/.mempalace) keeps KG/hallways/tunnels resolving; B3 flips both
    # backend AND palace_path to the sibling qdrant dir.
    source = str(tmp_path / "palace")
    target = str(tmp_path / "palace_qdrant")
    _seed(source, _SAMPLE)
    migrate(source, target, batch_size=2)

    report = validate_parity(
        source,
        target,
        queries=["migration tooling", "diary reflection"],
        anchors={"migration tooling": "drawer_ves_sessions_a"},
    )
    assert isinstance(report, ParityReport)
    assert report.embedder_model == "minilm"
    assert report.source_total == report.target_total == 3
    # per-wing counts come from the LIVE target, and must match source
    per_wing = {w.wing: w for w in report.per_wing}
    assert per_wing["ves_sessions"].source == 2
    assert per_wing["ves_sessions"].target == 2
    assert per_wing["ves_sessions"].match
    assert per_wing["ves"].match
    # diary parity: the one ves/diary drawer surfaces in both
    assert report.diary["source"] == report.diary["target"] == 1
    assert report.ok


@requires_qdrant
def test_parity_flags_target_count_shortfall(tmp_path):
    """If the target is missing drawers, parity must FAIL — not pass silently."""
    from mempalace.migrate_backend import migrate

    source = str(tmp_path / "palace")
    target = str(tmp_path / "palace_qdrant")
    _seed(source, _SAMPLE)
    migrate(source, target, wings=["ves_sessions"], batch_size=2)  # only 2 of 3 migrated

    report = validate_parity(source, target)
    # source has 3, target has 2 -> not ok
    assert report.source_total == 3
    assert report.target_total == 2
    assert not report.ok
