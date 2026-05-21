import os
import shutil
import sqlite3
import subprocess
import sys

import chromadb
import pytest

from mempalace.backends.chroma import ChromaBackend, ChromaCollection, _fix_blob_seq_ids


class _FakeCollection:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return {"kind": "query"}

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return {"kind": "get"}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def count(self):
        self.calls.append(("count", {}))
        return 7


def test_chroma_collection_delegates_methods():
    fake = _FakeCollection()
    collection = ChromaCollection(fake)

    collection.add(documents=["d"], ids=["1"], metadatas=[{"wing": "w"}])
    collection.upsert(documents=["u"], ids=["2"], metadatas=[{"room": "r"}])
    assert collection.query(query_texts=["q"]) == {"kind": "query"}
    assert collection.get(where={"wing": "w"}) == {"kind": "get"}
    collection.delete(ids=["1"])
    assert collection.count() == 7

    assert fake.calls == [
        ("add", {"documents": ["d"], "ids": ["1"], "metadatas": [{"wing": "w"}]}),
        ("upsert", {"documents": ["u"], "ids": ["2"], "metadatas": [{"room": "r"}]}),
        ("query", {"query_texts": ["q"]}),
        ("get", {"where": {"wing": "w"}}),
        ("delete", {"ids": ["1"]}),
        ("count", {}),
    ]


def test_chroma_backend_create_false_raises_without_creating_directory(tmp_path):
    palace_path = tmp_path / "missing-palace"

    with pytest.raises(FileNotFoundError):
        ChromaBackend().get_collection(
            str(palace_path),
            collection_name="mempalace_drawers",
            create=False,
        )

    assert not palace_path.exists()


def test_chroma_backend_create_true_creates_directory_and_collection(tmp_path):
    palace_path = tmp_path / "palace"

    collection = ChromaBackend().get_collection(
        str(palace_path),
        collection_name="mempalace_drawers",
        create=True,
    )

    assert palace_path.is_dir()
    assert isinstance(collection, ChromaCollection)

    client = chromadb.PersistentClient(path=str(palace_path))
    client.get_collection("mempalace_drawers")


def test_chroma_backend_creates_collection_with_cosine_distance(tmp_path):
    palace_path = tmp_path / "palace"

    ChromaBackend().get_collection(
        str(palace_path),
        collection_name="mempalace_drawers",
        create=True,
    )

    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    assert col.metadata.get("hnsw:space") == "cosine"


def test_fix_blob_seq_ids_converts_blobs_to_integers(tmp_path):
    """Simulate a ChromaDB 0.6.x database with BLOB seq_ids and verify repair."""
    db_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, seq_id)")
    conn.execute("CREATE TABLE max_seq_id (rowid INTEGER PRIMARY KEY, seq_id)")
    # Insert BLOB seq_ids like ChromaDB 0.6.x would
    blob_42 = (42).to_bytes(8, byteorder="big")
    blob_99 = (99).to_bytes(8, byteorder="big")
    conn.execute("INSERT INTO embeddings (seq_id) VALUES (?)", (blob_42,))
    conn.execute("INSERT INTO max_seq_id (seq_id) VALUES (?)", (blob_99,))
    conn.commit()
    conn.close()

    _fix_blob_seq_ids(str(tmp_path))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM embeddings").fetchone()
    assert row == (42, "integer")
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM max_seq_id").fetchone()
    assert row == (99, "integer")
    conn.close()


def test_fix_blob_seq_ids_noop_without_blobs(tmp_path):
    """No error when seq_ids are already integers."""
    db_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE embeddings (rowid INTEGER PRIMARY KEY, seq_id INTEGER)")
    conn.execute("INSERT INTO embeddings (seq_id) VALUES (42)")
    conn.commit()
    conn.close()

    _fix_blob_seq_ids(str(tmp_path))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT seq_id, typeof(seq_id) FROM embeddings").fetchone()
    assert row == (42, "integer")
    conn.close()


def test_fix_blob_seq_ids_noop_without_database(tmp_path):
    """No error when palace has no chroma.sqlite3."""
    _fix_blob_seq_ids(str(tmp_path))  # should not raise


# Inline script run as a *separate process*: ensures the collection exists,
# adds one drawer with an explicit embedding (so no embedding model is
# needed), and self-confirms the drawer is retrievable through its own fresh
# client before exiting. The self-confirmation rules out a writer-indexing
# race — any miss the parent then sees is unambiguously a stale cached client.
_CROSS_PROCESS_ADD_SCRIPT = """
import sys
import chromadb

palace_path, drawer_id = sys.argv[1], sys.argv[2]
vec = [0.11, 0.22, 0.33, 0.44]

client = chromadb.PersistentClient(path=palace_path)
col = client.get_or_create_collection(
    "mempalace_drawers", metadata={"hnsw:space": "cosine"}
)
col.add(ids=[drawer_id], documents=["cross-process drawer"], embeddings=[vec])

got = col.get(ids=[drawer_id])
if got.get("ids") != [drawer_id]:
    print("ADD_GET_MISS", got.get("ids"))
    sys.exit(1)
q = col.query(query_embeddings=[vec], n_results=10)
if drawer_id not in (q.get("ids") or [[]])[0]:
    print("ADD_QUERY_MISS", q.get("ids"))
    sys.exit(1)
print("ADD_CONFIRMED")
"""


def test_chroma_backend_client_reconnects_to_cross_process_writes(tmp_path):
    """Regression: a cached ``_client`` with a materialized index must observe
    drawers written by another process, instead of serving a frozen HNSW
    vector segment.

    Reproduces the gateway staleness bug. ``ChromaBackend`` caches one
    ``PersistentClient`` per palace for the process lifetime. Once that
    client's in-memory HNSW index is materialized by a query, drawers written
    by other processes (miners, the MCP server, the CLI) stay invisible to
    ``query()`` until the holding process restarts — ``count()`` reads fresh
    from SQLite, but vector search does not.
    """
    palace_path = str(tmp_path / "palace")
    os.makedirs(palace_path)
    vec = [0.11, 0.22, 0.33, 0.44]

    def _add_drawer_in_subprocess(drawer_id):
        result = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_ADD_SCRIPT, palace_path, drawer_id],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"subprocess add of {drawer_id!r} failed: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "ADD_CONFIRMED" in result.stdout, (
            f"subprocess did not self-confirm {drawer_id!r} is retrievable — "
            f"cannot distinguish staleness from a setup race: "
            f"stdout={result.stdout!r}"
        )

    # Another process pre-seeds one drawer so the collection is non-empty.
    _add_drawer_in_subprocess("drawer_seed")

    backend = ChromaBackend()

    # Parent: warm the per-instance client cache AND materialize its HNSW
    # index by running a query. Without this query the index is lazily
    # (re)loaded on next access and the staleness never manifests.
    col = backend.get_collection(palace_path, "mempalace_drawers", create=True)
    warm = col.query(query_embeddings=[vec], n_results=10)
    assert "drawer_seed" in (warm.get("ids") or [[]])[0]

    # Another process plants a second drawer and self-confirms it.
    _add_drawer_in_subprocess("drawer_planted")

    # Parent: re-acquire through the SAME backend instance. On the unfixed
    # code the cached client's frozen HNSW segment never sees the new drawer.
    col_after = backend.get_collection(palace_path, "mempalace_drawers", create=True)
    found = col_after.query(query_embeddings=[vec], n_results=10)
    found_ids = (found.get("ids") or [[]])[0]
    assert "drawer_planted" in found_ids, (
        "cached _client served a stale HNSW index — the cross-process write "
        f"is invisible to vector search. query returned {found_ids!r}, "
        "expected to contain 'drawer_planted'"
    )


def test_chroma_backend_client_cached_when_db_unchanged(tmp_path):
    """``_client`` returns the same cached client while the palace DB on
    disk is unchanged — the lock-free cache-hit fast path."""
    palace_path = str(tmp_path / "palace")
    backend = ChromaBackend()
    backend.get_collection(palace_path, "mempalace_drawers", create=True)

    c1 = backend._client(palace_path)
    c2 = backend._client(palace_path)
    assert c2 is c1


def test_chroma_backend_client_reconnects_when_db_mtime_changes(tmp_path):
    """``_client`` reconnects when the palace DB mtime advances (an in-place
    write by another process), and the refreshed cache entry is then stable
    until the next change."""
    palace_path = str(tmp_path / "palace")
    backend = ChromaBackend()
    backend.get_collection(palace_path, "mempalace_drawers", create=True)
    c1 = backend._client(palace_path)

    # Simulate an external in-place write by advancing the DB file mtime.
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    st = os.stat(db_path)
    os.utime(db_path, (st.st_atime, st.st_mtime + 5))

    c2 = backend._client(palace_path)
    assert c2 is not c1, "client should reconnect after the DB mtime changed"

    # The cache entry was refreshed — a further call with no change is cached.
    c3 = backend._client(palace_path)
    assert c3 is c2, "client should be cached once the DB is unchanged again"


def test_chroma_backend_client_reconnects_after_palace_rebuild(tmp_path):
    """``_client`` reconnects when the palace DB is replaced with a new inode
    — the repair/nuke/re-mine rebuild case, which the mtime check alone can
    miss (a rebuild can preserve mtime)."""
    palace_path = str(tmp_path / "palace")
    backend = ChromaBackend()
    backend.get_collection(palace_path, "mempalace_drawers", create=True)
    c1 = backend._client(palace_path)

    db_path = os.path.join(palace_path, "chroma.sqlite3")
    old_inode = os.stat(db_path).st_ino

    # Rebuild: replace the DB file with a copy so the path gets a new inode
    # while mtime is preserved — isolating the inode-change detection path.
    replacement = str(tmp_path / "rebuilt.sqlite3")
    shutil.copy2(db_path, replacement)
    os.replace(replacement, db_path)
    assert os.stat(db_path).st_ino != old_inode, "test setup: inode should differ"

    c2 = backend._client(palace_path)
    assert c2 is not c1, "client should reconnect after the palace DB was replaced"


def test_chroma_backend_client_reconnects_when_db_file_missing(tmp_path):
    """``_client`` reconnects when the palace DB file is absent (e.g.
    mid-rebuild) — inode and mtime both read as 0, so the change checks
    cannot detect it and the missing-file guard must."""
    palace_path = str(tmp_path / "palace")
    backend = ChromaBackend()
    backend.get_collection(palace_path, "mempalace_drawers", create=True)
    c1 = backend._client(palace_path)

    os.remove(os.path.join(palace_path, "chroma.sqlite3"))

    c2 = backend._client(palace_path)
    assert c2 is not c1, "client should reconnect when the DB file is gone"
