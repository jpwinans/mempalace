"""ChromaDB-backed MemPalace collection adapter."""

import logging
import os
import sqlite3
import threading
from typing import Any, NamedTuple

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient

from .base import BaseCollection

logger = logging.getLogger(__name__)


def _fix_blob_seq_ids(palace_path: str):
    """Fix ChromaDB 0.6.x -> 1.5.x migration bug: BLOB seq_ids -> INTEGER.

    ChromaDB 0.6.x stored seq_id as big-endian 8-byte BLOBs. ChromaDB 1.5.x
    expects INTEGER. The auto-migration doesn't convert existing rows, causing
    the Rust compactor to crash with "mismatched types; Rust type u64 (as SQL
    type INTEGER) is not compatible with SQL type BLOB".

    Must run BEFORE PersistentClient is created (the compactor fires on init).
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            for table in ("embeddings", "max_seq_id"):
                try:
                    rows = conn.execute(
                        f"SELECT rowid, seq_id FROM {table} WHERE typeof(seq_id) = 'blob'"
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if not rows:
                    continue
                updates = [(int.from_bytes(blob, byteorder="big"), rowid) for rowid, blob in rows]
                conn.executemany(f"UPDATE {table} SET seq_id = ? WHERE rowid = ?", updates)
                logger.info("Fixed %d BLOB seq_ids in %s", len(updates), table)
            conn.commit()
    except Exception:
        logger.exception("Could not fix BLOB seq_ids in %s", db_path)


class _CachedClient(NamedTuple):
    """A cached PersistentClient tagged with the palace DB identity it was
    opened against, so staleness can be detected on later access."""

    client: Any
    inode: int
    mtime: float


def _stat_db(db_path: str) -> tuple[int, float]:
    """Return ``(inode, mtime)`` of *db_path*, or ``(0, 0.0)`` if it cannot
    be stat-ed (missing file, or a filesystem that reports no inode)."""
    try:
        st = os.stat(db_path)
        return st.st_ino, st.st_mtime
    except OSError:
        return 0, 0.0


def _cache_entry_fresh(
    entry: "_CachedClient | None", db_path: str, inode: int, mtime: float
) -> bool:
    """True when a cached client can still be served without reconnecting.

    A reconnect is needed when the palace's ``chroma.sqlite3`` has a new
    inode (a full rebuild via repair/nuke/re-mine replaces the file) or a
    changed mtime (in-place writes by other processes that the cached
    client's in-memory HNSW index never saw). ``st_ino == 0`` (FAT/exFAT,
    which do not report inodes) disables the inode check as a safe fallback
    — on those filesystems staleness detection relies on mtime alone.
    """
    if entry is None:
        return False
    if not os.path.isfile(db_path):
        # DB file gone (e.g. mid-rebuild): inode and mtime both read as 0,
        # so the change checks below would both be False and wrongly keep
        # the stale client. Force a reconnect instead.
        return False
    inode_changed = inode != 0 and inode != entry.inode
    mtime_changed = mtime != 0.0 and abs(mtime - entry.mtime) > 0.01
    return not (inode_changed or mtime_changed)


class ChromaCollection(BaseCollection):
    """Thin adapter over a ChromaDB collection."""

    def __init__(self, collection):
        self._collection = collection

    def add(self, *, documents, ids, metadatas=None):
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def upsert(self, *, documents, ids, metadatas=None):
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def update(self, **kwargs):
        self._collection.update(**kwargs)

    def query(self, **kwargs):
        return self._collection.query(**kwargs)

    def get(self, **kwargs):
        return self._collection.get(**kwargs)

    def delete(self, **kwargs):
        self._collection.delete(**kwargs)

    def count(self):
        return self._collection.count()


class ChromaBackend:
    """Factory for MemPalace's default ChromaDB backend."""

    def __init__(self):
        # Per-instance client cache: palace_path -> _CachedClient
        self._clients: dict = {}
        # Serializes the reconnect path; the cache-hit fast path is lock-free.
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self, palace_path: str):
        """Return a cached PersistentClient for *palace_path*, reconnecting
        when the palace's ``chroma.sqlite3`` changed on disk.

        A long-lived client holds a frozen in-memory HNSW index; without
        reconnection it never sees drawers written by other processes
        (miners, the MCP server, the CLI) or a palace rebuilt underneath it.
        The cache-hit fast path is lock-free; only the rare reconnect is
        serialized, so concurrent callers cannot create duplicate clients.
        """
        db_path = os.path.join(palace_path, "chroma.sqlite3")
        inode, mtime = _stat_db(db_path)

        entry = self._clients.get(palace_path)
        if _cache_entry_fresh(entry, db_path, inode, mtime):
            return entry.client

        with self._client_lock:
            # Re-stat and re-check under the lock — another thread may have
            # reconnected while this one waited.
            inode, mtime = _stat_db(db_path)
            entry = self._clients.get(palace_path)
            if _cache_entry_fresh(entry, db_path, inode, mtime):
                return entry.client
            # A transiently-absent DB (mid-rebuild) yields a fresh empty DB
            # here; the next call self-heals onto the rebuilt file (new inode).
            client = ChromaBackend.make_client(palace_path)
            self._clients[palace_path] = _CachedClient(client, inode, mtime)
            return client

    # ------------------------------------------------------------------
    # Public static helpers (for callers that manage their own caching)
    # ------------------------------------------------------------------

    @staticmethod
    def make_client(palace_path: str):
        """Create and return a genuinely fresh PersistentClient.

        ChromaDB's ``SharedSystemClient`` caches the underlying ``System``
        (and its in-memory HNSW segments) by path, process-wide — so a bare
        ``PersistentClient(path)`` hands back a new client object that still
        reuses the *frozen* segments of any earlier client for that path.
        ``clear_system_cache()`` evicts those cached Systems so the new
        client reloads segments from disk and observes writes made by other
        processes (or a palace rebuilt underneath us).

        The eviction is process-wide: it also drops other in-process
        ChromaDB Systems, which then reload lazily on next use — a bounded
        re-load cost, not a correctness change (cached Systems are evicted,
        not stopped, so client objects already holding one keep working).
        Both callers (``ChromaBackend._client`` and
        ``mcp_server._get_client``) gate this behind inode/mtime change
        detection, so it runs on reconnect or initial client creation,
        never on the cache-hit path.
        """
        _fix_blob_seq_ids(palace_path)
        # clear_system_cache() is ChromaDB-internal API (imported from
        # chromadb.api.shared_system_client) and is load-bearing: without
        # the System eviction the reconnect silently reuses frozen segments.
        # Verified against chromadb 0.6.3 — revisit on a chromadb upgrade.
        SharedSystemClient.clear_system_cache()
        return chromadb.PersistentClient(path=palace_path)

    @staticmethod
    def backend_version() -> str:
        """Return the installed chromadb package version string."""
        return chromadb.__version__

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def get_collection(self, palace_path: str, collection_name: str, create: bool = False):
        if not create and not os.path.isdir(palace_path):
            raise FileNotFoundError(palace_path)

        if create:
            os.makedirs(palace_path, exist_ok=True)
            try:
                os.chmod(palace_path, 0o700)
            except (OSError, NotImplementedError):
                pass

        client = self._client(palace_path)
        if create:
            collection = client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = client.get_collection(collection_name)
        return ChromaCollection(collection)

    def get_or_create_collection(
        self, palace_path: str, collection_name: str
    ) -> "ChromaCollection":
        """Shorthand for get_collection(..., create=True)."""
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        """Delete *collection_name* from the palace at *palace_path*."""
        self._client(palace_path).delete_collection(collection_name)

    def create_collection(
        self, palace_path: str, collection_name: str, hnsw_space: str = "cosine"
    ) -> "ChromaCollection":
        """Create (not get-or-create) *collection_name* with cosine HNSW space."""
        collection = self._client(palace_path).create_collection(
            collection_name, metadata={"hnsw:space": hnsw_space}
        )
        return ChromaCollection(collection)
