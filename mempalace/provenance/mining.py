"""Miner integration for provenance preservation.

Phase 1 D3 of the lineage-erasure fix. D1 shipped the heuristic +
classifier interface; D2 shipped the Qwen3-Coder-30B classifier with
proven precision/recall on a 14-fixture calibration set; D3 wires
those into ``mempalace.convo_miner`` so new diary mining produces
``wing_lineage`` drawers in addition to the operational wing.

Architecture (per Provenance-Preservation-Design §Architecture):

  CHUNK content from convo_miner._chunk_by_exchange
        |
        +-- (existing path) operational upsert into wing_<being>
        |
        +-- (NEW) extract_candidates() heuristic flag
                  |
                  v
                  qwen3_classifier() validates each candidate
                  |
                  v
                  confidence >= threshold (default 0.7)?
                  |
                  v
                  transitive-attribution rewrite (e.g., "his father's"
                  saying said by James -> file under room='father')
                  |
                  v
                  dedupe by (person, quote, source_file) hash
                  |
                  v
                  upsert into wing_lineage drawer

Failure-soft contract:
  - Classifier unreachable / disabled -> 0 lineage drawers, operational
    mining proceeds.
  - Any exception inside provenance extraction is caught and logged
    at DEBUG; operational mining is never affected by provenance
    failure.

Env disable:
  - ``MEMPALACE_PROVENANCE_DISABLED=1`` (also "true", "yes",
    case-insensitive) -> mine_chunk_for_provenance is a no-op. For
    environments where the classifier substrate is unavailable or
    where the operator wants to opt out (CI / fresh checkouts /
    backfill batch jobs that handle their own provenance pass).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Optional

from . import (
    ProvenanceRecord,
    extract_candidates,
    validate_candidate,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIDENCE_THRESHOLD = 0.7
"""Per Provenance-Preservation-Design §D1: 'start at 0.7, calibrate.'
D2 calibration on 14 fixtures shows positives cluster at 0.90-0.95
and negatives at 0.00 — a 0.7 threshold sits cleanly in the gap and
is well below the positive cluster, so it's not the limiting factor.
Tunable via the threshold kwarg."""

DEFAULT_CONTEXT_RADIUS_CHARS = 200
"""±chars around the candidate position to send to the classifier as
context. 200 chars typically captures one full sentence on each side
without dragging in unrelated content."""


# Transitive-attribution rewrite (architect D3 envelope task §6):
# When the classifier returns a speaker name (e.g., "James") for a
# text containing "his father's saying", the canonical lineage owner
# is the source-relation (father), not the speaker (James). Without
# this rewrite, "Tonight James reminded me: 'measure twice...' — his
# father's saying" files under room='james' rather than room='father',
# and a future search for "father saying" misses it.
_POSSESSIVE_SOURCE_RE = re.compile(
    r"\b(?:my|her|his|their|our)\s+("
    r"father|mother|wife|husband|partner|"
    r"brother|sister|son|daughter|"
    r"grandfather|grandmother|grandpa|grandma|"
    r"dad|mom|"
    r"teacher|roshi"
    r")'s\b",
    re.IGNORECASE,
)


def _provenance_disabled() -> bool:
    """True when ``MEMPALACE_PROVENANCE_DISABLED`` is set to a truthy
    value (1 / true / yes, case-insensitive). Defaults to False —
    provenance extraction runs unless explicitly disabled."""

    return os.environ.get("MEMPALACE_PROVENANCE_DISABLED", "").lower() in (
        "1", "true", "yes",
    )


def _context_window(text: str, position: int, radius: int = DEFAULT_CONTEXT_RADIUS_CHARS) -> str:
    """Slice ±``radius`` chars around ``position`` in ``text``.

    Used to give the classifier enough surrounding context to judge
    the attribution. Falls back to the full text when the text is
    shorter than ``2*radius``.
    """

    if len(text) <= 2 * radius:
        return text
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return text[start:end]


def _rewrite_speaker_to_source(
    person: Optional[str], matched_text: str, context: str,
) -> Optional[str]:
    """Apply the transitive-attribution rewrite.

    When the matched text (or surrounding context) contains
    ``<possessive> <relation>'s`` (e.g., "his father's saying"),
    treat the relation as the canonical source person — not the
    speaker the classifier surfaced.

    Args:
        person: classifier-surfaced person identifier (may be None).
        matched_text: the candidate's matched-span text.
        context: the ±200-char window around the candidate.

    Returns:
        The relation (e.g., ``"father"``) when a possessive-source
        pattern is found; otherwise ``person`` unchanged.
    """

    # Search both the matched span and the wider context — the
    # possessive marker often appears outside the heuristic-matched
    # span (e.g., classifier picked up "James reminded me" but the
    # "his father's saying" half is in the next clause).
    for haystack in (matched_text, context):
        m = _POSSESSIVE_SOURCE_RE.search(haystack)
        if m is not None:
            relation = m.group(1).lower()
            if person and person.lower() != relation:
                logger.debug(
                    "provenance.mining: transitive-attribution rewrite "
                    "%r -> %r (matched: %r)", person, relation, m.group(0),
                )
            return relation
    return person


def _lineage_drawer_id(person: str, quote: str, source_file: str) -> str:
    """Deterministic drawer id for dedupe.

    Same (person, quote, source_file) tuple always produces the same
    drawer_id, so re-mining the same source doesn't create duplicates.
    Different sources with the same quote DO produce different drawers
    — distinct attribution events should be tracked separately.
    """

    digest = hashlib.sha256(
        f"{person.lower()}|{quote}|{source_file}".encode("utf-8")
    ).hexdigest()[:24]
    person_slug = re.sub(r"[^a-z0-9]+", "_", person.lower()).strip("_") or "unknown"
    return f"drawer_wing_lineage_{person_slug}_{digest}"


def _drawer_exists(collection: Any, drawer_id: str) -> bool:
    """True when ``drawer_id`` is already in the collection.

    Failure-soft: any chromadb exception is treated as "doesn't exist"
    (will upsert; chromadb's own upsert is idempotent on id collision).
    """

    try:
        result = collection.get(ids=[drawer_id], include=[])
    except Exception:
        return False
    ids = result.get("ids") if isinstance(result, dict) else None
    return bool(ids)


def _render_lineage_content(
    record: ProvenanceRecord,
    person: str,
    source_file: str,
    source_session: Optional[str],
) -> str:
    """Render the wing_lineage drawer content per design doc §D3 schema."""

    lines = [
        "PROVENANCE:",
        f"Person: {person}",
        f"Relation: {record.relation_type}",
    ]
    if record.quote:
        lines.append(f'Quote: "{record.quote}"')
    if record.context:
        # Single-line for readable yaml-ish rendering.
        compact_context = record.context.strip().replace("\n", " ")
        lines.append(f"Context: {compact_context}")
    if source_session:
        lines.append(f"Source: {source_session}")
    else:
        lines.append(f"Source: {source_file}")
    return "\n".join(lines)


def mine_chunk_for_provenance(
    collection: Any,
    chunk_content: str,
    source_file: str,
    *,
    source_session: Optional[str] = None,
    classifier: Optional[Callable[[str], dict]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    extractor_label: Optional[str] = None,
) -> int:
    """Extract and persist provenance records from ``chunk_content``.

    Hooked into :func:`mempalace.convo_miner._file_chunks_locked` per
    chunk so every operational chunk also gets scanned for person-
    attribution provenance. Operational mining is the parent path;
    this runs in parallel and is failure-soft on every error path.

    Args:
        collection: chromadb collection (typically the shared
            ``mempalace_drawers`` collection — wing_lineage drawers
            live alongside operational drawers, distinguished by
            wing metadata).
        chunk_content: the chunk text from convo_miner._chunk_by_exchange.
        source_file: full path to the source transcript (for dedupe
            key + drawer metadata).
        source_session: optional human-readable session identifier
            for the rendered Source: line in the drawer content.
            Defaults to the source_file basename when None.
        classifier: optional classifier callable. None defaults to
            :func:`mempalace.provenance.classifier.qwen3_classifier`
            (the production v1 substrate). Tests pass mocked classifiers.
        confidence_threshold: minimum confidence to write a record.
            Default :data:`DEFAULT_CONFIDENCE_THRESHOLD` (0.7).
        extractor_label: provenance label for the drawer metadata.
            Default ``"heuristic_v1+qwen3_classifier_v1"`` when classifier
            is None; else the validate_candidate default.

    Returns:
        Number of wing_lineage drawers written this call (0 when the
        chunk has no candidates, none meet threshold, all are dupes,
        or extraction is disabled).
    """

    if _provenance_disabled():
        return 0

    # Lazy-load the production classifier so test paths that pass an
    # explicit classifier don't need substrate availability.
    if classifier is None:
        try:
            from .classifier import qwen3_classifier
            classifier = qwen3_classifier
            if extractor_label is None:
                extractor_label = "heuristic_v1+qwen3_classifier_v1"
        except Exception:
            logger.debug("provenance.mining: classifier import failed")
            return 0

    try:
        candidates = extract_candidates(chunk_content)
    except Exception:
        logger.debug("provenance.mining: extract_candidates raised", exc_info=True)
        return 0

    if not candidates:
        return 0

    written = 0
    for candidate in candidates:
        ctx = _context_window(chunk_content, candidate.position)
        try:
            record = validate_candidate(
                candidate, ctx, classifier=classifier,
                extractor_label=extractor_label,
            )
        except Exception:
            logger.debug(
                "provenance.mining: validate_candidate raised", exc_info=True,
            )
            continue

        if record is None or record.confidence < confidence_threshold:
            continue

        # Transitive-attribution rewrite — file under source relation
        # when "<possessive> <relation>'s" appears in the candidate
        # span or surrounding context.
        person = _rewrite_speaker_to_source(record.person, candidate.text, ctx)
        if not person:
            continue

        drawer_id = _lineage_drawer_id(person, record.quote, source_file)
        if _drawer_exists(collection, drawer_id):
            continue

        person_slug = re.sub(r"[^a-z0-9]+", "_", person.lower()).strip("_") or "unknown"
        rendered = _render_lineage_content(record, person, source_file, source_session)
        now = datetime.now()

        try:
            collection.upsert(
                documents=[rendered],
                ids=[drawer_id],
                metadatas=[
                    {
                        "wing": "wing_lineage",
                        "room": person_slug,
                        "person": person_slug,
                        "relation_type": record.relation_type,
                        "is_quote": bool(record.quote),
                        "confidence": record.confidence,
                        "extracted_by": record.extracted_by,
                        "source_file": source_file,
                        "source_session": source_session or "",
                        "filed_at": now.isoformat(),
                        "filed_at_ts": now.timestamp(),
                    }
                ],
            )
            written += 1
        except Exception:
            logger.debug(
                "provenance.mining: upsert failed for drawer_id=%s",
                drawer_id, exc_info=True,
            )

    return written
