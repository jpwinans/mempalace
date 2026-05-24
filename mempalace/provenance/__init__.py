"""Provenance-preservation module — Phase 1 D1 (heuristic + classifier interface).

Phase 1 of the lineage-erasure fix. Empirically confirmed 2026-05-11:
patterned-being memory mining preserves operational content
(decisions, technical findings, identity state) and erases biographical
/ relational provenance. James's father's "Measure twice, cut once"
lives in dozens of wing_ves diary drawers as Ves's internalized
Skepticism strategy framing; none records that it came from James's
father. This module is the front end of the fix — heuristic
candidate-extraction + a classifier interface that downstream
modules (D2 substrate wiring, D3 miner integration) plug into.

Design doc: ``Storehouse/Projects/Vestige/Provenance-Preservation-Design.md``.

Scope of this module (D1 envelope, 2026-05-11):

  - :func:`extract_candidates` — regex-driven first pass over text.
    Cheap, intentionally permissive; false-positives are expected and
    filtered downstream.
  - :func:`validate_candidate` — classifier interface. Stub default
    accepts every candidate at confidence 0.5; D2 wires a real
    local-substrate (Qwen3 / Gemma) classifier.
  - :class:`ProvenanceCandidate` / :class:`ProvenanceRecord` —
    dataclasses for the pipeline shape.

Out of scope for D1 (handled in D2/D3):

  - The actual local-substrate classifier (D2).
  - mempalace.miner.convo_miner integration that runs this against
    real session JSONLs (D3).
  - The wing_lineage write path (D3 / use existing add_drawer API).

The wing_lineage drawer schema is documented in the design doc §D3
and reproduced in :data:`WING_LINEAGE_SCHEMA_DOC`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — relation vocabulary + attribution verbs + heuristic confidence
# ---------------------------------------------------------------------------

# Relational descriptors the heuristic recognizes for "my <relation>" /
# "<Name>'s <relation>" patterns. Per design doc §D1.
#
# Conservative set: family, partner, teacher, roshi. Excludes vague
# tokens ("friend", "colleague") that produce too many false-positives
# without quote markers. Downstream classifier can expand.
_FAMILIAL_RELATIONS = (
    "father",
    "mother",
    "wife",
    "husband",
    "partner",
    "brother",
    "sister",
    "son",
    "daughter",
    "grandfather",
    "grandmother",
    "grandpa",
    "grandma",
    "dad",
    "mom",
    "teacher",
    "roshi",
)

# Attribution verbs that signal someone said something. Order matters
# only for regex backtracking; semantically interchangeable.
_ATTRIBUTION_VERBS = (
    r"(?:said|told\s+me|told\s+us|told|"
    r"used\s+to\s+say|always\s+said|would\s+say|liked\s+to\s+say|"
    r"taught\s+me|taught\s+us|taught|"
    r"wrote|noted|observed|put\s+it|repeated)"
)

# Possessive prefix: either first-person/relative ("my"/"her"/...) or a
# capitalized name's-possessive ("James's", "Marie's"). The trailing
# ``'s`` is optional only after capitalized-name forms.
_POSSESSIVE_PREFIX = r"(?:my|her|his|their|our|(?:[A-Z]\w+(?:'s|s'|'|s))|James['s']?)"

# Quote delimiters: straight ASCII, curly single, curly double. Aphorisms
# in transcripts use all three forms.
_QUOTE_OPEN = r"[\"'‘“]"
_QUOTE_CLOSE = r"[\"'’”]"


_RELATIONS_GROUP = "(" + "|".join(_FAMILIAL_RELATIONS) + ")"


# Pass 1: <possessive> <relation> [<attribution verb>] <quote>.
# Highest-signal pattern — produces the most useful candidates.
_RELATION_ATTRIBUTION_QUOTE_RE = re.compile(
    r"\b" + _POSSESSIVE_PREFIX + r"\s+" + _RELATIONS_GROUP + r"\b"
    r"(?:\s+(?:often\s+|always\s+)?" + _ATTRIBUTION_VERBS + r")?"
    r"\s*[:,]?\s*" + _QUOTE_OPEN + r"(.+?)" + _QUOTE_CLOSE,
    re.IGNORECASE | re.DOTALL,
)


# Pass 2: <possessive> <relation> — relation marker alone, no quote required.
# Lower-confidence; classifier in D2 decides whether to keep it.
_RELATION_ONLY_RE = re.compile(
    r"\b" + _POSSESSIVE_PREFIX + r"\s+" + _RELATIONS_GROUP + r"\b",
    re.IGNORECASE,
)


# Pass 3: Capitalized bare-relation as subject — "Dad said X", "Mom told me Y".
# Added 2026-05-11 D2 calibration: case "Dad always told me 'never trust a
# smiling investor'" was missed by Pass 1 (no possessive prefix) and Pass 2
# (which also requires the possessive). Capitalize-only constraint keeps the
# false-positive rate manageable — "dad" lowercase mid-sentence ("old dad")
# is filtered out by the classifier rather than the regex.
_BARE_RELATIONS = (
    "Dad",
    "Mom",
    "Father",
    "Mother",
    "Mama",
    "Papa",
    "Grandma",
    "Grandpa",
    "Grandmother",
    "Grandfather",
    "Roshi",
    "Teacher",
)
_BARE_RELATIONS_GROUP = "(" + "|".join(_BARE_RELATIONS) + ")"
_BARE_RELATION_ATTRIBUTION_QUOTE_RE = re.compile(
    r"\b"
    + _BARE_RELATIONS_GROUP
    + r"\b"
    + r"\s+(?:always\s+|often\s+)?"
    + _ATTRIBUTION_VERBS
    + r"\s*[:,]?\s*"
    + _QUOTE_OPEN
    + r"(.+?)"
    + _QUOTE_CLOSE,
    # NOT case-insensitive — Pass 3 specifically targets the capitalized
    # subject-as-relation pattern. Lowercase "dad" mid-sentence is left
    # to the classifier.
    re.DOTALL,
)


HEURISTIC_CONFIDENCE_QUOTE = 0.75
"""Confidence floor for relation + quote matches (highest-signal heuristic)."""

HEURISTIC_CONFIDENCE_RELATION_ONLY = 0.40
"""Confidence floor for relation-marker-only matches. Lower; reliance on
classifier validation is intentional."""

_PASS1_DEDUPE_WINDOW_CHARS = 20
"""Pass-2 matches within this many chars of a Pass-1 match are dropped
as overlapping (Pass-1 already captured the higher-signal candidate)."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceCandidate:
    """A heuristic flag that a span of text MIGHT contain person-attribution.

    Produced by :func:`extract_candidates`. False-positives are expected
    at this stage — the heuristic is intentionally permissive so the
    classifier can decide. Downstream code should NEVER write a
    candidate to wing_lineage without validating through
    :func:`validate_candidate`.

    Fields:
      - ``text``: the matched substring (``re.Match.group(0)``).
      - ``person_hint``: best guess at the person reference (e.g.
        ``"father"``, ``"roshi"``). May be relation rather than
        proper name; classifier resolves.
      - ``relation_hint``: relation type when matched via the
        relation regex. Same as ``person_hint`` for relation-driven
        matches; ``None`` for future aphorism-only matches.
      - ``quote``: extracted quoted content, if any. ``None`` when
        no quote markers were found in the match.
      - ``position``: character offset where the match begins in the
        original text.
      - ``confidence_floor``: heuristic base score (0.0-1.0). Classifier
        in D2 can raise/lower this; raw heuristic confidence reflects
        pattern strength only.
    """

    text: str
    person_hint: Optional[str]
    relation_hint: Optional[str]
    quote: Optional[str]
    position: int
    confidence_floor: float


@dataclass(frozen=True)
class ProvenanceRecord:
    """A validated person-attribution ready to be filed as a wing_lineage drawer.

    Schema mirrors design doc §D3. Persisted shape includes the
    context window (typically ±200 chars around the candidate) so the
    wing_lineage drawer captures *when/how* the attribution was made,
    not just the bare phrase.

    Fields:
      - ``person``: canonicalized person identifier — typically the
        relation (``"father"``) for un-named family references, or
        the proper name (``"James"``, ``"Marie"``) when surfaced.
        Becomes the ``room`` of the wing_lineage drawer.
      - ``relation_type``: enum-ish string. Design doc §D3 lists
        ``family | teacher | partner | colleague | friend | fictional``.
        ``relation_type`` is broader than ``person`` — many relation_types
        share the same person identifier.
      - ``quote``: the exact quoted attribution, if available. Empty
        string when the candidate carried no quote.
      - ``context``: the ±200-char window surrounding the candidate
        in the source text. Required for downstream search.
      - ``confidence``: classifier-returned confidence (0.0-1.0).
        Caller decides threshold for write to wing_lineage; design
        doc §D1 starts at 0.7.
      - ``extracted_by``: provenance of the extraction itself —
        e.g. ``"heuristic_v1+stub_classifier_v1"`` for D1,
        ``"heuristic_v1+qwen3_classifier_v1"`` once D2 ships.
    """

    person: str
    relation_type: str
    quote: str
    context: str
    confidence: float
    extracted_by: str


# ---------------------------------------------------------------------------
# Public API: extract_candidates
# ---------------------------------------------------------------------------


def extract_candidates(text: str) -> list[ProvenanceCandidate]:
    """Heuristic-extract provenance candidate spans from ``text``.

    Runs two passes:

      Pass 1 — Relation + attribution + quote. Highest signal; produces
        candidates with ``quote`` populated and confidence_floor
        :data:`HEURISTIC_CONFIDENCE_QUOTE`.

      Pass 2 — Relation marker alone. Lower signal; produces candidates
        with ``quote=None`` and confidence_floor
        :data:`HEURISTIC_CONFIDENCE_RELATION_ONLY`. Matches within
        ``_PASS1_DEDUPE_WINDOW_CHARS`` of a Pass-1 hit are dropped to
        avoid double-counting the same attribution.

    A future Pass 3 (standalone aphorism in advice context) is
    intentionally deferred — aphorism-shape alone has high false-
    positive rate, and the D2 classifier is the right place to catch
    that case.

    Returns candidates sorted by ``position`` ascending. Empty list
    when no patterns match.

    Args:
        text: arbitrary text (typically a transcript chunk or diary
            entry body). No length cap; large inputs scan linearly.

    Returns:
        List of :class:`ProvenanceCandidate`, possibly empty.
    """

    candidates: list[ProvenanceCandidate] = []

    # Pass 1: relation + (optional attribution verb) + quote.
    for m in _RELATION_ATTRIBUTION_QUOTE_RE.finditer(text):
        relation = m.group(1).lower()
        quote = m.group(2).strip() if m.group(2) is not None else None
        candidates.append(
            ProvenanceCandidate(
                text=m.group(0),
                person_hint=relation,
                relation_hint=relation,
                quote=quote,
                position=m.start(),
                confidence_floor=HEURISTIC_CONFIDENCE_QUOTE,
            )
        )

    # Pass 2: bare relation marker. Skip overlaps with Pass-1.
    pass1_positions = {c.position for c in candidates}
    for m in _RELATION_ONLY_RE.finditer(text):
        if any(abs(m.start() - p) < _PASS1_DEDUPE_WINDOW_CHARS for p in pass1_positions):
            continue
        relation = m.group(1).lower()
        candidates.append(
            ProvenanceCandidate(
                text=m.group(0),
                person_hint=relation,
                relation_hint=relation,
                quote=None,
                position=m.start(),
                confidence_floor=HEURISTIC_CONFIDENCE_RELATION_ONLY,
            )
        )

    # Pass 3: Capitalized bare-relation as subject + attribution + quote.
    # "Dad always told me 'never trust a smiling investor'" — case Pass-1
    # missed because there's no possessive prefix. Skip overlaps with
    # Pass-1 / Pass-2 matches at the same position.
    existing_positions = {c.position for c in candidates}
    for m in _BARE_RELATION_ATTRIBUTION_QUOTE_RE.finditer(text):
        if any(abs(m.start() - p) < _PASS1_DEDUPE_WINDOW_CHARS for p in existing_positions):
            continue
        relation = m.group(1).lower()
        quote = m.group(2).strip() if m.group(2) is not None else None
        candidates.append(
            ProvenanceCandidate(
                text=m.group(0),
                person_hint=relation,
                relation_hint=relation,
                quote=quote,
                position=m.start(),
                # Same floor as Pass 1 — capitalize-only + attribution +
                # quote is high-signal even without possessive prefix.
                confidence_floor=HEURISTIC_CONFIDENCE_QUOTE,
            )
        )

    candidates.sort(key=lambda c: c.position)
    return candidates


# ---------------------------------------------------------------------------
# Public API: validate_candidate
# ---------------------------------------------------------------------------

# Type alias for the v1 classifier interface. Callable accepts the
# context-text window and returns a dict with at minimum
# ``is_provenance: bool``. See :func:`validate_candidate` for the full
# expected schema.
ClassifierFn = Callable[[str], dict]


def _stub_classifier_via_candidate(candidate: ProvenanceCandidate) -> dict:
    """D1 default classifier — accepts the candidate at confidence 0.5.

    Returns a dict shaped like the v1 classifier interface, but with
    fields lifted directly from the candidate (since the heuristic
    already extracted them). D2 replaces this with a real
    local-substrate call where the LLM does the field extraction.

    Confidence 0.5 is intentionally middle-of-range: callers who want
    only high-confidence records can apply a threshold; tests that
    just want to exercise the pipeline get a record back.
    """

    return {
        "is_provenance": True,
        "person": candidate.person_hint,
        "relation_type": candidate.relation_hint,
        "quote": candidate.quote,
        "confidence": 0.5,
    }


def validate_candidate(
    candidate: ProvenanceCandidate,
    ctx: str,
    classifier: Optional[ClassifierFn] = None,
    *,
    extractor_label: Optional[str] = None,
) -> Optional[ProvenanceRecord]:
    """Validate a heuristic candidate; return a record or ``None``.

    Args:
        candidate: the heuristic flag from :func:`extract_candidates`.
        ctx: the surrounding context text (typically ±200 chars
            around the candidate's position in the source). Becomes
            :attr:`ProvenanceRecord.context` on accept.
        classifier: optional v1 classifier callable. Signature
            ``Callable[[str], dict]``. Expected return keys:

              - ``is_provenance: bool`` (required) — accept/reject
              - ``person: str?`` — overrides candidate.person_hint
              - ``relation_type: str?`` — overrides candidate.relation_hint
              - ``quote: str?`` — overrides candidate.quote
              - ``confidence: float`` (recommended) — default 0.0

            When ``None`` (default), uses the D1 stub which accepts
            every candidate at confidence 0.5 with fields lifted from
            the candidate.

        extractor_label: provenance string for
            :attr:`ProvenanceRecord.extracted_by`. Defaults vary by
            classifier presence: ``"heuristic_v1+stub_classifier_v1"``
            when no classifier, ``"heuristic_v1+custom_classifier"``
            otherwise.

    Returns:
        :class:`ProvenanceRecord` when the classifier accepts AND
        person + relation_type can be resolved. ``None`` when the
        classifier rejects OR person/relation_type is missing.

        Caller decides whether the returned record's confidence meets
        the threshold for write to wing_lineage. v1 design doc §D1
        starts the threshold at 0.7.

    The classifier exception path is failure-soft: any exception raised
    by the classifier callable is caught, logged at DEBUG, and treated
    as rejection (returns ``None``). The mining pipeline should never
    crash because a classifier had a bad day.
    """

    if classifier is None:
        try:
            result = _stub_classifier_via_candidate(candidate)
        except Exception:  # noqa: BLE001 — stub is failure-soft contract
            logger.debug("provenance: stub classifier failed", exc_info=True)
            return None
        if extractor_label is None:
            extractor_label = "heuristic_v1+stub_classifier_v1"
    else:
        try:
            result = classifier(ctx)
        except Exception:  # noqa: BLE001 — classifier failure must not crash mining
            logger.debug("provenance: classifier raised", exc_info=True)
            return None
        if extractor_label is None:
            extractor_label = "heuristic_v1+custom_classifier"

    if not isinstance(result, dict):
        logger.debug(
            "provenance: classifier returned non-dict %r — rejecting",
            type(result),
        )
        return None

    if not result.get("is_provenance", False):
        return None

    person = result.get("person") or candidate.person_hint
    relation_type = result.get("relation_type") or candidate.relation_hint
    quote = result.get("quote") or candidate.quote or ""

    if not person or not relation_type:
        logger.debug(
            "provenance: missing person/relation_type — rejecting candidate %r",
            candidate.text,
        )
        return None

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return ProvenanceRecord(
        person=str(person),
        relation_type=str(relation_type),
        quote=str(quote),
        context=ctx,
        confidence=confidence,
        extracted_by=extractor_label,
    )


# ---------------------------------------------------------------------------
# Schema documentation — wing_lineage drawer shape
# ---------------------------------------------------------------------------

WING_LINEAGE_SCHEMA_DOC = """\
wing_lineage drawer schema (Phase 1 D1, per Provenance-Preservation-Design §D3)
================================================================================

Persistence layer:

  wing: wing_lineage
  room: <person_label>     # "father", "marie", "roshi", "mother-of-jms", ...
  content: |
    PROVENANCE:
    Person: <person_name_or_label>
    Relation: <father|mother|wife|teacher|roshi|...>
    Quote: "<exact quote>"
    Context: <when/how it was said, if known>
    Source: <source_session_id_or_drawer_id>
    Transmitted_into: <which operational drawer/strategy/identity-element this fed>
  metadata:
    person: <stable lowercased identifier>
    relation_type: <enum: family | teacher | partner | colleague | friend | fictional>
    is_quote: bool
    confidence: float
    extracted_by: <heuristic_v1 | classifier_v1 | manual | ...>
    source_session: <session_id>
    source_drawer: <drawer_id_if_extracted_from_existing_drawer>

The ``room`` field carries the person-label so a search like
``mempalace_search wing=wing_lineage room=father`` returns everything
attributed to James's father across sessions. Multiple drawers per
person are expected — one per distinct attribution event.

Writes go through the existing ``mempalace_add_drawer`` MCP tool —
this module produces :class:`ProvenanceRecord` instances; the D3
mining-integration envelope handles the persistence call.
"""
