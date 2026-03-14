"""
Proof network — SQLite semantic network of mathematical entities.

Adapted from ModelAtlas's db.py for mathematical entities: lemmas, tactics,
and proof states positioned along 6 signed orthogonal banks, connected
through a shared anchor dictionary with IDF weighting.

Core operations:
    navigate()  — bank-aligned, IDF-weighted entity retrieval
    spread()    — priority-queue activation through entity links
    compose_bank_scores() — configurable scoring composition
"""

from __future__ import annotations

import functools
import heapq
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.nav_contracts import BANK_NAMES, ScoredEntity, StructuredQuery

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'lemma',  -- lemma, tactic, definition
    namespace   TEXT NOT NULL DEFAULT '',
    file_path   TEXT NOT NULL DEFAULT '',
    provenance  TEXT NOT NULL DEFAULT 'traced'   -- traced, premise_only, tactic
);

CREATE TABLE IF NOT EXISTS entity_positions (
    entity_id   INTEGER NOT NULL,
    bank        TEXT NOT NULL,
    sign        INTEGER NOT NULL,  -- -1, 0, +1
    depth       INTEGER NOT NULL DEFAULT 0,  -- 0..3
    PRIMARY KEY (entity_id, bank),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS anchors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL UNIQUE,
    category    TEXT NOT NULL DEFAULT 'general'
);

CREATE TABLE IF NOT EXISTS entity_anchors (
    entity_id   INTEGER NOT NULL,
    anchor_id   INTEGER NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (entity_id, anchor_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (anchor_id) REFERENCES anchors(id)
);

CREATE TABLE IF NOT EXISTS entity_links (
    source_id   INTEGER NOT NULL,
    target_id   INTEGER NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS anchor_idf (
    anchor_id   INTEGER PRIMARY KEY,
    idf_value   REAL NOT NULL,
    FOREIGN KEY (anchor_id) REFERENCES anchors(id)
);

CREATE TABLE IF NOT EXISTS accessible_premises (
    theorem_id  INTEGER NOT NULL,
    premise_id  INTEGER NOT NULL,
    PRIMARY KEY (theorem_id, premise_id),
    FOREIGN KEY (theorem_id) REFERENCES entities(id),
    FOREIGN KEY (premise_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_entity_positions_bank
    ON entity_positions(bank, sign);
CREATE INDEX IF NOT EXISTS idx_entity_anchors_anchor
    ON entity_anchors(anchor_id);
CREATE INDEX IF NOT EXISTS idx_entity_anchors_entity
    ON entity_anchors(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_source
    ON entity_links(source_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_target
    ON entity_links(target_id);
CREATE INDEX IF NOT EXISTS idx_accessible_premises_theorem
    ON accessible_premises(theorem_id);
CREATE INDEX IF NOT EXISTS idx_entities_type
    ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_provenance
    ON entities(provenance);
"""

# Missing bank penalty: not zero, but penalized.
_MISSING_BANK_SCORE = 0.3

# Cache entity ID sets by (conn_id, entity_type) to avoid re-querying
# 242K IDs on every navigate() call. Cleared by clear_caches().
_entity_id_cache: dict[tuple[int, str | None], set[int]] = {}

# In-memory data cache: pre-load positions, anchor sets, and names for all
# entities on first navigate() call. Eliminates SQLite I/O on subsequent calls.
# Key: conn_id. Cleared by clear_caches().
class _DataCache:
    """Typed container for pre-loaded entity data."""

    __slots__ = ("positions", "anchor_sets", "names")

    def __init__(
        self,
        positions: dict[int, dict[str, int]],
        anchor_sets: dict[int, set[int]],
        names: dict[int, str],
    ) -> None:
        self.positions = positions
        self.anchor_sets = anchor_sets
        self.names = names


_data_cache: dict[int, _DataCache] = {}


def _get_data_cache(conn: sqlite3.Connection) -> _DataCache:
    """Load all entity data into memory on first call, return cached data.

    Pre-loads:
        positions: {eid: {bank: signed_pos}}
        anchor_sets: {eid: {anchor_id, ...}}
        names: {eid: name}

    On a 242K entity DB this costs ~3 seconds and ~50MB RAM but
    eliminates ~3100ms of SQLite I/O per navigate() call.
    """
    conn_id = id(conn)
    if conn_id in _data_cache:
        return _data_cache[conn_id]

    # Load all positions
    positions: dict[int, dict[str, int]] = defaultdict(dict)
    for eid, bank, signed_pos in conn.execute(
        "SELECT entity_id, bank, sign * depth FROM entity_positions"
    ).fetchall():
        positions[eid][bank] = signed_pos

    # Load all anchor sets
    anchor_sets: dict[int, set[int]] = defaultdict(set)
    for eid, aid in conn.execute(
        "SELECT entity_id, anchor_id FROM entity_anchors"
    ).fetchall():
        anchor_sets[eid].add(aid)

    # Load all names
    names: dict[int, str] = dict(
        conn.execute("SELECT id, name FROM entities").fetchall()
    )

    cache = _DataCache(
        positions=dict(positions),
        anchor_sets=dict(anchor_sets),
        names=names,
    )
    _data_cache[conn_id] = cache
    return cache


def clear_caches() -> None:
    """Clear all module-level caches. Call between test cases or DB swaps."""
    _idf_cache.clear()
    _accessible_cache.clear()
    _entity_id_cache.clear()
    _data_cache.clear()
    bank_score.cache_clear()


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


def init_db(path: str | Path) -> sqlite3.Connection:
    """Create or open a proof network database and ensure schema exists."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def bank_score(entity_signed_pos: int, query_direction: int) -> float:
    """Score a single bank alignment between entity position and query direction."""
    if query_direction == 0:
        return 1.0 / (1.0 + abs(entity_signed_pos))
    alignment = entity_signed_pos * query_direction
    if alignment > 0:
        return 1.0
    if alignment == 0:
        return 0.5
    return 1.0 / (1.0 + abs(alignment))


def compose_bank_scores(
    scores: dict[str, float],
    confidences: dict[str, float],
    mechanism: str = "confidence_weighted",
    floor_epsilon: float = 0.1,
) -> float:
    """Compose per-bank scores into a single alignment score.

    Mechanisms:
        multiplicative      — pure product (precise but fragile)
        confidence_weighted — Π score_i^confidence_i (default, graceful)
        soft_floor          — Π max(score_i, epsilon)
        geometric_mean      — Π^(1/n)
        log_additive        — exp(Σ log(score_i))
    """
    if not scores:
        return 1.0

    vals = list(scores.values())

    if mechanism == "multiplicative":
        return math.prod(vals)

    if mechanism == "confidence_weighted":
        return math.prod(s ** confidences.get(bank, 1.0) for bank, s in scores.items())

    if mechanism == "soft_floor":
        return math.prod(max(s, floor_epsilon) for s in vals)

    if mechanism == "geometric_mean":
        return math.prod(vals) ** (1.0 / len(vals))

    if mechanism == "log_additive":
        return math.exp(sum(math.log(max(s, 1e-6)) for s in vals))

    msg = f"Unknown scoring mechanism: {mechanism}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Navigation (core retrieval)
# ---------------------------------------------------------------------------


def _vectorized_bank_scores(
    candidate_ids: list[int],
    positions: dict[int, dict[str, int]],
    query: StructuredQuery,
) -> np.ndarray:
    """Compute bank scores for all candidates using NumPy vectorization.

    Returns 1D array of composite bank scores (one per candidate).
    Uses the confidence_weighted mechanism: prod(score_i^confidence_i).
    Falls back to per-entity computation for non-standard mechanisms.
    """
    n = len(candidate_ids)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Build position matrix: (n_candidates, n_banks) of signed positions
    n_banks = len(BANK_NAMES)
    pos_matrix = np.full((n, n_banks), np.nan, dtype=np.float64)
    for i, eid in enumerate(candidate_ids):
        epos = positions.get(eid, {})
        for j, bank in enumerate(BANK_NAMES):
            if bank in epos:
                pos_matrix[i, j] = epos[bank]

    # Query direction vector
    directions = np.array(
        [query.bank_directions.get(b, 0) for b in BANK_NAMES], dtype=np.float64
    )
    confidences = np.array(
        [query.bank_confidences.get(b, 1.0) for b in BANK_NAMES], dtype=np.float64
    )

    # Vectorized bank_score computation
    # For each (entity_pos, query_dir) pair:
    #   dir == 0: 1.0 / (1.0 + abs(pos))
    #   alignment > 0: 1.0
    #   alignment == 0: 0.5
    #   alignment < 0: 1.0 / (1.0 + abs(alignment))
    alignment = pos_matrix * directions  # (n, n_banks)

    scores = np.where(
        np.isnan(pos_matrix),
        _MISSING_BANK_SCORE,  # missing position
        np.where(
            directions == 0,
            1.0 / (1.0 + np.abs(np.nan_to_num(pos_matrix, nan=0.0))),  # query doesn't care
            np.where(
                alignment > 0, 1.0,  # right side
                np.where(
                    alignment == 0, 0.5,  # neutral
                    1.0 / (1.0 + np.abs(alignment)),  # wrong side
                ),
            ),
        ),
    )  # (n, n_banks)

    # Skip banks where direction==0 AND position is missing (don't penalize)
    skip_mask = (directions == 0) & np.isnan(pos_matrix)
    scores[skip_mask] = 1.0  # neutral contribution

    # Confidence-weighted composition: prod(score_i^confidence_i)
    log_scores = np.log(np.maximum(scores, 1e-10)) * confidences  # (n, n_banks)
    composite = np.exp(np.sum(log_scores, axis=1))  # (n,)

    return composite


def _score_candidates(
    candidate_ids: list[int],
    positions: dict[int, dict[str, int]],
    entity_anchor_sets: dict[int, set[int]],
    idf_cache: dict[int, float],
    names: dict[int, str],
    query: StructuredQuery,
    seed_anchors: set[int],
    mechanism: str,
) -> list[ScoredEntity]:
    """Score and rank candidate entities against a navigational query.

    Uses NumPy vectorization for bank scoring (the hot inner loop),
    with per-entity anchor/seed scoring. Only constructs ScoredEntity
    objects for candidates with final_score > 0.
    """
    if not candidate_ids:
        return []

    # Vectorized bank scoring (replaces 242K × 6 Python calls)
    if mechanism == "confidence_weighted":
        bank_scores = _vectorized_bank_scores(candidate_ids, positions, query)
    else:
        # Fallback for non-standard mechanisms
        bank_scores = np.array([
            _compute_bank_score(positions.get(eid, {}), query, mechanism)
            for eid in candidate_ids
        ])

    # Early pruning: skip candidates with zero bank score
    nonzero_mask = bank_scores > 0
    nonzero_indices = np.nonzero(nonzero_mask)[0]

    results: list[ScoredEntity] = []
    for idx in nonzero_indices:
        eid = candidate_ids[idx]
        b_score = float(bank_scores[idx])
        a_score = _compute_anchor_score(entity_anchor_sets.get(eid, set()), query, idf_cache)
        if a_score <= 0:
            continue
        s_score = _compute_seed_score(entity_anchor_sets.get(eid, set()), seed_anchors, idf_cache)
        final = b_score * a_score * s_score
        if final > 0:
            results.append(
                ScoredEntity(
                    entity_id=eid,
                    name=names.get(eid, ""),
                    final_score=final,
                    bank_score=b_score,
                    anchor_score=a_score,
                    seed_score=s_score,
                )
            )
    results.sort(key=lambda e: e.final_score, reverse=True)
    return results


def navigate(
    conn: sqlite3.Connection,
    query: StructuredQuery,
    limit: int = 16,
    mechanism: str = "confidence_weighted",
    entity_type: str | None = None,
) -> list[ScoredEntity]:
    """Retrieve entities from the proof network matching a navigational query.

    Orchestration: fetches data from DB, delegates scoring to _score_candidates.
    """
    # Fetch candidate set from DB
    candidate_ids = _get_candidates(conn, query, entity_type)
    if not candidate_ids:
        return []

    # Use in-memory data cache (pre-loaded on first call, ~3s startup, 0ms thereafter)
    data = _get_data_cache(conn)
    positions = data.positions
    entity_anchor_sets = data.anchor_sets
    names = data.names
    idf_cache = _get_idf_cache(conn)

    # Precompute seed anchor set
    seed_anchors: set[int] = set()
    if query.seed_entity_ids:
        for sid in query.seed_entity_ids:
            seed_anchors.update(entity_anchor_sets.get(sid, set()))

    # Pure scoring
    results = _score_candidates(
        candidate_ids,
        positions,
        entity_anchor_sets,
        idf_cache,
        names,
        query,
        seed_anchors,
        mechanism,
    )
    return results[:limit]


# ---------------------------------------------------------------------------
# Spreading activation
# ---------------------------------------------------------------------------


def spread(
    conn: sqlite3.Connection,
    seed_ids: Sequence[int],
    max_depth: int = 3,
    decay: float = 0.8,
    neighbor_slice: int = 20,
) -> dict[int, float]:
    """Spread activation from seed entities through entity links.

    Uses priority-queue BFS with decaying activation.
    Returns entity_id -> activation score.
    """
    activation: dict[int, float] = {}
    # Initialize seeds at activation 1.0
    frontier: list[tuple[float, int, int]] = []  # (-activation, entity_id, depth)
    for sid in seed_ids:
        activation[sid] = 1.0
        frontier.append((-1.0, sid, 0))

    # BFS with priority (highest activation first)
    heapq.heapify(frontier)

    while frontier:
        neg_act, eid, depth = heapq.heappop(frontier)
        current_act = -neg_act

        if depth >= max_depth:
            continue

        # Get neighbors via links (both directions)
        neighbors = _get_link_neighbors(conn, eid, neighbor_slice)
        for neighbor_id, weight in neighbors:
            new_act = current_act * weight * decay
            if new_act > activation.get(neighbor_id, 0):
                activation[neighbor_id] = new_act
                heapq.heappush(frontier, (-new_act, neighbor_id, depth + 1))

    return activation


# ---------------------------------------------------------------------------
# Accessible premises
# ---------------------------------------------------------------------------


_accessible_cache: dict[tuple[int, int], set[int]] = {}


def get_accessible_premises(conn: sqlite3.Connection, theorem_id: int) -> set[int]:
    """Return the set of entity IDs accessible to a given theorem (cached)."""
    key = (id(conn), theorem_id)
    if key not in _accessible_cache:
        rows = conn.execute(
            "SELECT premise_id FROM accessible_premises WHERE theorem_id = ?",
            (theorem_id,),
        ).fetchall()
        _accessible_cache[key] = {r[0] for r in rows}
    return _accessible_cache[key]


# ---------------------------------------------------------------------------
# IDF computation
# ---------------------------------------------------------------------------


def recompute_idf(conn: sqlite3.Connection) -> None:
    """Recompute IDF values for all anchors. Call after batch entity updates."""
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if total == 0:
        return
    log_total = math.log(total)

    conn.execute("DELETE FROM anchor_idf")
    conn.executemany(
        "INSERT INTO anchor_idf (anchor_id, idf_value) VALUES (?, ?)",
        conn.execute(
            """
            SELECT a.id, ? - LOG(MAX(COUNT(ea.entity_id), 1))
            FROM anchors a
            LEFT JOIN entity_anchors ea ON ea.anchor_id = a.id
            GROUP BY a.id
            """,
            (log_total,),
        ).fetchall(),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _get_candidates(
    conn: sqlite3.Connection,
    query: StructuredQuery,
    entity_type: str | None,
) -> list[int]:
    """Build candidate entity IDs from filters.

    Entity ID sets are cached per (connection, entity_type) to avoid
    re-querying 242K rows on every navigate() call. Cache is cleared
    by clear_caches().
    """
    # Start with accessible premises if specified
    if query.accessible_theorem_id is not None:
        accessible = get_accessible_premises(conn, query.accessible_theorem_id)
        if not accessible:
            return []
        candidate_ids = accessible
    else:
        # Cache entity ID sets to avoid re-querying 242K rows per navigate() call
        cache_key = (id(conn), entity_type)
        if cache_key in _entity_id_cache:
            candidate_ids = _entity_id_cache[cache_key]
        elif entity_type:
            rows = conn.execute(
                "SELECT id FROM entities WHERE entity_type = ?",
                (entity_type,),
            ).fetchall()
            candidate_ids = {r[0] for r in rows}
            _entity_id_cache[cache_key] = candidate_ids
        else:
            rows = conn.execute("SELECT id FROM entities").fetchall()
            candidate_ids = {r[0] for r in rows}
            _entity_id_cache[cache_key] = candidate_ids

    # Hard filter by required anchors
    if query.require_anchors:
        placeholders = ",".join("?" * len(query.require_anchors))
        rows = conn.execute(
            f"SELECT entity_id FROM entity_anchors "  # nosec B608
            f"WHERE anchor_id IN ({placeholders}) "
            f"GROUP BY entity_id "
            f"HAVING COUNT(DISTINCT anchor_id) = ?",
            [*query.require_anchors, len(query.require_anchors)],
        ).fetchall()
        required_set = {r[0] for r in rows}
        candidate_ids = candidate_ids & required_set

    return list(candidate_ids)


def _batch_get_positions(
    conn: sqlite3.Connection, entity_ids: list[int]
) -> dict[int, dict[str, int]]:
    """Fetch bank positions for a batch of entities. Returns {eid: {bank: signed_pos}}."""
    if not entity_ids:
        return {}
    placeholders = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"""SELECT entity_id, bank, sign * depth
        FROM entity_positions
        WHERE entity_id IN ({placeholders})
        """,  # nosec B608
        entity_ids,
    ).fetchall()
    result: dict[int, dict[str, int]] = defaultdict(dict)
    for eid, bank, signed_pos in rows:
        result[eid][bank] = signed_pos
    return dict(result)


def _batch_get_anchor_sets(conn: sqlite3.Connection, entity_ids: list[int]) -> dict[int, set[int]]:
    """Fetch anchor ID sets for a batch of entities."""
    if not entity_ids:
        return {}
    placeholders = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT entity_id, anchor_id FROM entity_anchors WHERE entity_id IN ({placeholders})",  # nosec B608
        entity_ids,
    ).fetchall()
    result: dict[int, set[int]] = defaultdict(set)
    for eid, aid in rows:
        result[eid].add(aid)
    return dict(result)


def _batch_get_names(conn: sqlite3.Connection, entity_ids: list[int]) -> dict[int, str]:
    """Fetch names for a batch of entities."""
    if not entity_ids:
        return {}
    placeholders = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"SELECT id, name FROM entities WHERE id IN ({placeholders})",  # nosec B608
        entity_ids,
    ).fetchall()
    return dict(rows)


_idf_cache: dict[int, dict[int, float]] = {}


def _get_idf_cache(conn: sqlite3.Connection) -> dict[int, float]:
    """Load the full IDF table into memory (cached per connection)."""
    conn_id = id(conn)
    if conn_id not in _idf_cache:
        rows = conn.execute("SELECT anchor_id, idf_value FROM anchor_idf").fetchall()
        _idf_cache[conn_id] = dict(rows)
    return _idf_cache[conn_id]


def _get_link_neighbors(
    conn: sqlite3.Connection, entity_id: int, limit: int
) -> list[tuple[int, float]]:
    """Get linked neighbors of an entity (both directions), ordered by weight."""
    rows = conn.execute(
        """
        SELECT target_id, weight FROM entity_links WHERE source_id = ?
        UNION
        SELECT source_id, weight FROM entity_links WHERE target_id = ?
        ORDER BY weight DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _compute_bank_score(
    positions: dict[str, int],
    query: StructuredQuery,
    mechanism: str,
) -> float:
    """Compute composite bank alignment score for one entity."""
    scores: dict[str, float] = {}
    for bank_name in BANK_NAMES:
        direction = query.bank_directions.get(bank_name, 0)
        if direction == 0 and bank_name not in positions:
            continue  # query doesn't care and entity has no position
        entity_pos = positions.get(bank_name)
        if entity_pos is None:
            scores[bank_name] = _MISSING_BANK_SCORE
        else:
            scores[bank_name] = bank_score(entity_pos, direction)

    return compose_bank_scores(scores, query.bank_confidences, mechanism)


def _compute_anchor_score(
    entity_anchors: set[int],
    query: StructuredQuery,
    idf_cache: dict[int, float],
) -> float:
    """Compute anchor relevance score (IDF-weighted)."""
    if not query.prefer_anchors:
        return 1.0  # no anchor preference → neutral

    total_idf = 0.0
    matched_idf = 0.0
    for aid, weight in zip(query.prefer_anchors, query.prefer_weights):
        idf = idf_cache.get(aid, 1.0) * weight
        total_idf += idf
        if aid in entity_anchors:
            matched_idf += idf

    # Avoid penalty
    avoid_penalty = 1.0
    for aid in query.avoid_anchors:
        if aid in entity_anchors:
            avoid_penalty *= 0.5

    if total_idf == 0:
        return avoid_penalty
    return (matched_idf / total_idf) * avoid_penalty


def _compute_seed_score(
    entity_anchors: set[int],
    seed_anchors: set[int],
    idf_cache: dict[int, float],
) -> float:
    """Compute IDF-weighted Jaccard similarity with seed entities."""
    if not seed_anchors:
        return 1.0  # no seeds → neutral

    shared = entity_anchors & seed_anchors
    union = entity_anchors | seed_anchors
    if not union:
        return 1.0

    shared_idf = sum(idf_cache.get(a, 1.0) for a in shared)
    union_idf = sum(idf_cache.get(a, 1.0) for a in union)
    if union_idf == 0:
        return 1.0
    return shared_idf / union_idf
