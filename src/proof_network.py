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

import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.nav_contracts import ScoredEntity, StructuredQuery
from src.proof_scoring import (  # noqa: F401
    _MISSING_BANK_SCORE,
    _compute_anchor_score,
    _compute_bank_score,
    _compute_seed_score,
    _vectorized_bank_scores,
    bank_score,
    compose_bank_scores,
)

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
    for eid, aid in conn.execute("SELECT entity_id, anchor_id FROM entity_anchors").fetchall():
        anchor_sets[eid].add(aid)

    # Load all names
    names: dict[int, str] = dict(conn.execute("SELECT id, name FROM entities").fetchall())

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


# Scoring — extracted to src/proof_scoring.py (bank_score, compose_bank_scores,
# _vectorized_bank_scores, _compute_bank_score, _compute_anchor_score, _compute_seed_score)


# ---------------------------------------------------------------------------
# Navigation (core retrieval)
# ---------------------------------------------------------------------------


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
        bank_scores = np.array(
            [_compute_bank_score(positions.get(eid, {}), query, mechanism) for eid in candidate_ids]
        )

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


# Spreading activation — extracted to src/proof_spreading.py
from src.proof_spreading import spread  # noqa: F401, E402

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


_idf_cache: dict[int, dict[int, float]] = {}


def _get_idf_cache(conn: sqlite3.Connection) -> dict[int, float]:
    """Load the full IDF table into memory (cached per connection)."""
    conn_id = id(conn)
    if conn_id not in _idf_cache:
        rows = conn.execute("SELECT anchor_id, idf_value FROM anchor_idf").fetchall()
        _idf_cache[conn_id] = dict(rows)
    return _idf_cache[conn_id]
