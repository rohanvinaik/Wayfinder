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
    compute_lens_coherence,
    compute_lens_scores,
    compute_observability_score,
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

    __slots__ = (
        "positions",
        "anchor_sets",
        "names",
        "provenances",
        "anchor_categories",
        "anchor_confidences",
        "hub_in_degrees",
        "landmark_profiles",
    )

    def __init__(
        self,
        positions: dict[int, dict[str, int]],
        anchor_sets: dict[int, set[int]],
        names: dict[int, str],
        provenances: dict[int, str],
        anchor_categories: dict[int, str] | None = None,
        anchor_confidences: dict[int, dict[int, float]] | None = None,
        hub_in_degrees: dict[int, int] | None = None,
        landmark_profiles: dict[int, dict] | None = None,
    ) -> None:
        self.positions = positions
        self.anchor_sets = anchor_sets
        self.names = names
        self.provenances = provenances
        self.anchor_categories = anchor_categories or {}
        self.anchor_confidences = anchor_confidences or {}
        self.hub_in_degrees = hub_in_degrees or {}
        self.landmark_profiles = landmark_profiles or {}


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

    # Load all names and provenances
    names: dict[int, str] = {}
    provenances: dict[int, str] = {}
    for eid, name, prov in conn.execute("SELECT id, name, provenance FROM entities").fetchall():
        names[eid] = name
        provenances[eid] = prov

    # Load anchor categories (anchor_id → category)
    anchor_categories: dict[int, str] = {}
    for aid, cat in conn.execute("SELECT id, category FROM anchors").fetchall():
        anchor_categories[aid] = cat

    # Load per-entity anchor confidences (entity_id → {anchor_id: confidence})
    anchor_confidences: dict[int, dict[int, float]] = defaultdict(dict)
    for eid, aid, conf in conn.execute(
        "SELECT entity_id, anchor_id, confidence FROM entity_anchors"
    ).fetchall():
        anchor_confidences[eid][aid] = conf

    # Load hub in-degrees (depends_on incoming edge counts)
    hub_in_degrees: dict[int, int] = {}
    try:
        for tid, cnt in conn.execute(
            "SELECT target_id, COUNT(*) FROM entity_links "
            "WHERE relation = 'depends_on' GROUP BY target_id"
        ).fetchall():
            hub_in_degrees[tid] = cnt
    except Exception:
        pass  # graceful: entity_links may not exist

    # Load landmark neighborhood profiles (graceful: table may not exist)
    landmark_profiles: dict[int, dict] = {}
    try:
        for row in conn.execute(
            "SELECT entity_id, outdegree, nbr_constant_anchor_count, "
            "nbr_structural_anchor_count, proof_motif, bank_signature "
            "FROM landmark_neighborhood_profiles"
        ).fetchall():
            landmark_profiles[row[0]] = {
                "outdegree": row[1],
                "nbr_constant_anchor_count": row[2],
                "nbr_structural_anchor_count": row[3],
                "proof_motif": row[4],
                "bank_signature": row[5],
            }
    except Exception:
        pass  # graceful: table may not exist

    cache = _DataCache(
        positions=dict(positions),
        anchor_sets=dict(anchor_sets),
        names=names,
        provenances=provenances,
        anchor_categories=anchor_categories,
        anchor_confidences=dict(anchor_confidences),
        hub_in_degrees=hub_in_degrees,
        landmark_profiles=landmark_profiles,
    )
    _data_cache[conn_id] = cache
    return cache


def clear_caches() -> None:
    """Clear all module-level caches. Call between test cases or DB swaps.

    Also clears caches in downstream modules (retrieval_stages, landmark_selectors)
    to prevent stale neighborhood/accessibility data in long-lived processes.
    """
    _idf_cache.clear()
    _accessible_cache.clear()
    _entity_id_cache.clear()
    _data_cache.clear()
    bank_score.cache_clear()

    # Clear downstream module caches (import here to avoid circular deps)
    try:
        from src.landmark_selectors import clear_selector_caches
        from src.retrieval_stages import _neighborhood_cache

        clear_selector_caches()
        _neighborhood_cache.clear()
    except ImportError:
        pass  # modules may not be available in minimal test setups


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
    data: _DataCache,
    idf_cache: dict[int, float],
    query: StructuredQuery,
    seed_anchors: set[int],
    mechanism: str,
) -> list[ScoredEntity]:
    """Score and rank candidate entities against a navigational query.

    Uses NumPy vectorization for bank scoring (the hot inner loop),
    with per-entity anchor/seed/observability scoring.

    When anchor_categories is populated, uses multi-lens coherence scoring:
    per-category anchor overlap with geometric mean across populated lenses.
    Otherwise falls back to flat anchor scoring.
    """
    if not candidate_ids:
        return []

    # Vectorized bank scoring (replaces 242K × 6 Python calls)
    if mechanism == "confidence_weighted":
        bank_scores = _vectorized_bank_scores(candidate_ids, data.positions, query)
    else:
        bank_scores = np.array(
            [
                _compute_bank_score(data.positions.get(eid, {}), query, mechanism)
                for eid in candidate_ids
            ]
        )

    # Early pruning: skip candidates with zero bank score
    nonzero_mask = bank_scores > 0
    nonzero_indices = np.nonzero(nonzero_mask)[0]

    use_lenses = len(data.anchor_categories) > 0

    results: list[ScoredEntity] = []
    for idx in nonzero_indices:
        eid = candidate_ids[idx]
        b_score = float(bank_scores[idx])
        ea_set = data.anchor_sets.get(eid, set())

        if use_lenses and query.prefer_anchors:
            # Multi-lens coherence: per-category anchor overlap
            entity_confs = data.anchor_confidences.get(eid)
            lens_scores = compute_lens_scores(
                ea_set,
                query.prefer_anchors,
                query.prefer_weights,
                idf_cache,
                data.anchor_categories,
                entity_confs,
            )
            if lens_scores:
                a_score = compute_lens_coherence(lens_scores)
                # Avoid anchor penalty still applies
                for aid in query.avoid_anchors:
                    if aid in ea_set:
                        a_score *= 0.5
            else:
                # No typed categories matched — fall back to flat scoring
                # (handles legacy DBs where all anchors have category='general')
                a_score = _compute_anchor_score(ea_set, query, idf_cache)
        else:
            # Flat anchor scoring (v2 fallback)
            a_score = _compute_anchor_score(ea_set, query, idf_cache)

        if a_score <= 0:
            continue
        s_score = _compute_seed_score(ea_set, seed_anchors, idf_cache)

        # Observability: channel coverage, not just anchor count
        if data.provenances:
            o_score = compute_observability_score(
                data.positions.get(eid, {}),
                ea_set,
                data.provenances.get(eid, "traced"),
                data.anchor_categories,
            )
        else:
            o_score = 1.0

        final = b_score * a_score * s_score * o_score
        if final > 0:
            results.append(
                ScoredEntity(
                    entity_id=eid,
                    name=data.names.get(eid, ""),
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
    idf_cache = _get_idf_cache(conn)

    # Precompute seed anchor set
    seed_anchors: set[int] = set()
    if query.seed_entity_ids:
        for sid in query.seed_entity_ids:
            seed_anchors.update(data.anchor_sets.get(sid, set()))

    # Pure scoring with multi-lens coherence + observability
    results = _score_candidates(candidate_ids, data, idf_cache, query, seed_anchors, mechanism)
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
    """Recompute IDF values for all anchors (global + per-category).

    Global IDF: log(N / df) across all entities.
    Per-category IDF: log(N_cat / df_cat) within each anchor category,
    so anchors are weighted relative to peers in the same lens.
    """
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if total == 0:
        return
    log_total = math.log(total)

    # Global IDF (unchanged)
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

    # Per-category IDF table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS anchor_category_idf (
            anchor_id   INTEGER PRIMARY KEY,
            category    TEXT NOT NULL,
            idf_value   REAL NOT NULL,
            FOREIGN KEY (anchor_id) REFERENCES anchors(id)
        );
    """)
    conn.execute("DELETE FROM anchor_category_idf")

    # Get per-category entity counts
    categories = conn.execute("SELECT DISTINCT category FROM anchors").fetchall()

    for (cat,) in categories:
        # Count entities that have at least one anchor in this category
        cat_total_row = conn.execute(
            """
            SELECT COUNT(DISTINCT ea.entity_id)
            FROM entity_anchors ea
            JOIN anchors a ON a.id = ea.anchor_id
            WHERE a.category = ?
            """,
            (cat,),
        ).fetchone()
        cat_total = cat_total_row[0] if cat_total_row else 0
        if cat_total == 0:
            continue
        log_cat_total = math.log(cat_total)

        conn.executemany(
            "INSERT INTO anchor_category_idf (anchor_id, category, idf_value) VALUES (?, ?, ?)",
            conn.execute(
                """
                SELECT a.id, a.category, ? - LOG(MAX(COUNT(ea.entity_id), 1))
                FROM anchors a
                LEFT JOIN entity_anchors ea ON ea.anchor_id = a.id
                WHERE a.category = ?
                GROUP BY a.id
                """,
                (log_cat_total, cat),
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
