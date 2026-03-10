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

import heapq
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Sequence

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
    file_path   TEXT NOT NULL DEFAULT ''
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
"""

# Missing bank penalty: not zero, but penalized.
_MISSING_BANK_SCORE = 0.3


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


def navigate(
    conn: sqlite3.Connection,
    query: StructuredQuery,
    limit: int = 16,
    mechanism: str = "confidence_weighted",
    entity_type: str | None = None,
) -> list[ScoredEntity]:
    """Retrieve entities from the proof network matching a navigational query.

    Steps:
        1. Pre-filter by accessible premises (if theorem_id given)
        2. Pre-filter by required anchors (hard filter)
        3. Batch-fetch positions and anchor sets for candidates
        4. Score: bank_alignment × anchor_relevance × seed_similarity
        5. Return top-k by final_score
    """
    # Step 1+2: Build candidate set
    candidate_ids = _get_candidates(conn, query, entity_type)
    if not candidate_ids:
        return []

    # Step 3: Batch-fetch positions and anchors
    positions = _batch_get_positions(conn, candidate_ids)
    entity_anchor_sets = _batch_get_anchor_sets(conn, candidate_ids)
    idf_cache = _get_idf_cache(conn)
    names = _batch_get_names(conn, candidate_ids)

    # Precompute seed anchor set for seed similarity
    seed_anchors: set[int] = set()
    if query.seed_entity_ids:
        for sid in query.seed_entity_ids:
            seed_anchors.update(entity_anchor_sets.get(sid, set()))

    # Step 4: Score each candidate
    results: list[ScoredEntity] = []
    for eid in candidate_ids:
        b_score = _compute_bank_score(positions.get(eid, {}), query, mechanism)
        a_score = _compute_anchor_score(entity_anchor_sets.get(eid, set()), query, idf_cache)
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


def get_accessible_premises(conn: sqlite3.Connection, theorem_id: int) -> set[int]:
    """Return the set of entity IDs accessible to a given theorem."""
    rows = conn.execute(
        "SELECT premise_id FROM accessible_premises WHERE theorem_id = ?",
        (theorem_id,),
    ).fetchall()
    return {r[0] for r in rows}


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
    """Build candidate entity IDs from filters."""
    # Start with accessible premises if specified
    if query.accessible_theorem_id is not None:
        accessible = get_accessible_premises(conn, query.accessible_theorem_id)
        if not accessible:
            return []
        candidate_ids = accessible
    else:
        # All entities (optionally filtered by type)
        if entity_type:
            rows = conn.execute(
                "SELECT id FROM entities WHERE entity_type = ?",
                (entity_type,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id FROM entities").fetchall()
        candidate_ids = {r[0] for r in rows}

    # Hard filter by required anchors
    if query.require_anchors:
        placeholders = ",".join("?" * len(query.require_anchors))
        rows = conn.execute(
            f"""
            SELECT entity_id FROM entity_anchors
            WHERE anchor_id IN ({placeholders})
            GROUP BY entity_id
            HAVING COUNT(DISTINCT anchor_id) = ?
            """,
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
        f"""
        SELECT entity_id, bank, sign * depth
        FROM entity_positions
        WHERE entity_id IN ({placeholders})
        """,
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
        f"SELECT entity_id, anchor_id FROM entity_anchors WHERE entity_id IN ({placeholders})",
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
        f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
        entity_ids,
    ).fetchall()
    return dict(rows)


def _get_idf_cache(conn: sqlite3.Connection) -> dict[int, float]:
    """Load the full IDF table into memory."""
    rows = conn.execute("SELECT anchor_id, idf_value FROM anchor_idf").fetchall()
    return dict(rows)


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
