"""Spreading activation through entity links in the proof network.

Priority-queue BFS from seed entities, with decaying activation
through entity_links (both directions). Bank-scoped to prevent
semantic bleeding across mathematical domains.

Extracted from proof_network.py per LintGate file-too-long split proposal.
"""

from __future__ import annotations

import heapq
import sqlite3
from typing import Sequence


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
