"""Build proof_network.db from proof_network_entities.jsonl.

Loads extracted entity records into the SQLite proof network schema
(entities, entity_positions, anchors, entity_anchors, accessible_premises)
and computes IDF values.

Usage:
    python -m scripts.build_proof_network_db \
        --input data/proof_network_entities.jsonl \
        --output data/proof_network.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.tactic_maps import TACTIC_ANCHORS, TACTIC_DIRECTIONS
from src.proof_network import init_db, recompute_idf


def _get_or_create_anchor(
    conn, label: str, anchor_cache: dict[str, int], category: str = "general"
) -> int:
    """Get or create an anchor, using cache to avoid repeated lookups."""
    if label in anchor_cache:
        return anchor_cache[label]
    row = conn.execute("SELECT id FROM anchors WHERE label = ?", (label,)).fetchone()
    if row:
        anchor_cache[label] = row[0]
        return row[0]
    conn.execute(
        "INSERT INTO anchors (label, category) VALUES (?, ?)",
        (label, category),
    )
    aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    anchor_cache[label] = aid
    return aid


def _collect_all_tactics(input_path: Path) -> set[str]:
    """Scan input JSONL and return all unique non-empty tactic names."""
    all_tactics: set[str] = set()
    for entity in _iter_entities(input_path):
        for tname in entity.get("tactic_names", []):
            if tname:
                all_tactics.add(tname)
    return all_tactics


def _insert_tactic_positions(conn, eid: int, tname: str) -> None:
    """Insert bank positions for a tactic entity from TACTIC_DIRECTIONS."""
    for bank, sign in TACTIC_DIRECTIONS.get(tname, {}).items():
        conn.execute(
            "INSERT INTO entity_positions (entity_id, bank, sign, depth) VALUES (?, ?, ?, ?)",
            (eid, bank, sign, 1),
        )


def _insert_tactic_anchors(conn, eid: int, tname: str, anchor_cache: dict[str, int]) -> None:
    """Insert anchors for a tactic entity from TACTIC_ANCHORS."""
    for anchor_label in TACTIC_ANCHORS.get(tname, []):
        aid = _get_or_create_anchor(conn, anchor_label, anchor_cache)
        conn.execute(
            "INSERT OR IGNORE INTO entity_anchors (entity_id, anchor_id) VALUES (?, ?)",
            (eid, aid),
        )


def _build_tactic_entities(
    conn,
    input_path: Path,
    anchor_cache: dict[str, int],
    name_to_id: dict[str, int],
) -> int:
    """Create tactic entities from tactic_names found in lemma records.

    Each unique tactic name becomes an entity with entity_type='tactic',
    bank positions from TACTIC_DIRECTIONS, and anchors from TACTIC_ANCHORS.
    """
    all_tactics = _collect_all_tactics(input_path)

    created = 0
    for tname in sorted(all_tactics):
        if tname in name_to_id:
            continue

        conn.execute(
            "INSERT INTO entities (name, entity_type, namespace, file_path, provenance) "
            "VALUES (?, ?, ?, ?, ?)",
            (tname, "tactic", "", "", "tactic"),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        name_to_id[tname] = eid

        _insert_tactic_positions(conn, eid, tname)
        _insert_tactic_anchors(conn, eid, tname, anchor_cache)

        created += 1

    conn.commit()
    return created


def _iter_entities(input_path: Path):
    """Yield parsed entity dicts from a JSONL file."""
    with open(input_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _insert_tactic_link(conn, tactic_id: int, lemma_id: int) -> None:
    """Insert a single tactic-to-lemma link."""
    conn.execute(
        "INSERT OR IGNORE INTO entity_links "
        "(source_id, target_id, relation, weight) "
        "VALUES (?, ?, ?, ?)",
        (tactic_id, lemma_id, "used_in", 0.5),
    )


def _link_tactics_for_entity(conn, entity: dict, name_to_id: dict[str, int]) -> int:
    """Insert tactic-lemma links for one entity. Returns count of links inserted."""
    lemma_id = name_to_id.get(entity["theorem_id"])
    if lemma_id is None:
        return 0
    count = 0
    for tname in set(entity.get("tactic_names", [])):
        tactic_id = name_to_id.get(tname)
        if tactic_id is not None and tactic_id != lemma_id:
            _insert_tactic_link(conn, tactic_id, lemma_id)
            count += 1
    return count


def _build_tactic_links(
    conn,
    input_path: Path,
    name_to_id: dict[str, int],
) -> int:
    """Create entity_links between lemmas and the tactics used in their proofs.

    These links enable spreading activation from known tactics to related lemmas.
    """
    link_count = 0
    for entity in _iter_entities(input_path):
        link_count += _link_tactics_for_entity(conn, entity, name_to_id)
        if link_count % 50000 == 0 and link_count > 0:
            conn.commit()

    conn.commit()
    return link_count


def _insert_entity_positions(conn, eid: int, positions: dict) -> None:
    """Insert bank positions for a lemma entity."""
    for bank, info in positions.items():
        conn.execute(
            "INSERT INTO entity_positions (entity_id, bank, sign, depth) VALUES (?, ?, ?, ?)",
            (eid, bank, info.get("sign", 0), info.get("depth", 0)),
        )


def _insert_entity_anchors(conn, eid: int, anchors: list, anchor_cache: dict[str, int]) -> None:
    """Insert anchors for a lemma entity.

    Supports both typed anchors ({label, category, confidence} dicts) and
    legacy plain string anchors for backward compatibility.
    """
    for anchor in anchors:
        if isinstance(anchor, dict):
            label = anchor["label"]
            category = anchor.get("category", "general")
            confidence = anchor.get("confidence", 1.0)
        else:
            label = anchor
            category = "general"
            confidence = 1.0
        aid = _get_or_create_anchor(conn, label, anchor_cache, category)
        conn.execute(
            "INSERT OR IGNORE INTO entity_anchors (entity_id, anchor_id, confidence) "
            "VALUES (?, ?, ?)",
            (eid, aid, confidence),
        )


def _insert_entity(
    conn,
    entity: dict,
    anchor_cache: dict[str, int],
    name_to_id: dict[str, int],
) -> bool:
    """Insert a single entity record into the DB. Returns True if inserted."""
    name = entity["theorem_id"]
    if name in name_to_id:
        return False

    conn.execute(
        "INSERT INTO entities (name, entity_type, namespace, file_path, provenance) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            name,
            entity.get("entity_type", "lemma"),
            entity.get("namespace", ""),
            entity.get("file_path", ""),
            entity.get("provenance", "traced"),
        ),
    )
    eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    name_to_id[name] = eid

    _insert_entity_positions(conn, eid, entity.get("positions", {}))
    _insert_entity_anchors(conn, eid, entity.get("anchors", []), anchor_cache)
    return True


def _load_lemma_entities(
    conn, input_path: Path, anchor_cache: dict[str, int], name_to_id: dict[str, int]
) -> int:
    """Load lemma entities from JSONL into DB."""
    processed = 0
    for entity in _iter_entities(input_path):
        if not _insert_entity(conn, entity, anchor_cache, name_to_id):
            continue
        processed += 1
        if processed % 10000 == 0:
            conn.commit()
            print(f"  Loaded {processed} entities...")
    conn.commit()
    return processed


def _link_premises_for_entity(conn, entity: dict, name_to_id: dict[str, int]) -> int:
    """Insert premise links for one entity. Returns count of links inserted."""
    eid = name_to_id.get(entity["theorem_id"])
    if eid is None:
        return 0
    count = 0
    for premise_name in entity.get("premises", []):
        pid = name_to_id.get(premise_name)
        if pid is not None and pid != eid:
            conn.execute(
                "INSERT OR IGNORE INTO accessible_premises (theorem_id, premise_id) VALUES (?, ?)",
                (eid, pid),
            )
            count += 1
    return count


def _build_premise_links(conn, input_path: Path, name_to_id: dict[str, int]) -> int:
    """Build accessible_premises links from entity premise metadata."""
    premise_links = 0
    for entity in _iter_entities(input_path):
        premise_links += _link_premises_for_entity(conn, entity, name_to_id)
        if premise_links % 50000 == 0 and premise_links > 0:
            conn.commit()
    conn.commit()
    return premise_links


def _print_db_summary(conn) -> None:
    """Print summary statistics for the built database."""
    n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_tactics = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE entity_type = 'tactic'"
    ).fetchone()[0]
    n_lemmas = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type = 'lemma'").fetchone()[
        0
    ]
    n_anchors = conn.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    n_positions = conn.execute("SELECT COUNT(*) FROM entity_positions").fetchone()[0]
    n_links = conn.execute("SELECT COUNT(*) FROM accessible_premises").fetchone()[0]
    n_elinks = conn.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
    print(f"\n  DB summary: {n_entities} entities ({n_lemmas} lemmas, {n_tactics} tactics)")
    print(
        f"  {n_anchors} anchors, {n_positions} positions, "
        f"{n_links} premise links, {n_elinks} entity links"
    )


def load_entities(input_path: Path, db_path: Path) -> None:
    """Load all entities from JSONL into the proof network database."""
    conn = init_db(db_path)
    anchor_cache: dict[str, int] = {}
    name_to_id: dict[str, int] = {}

    processed = _load_lemma_entities(conn, input_path, anchor_cache, name_to_id)
    print(f"  Loaded {processed} entities total")

    tactic_count = _build_tactic_entities(conn, input_path, anchor_cache, name_to_id)
    print(f"  Created {tactic_count} tactic entities")

    link_count = _build_tactic_links(conn, input_path, name_to_id)
    print(f"  Created {link_count} tactic-lemma links")

    print("  Building accessible_premises links...")
    premise_links = _build_premise_links(conn, input_path, name_to_id)
    print(f"  Inserted {premise_links} premise links")

    print("  Computing IDF values...")
    recompute_idf(conn)

    _print_db_summary(conn)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proof_network.db from entities JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/proof_network_entities.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/proof_network.db"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if args.output.exists():
        args.output.unlink()
        print(f"Removed existing {args.output}")

    print(f"Building {args.output} from {args.input}...")
    load_entities(args.input, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
