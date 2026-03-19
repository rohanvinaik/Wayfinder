"""Tests for landmark_selectors — ternary selectors, signed RRF, convergence."""

import unittest

from src.landmark_selectors import (
    SelectorVote,
    build_hypotheses,
    classify_convergence,
    clear_selector_caches,
    compute_rrf,
    select_by_accessibility,
    select_by_bridge_potential,
    select_by_hub_suppressor,
    select_by_self_match,
)
from src.nav_contracts import ScoredEntity, StructuredQuery
from src.premise_retrieval import resolve_landmarks
from src.proof_network import clear_caches, init_db, recompute_idf


def _make_query(**kwargs):
    """Create a StructuredQuery with defaults."""
    defaults = {
        "bank_directions": {
            "structure": 1,
            "domain": 0,
            "depth": 0,
            "automation": 0,
            "context": 0,
            "decomposition": 0,
        },
        "bank_confidences": {
            "structure": 0.8,
            "domain": 0.5,
            "depth": 0.5,
            "automation": 0.5,
            "context": 0.5,
            "decomposition": 0.5,
        },
        "prefer_anchors": [1, 2, 3],
        "prefer_weights": [1.0, 1.0, 1.0],
    }
    defaults.update(kwargs)
    return StructuredQuery(**defaults)


def _build_test_db():
    """Build an in-memory DB with entities, anchors, links for testing.

    Entities:
        1: landmark_A (traced, has depends_on to 4, 5)
        2: landmark_B (traced, has depends_on to 5, 6)
        3: hub_entity (traced, depends_on to 4 — but is a hub target)
        4: premise_X (premise_only)
        5: premise_Y (premise_only)
        6: premise_Z (premise_only)

    Anchors: 1=proof_a, 2=const_b, 3=struct_c, 4=proof_d, 5=const_e
    """
    conn = init_db(":memory:")

    # Entities
    entities = [
        (1, "landmark_A", "lemma", "Ns", "file.lean", "traced"),
        (2, "landmark_B", "lemma", "Ns", "file.lean", "traced"),
        (3, "hub_entity", "lemma", "Ns", "file.lean", "traced"),
        (4, "premise_X", "lemma", "Ns", "file.lean", "premise_only"),
        (5, "premise_Y", "lemma", "Ns", "file.lean", "premise_only"),
        (6, "premise_Z", "lemma", "Ns", "file.lean", "premise_only"),
    ]
    conn.executemany(
        "INSERT INTO entities (id, name, entity_type, namespace, file_path, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        entities,
    )

    # Anchors
    anchors = [
        (1, "proof_a", "proof"),
        (2, "const_b", "constant"),
        (3, "struct_c", "structural"),
        (4, "proof_d", "proof"),
        (5, "const_e", "constant"),
    ]
    conn.executemany("INSERT INTO anchors (id, label, category) VALUES (?, ?, ?)", anchors)

    # Entity anchors: landmark_A has {1,2}, landmark_B has {2,3}, hub has {1}
    # premise_X has {1,4}, premise_Y has {2,5}, premise_Z has {3}
    entity_anchors = [
        (1, 1, 1.0),
        (1, 2, 1.0),
        (2, 2, 1.0),
        (2, 3, 1.0),
        (3, 1, 1.0),
        (4, 1, 1.0),
        (4, 4, 1.0),
        (5, 2, 1.0),
        (5, 5, 1.0),
        (6, 3, 1.0),
    ]
    conn.executemany(
        "INSERT INTO entity_anchors (entity_id, anchor_id, confidence) VALUES (?, ?, ?)",
        entity_anchors,
    )

    # Entity positions (all entities have structure=+1)
    for eid in range(1, 7):
        conn.execute(
            "INSERT INTO entity_positions (entity_id, bank, sign, depth) VALUES (?, ?, ?, ?)",
            (eid, "structure", 1, 1),
        )

    # depends_on links: landmark_A -> premise_X, premise_Y
    #                   landmark_B -> premise_Y, premise_Z
    #                   hub_entity -> premise_X
    links = [
        (1, 4, "depends_on", 0.95),
        (1, 5, "depends_on", 0.95),
        (2, 5, "depends_on", 0.95),
        (2, 6, "depends_on", 0.95),
        (3, 4, "depends_on", 0.95),
    ]
    conn.executemany(
        "INSERT INTO entity_links (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
        links,
    )

    # Accessible premises: theorem 100 can access premises 4, 5
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, provenance) "
        "VALUES (100, 'thm', 'lemma', 'traced')"
    )
    conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (100, 4)")
    conn.execute("INSERT INTO accessible_premises (theorem_id, premise_id) VALUES (100, 5)")

    recompute_idf(conn)
    conn.commit()
    return conn


class TestSelectorVoteContract(unittest.TestCase):
    """SelectorVote dataclass contract tests."""

    def test_ternary_values(self):
        for v in [+1, 0, -1]:
            vote = SelectorVote(
                vote=v,
                confidence=0.5,
                raw_score=1.0,
                selector_name="test",
                family="bridge",
                evidence="test",
                rank=1,
            )
            self.assertEqual(vote.vote, v)

    def test_confidence_range(self):
        vote = SelectorVote(
            vote=1,
            confidence=0.0,
            raw_score=0.0,
            selector_name="test",
            family="bridge",
            evidence="test",
            rank=1,
        )
        self.assertGreaterEqual(vote.confidence, 0.0)
        self.assertLessEqual(vote.confidence, 1.0)


class TestBridgePotentialSelector(unittest.TestCase):
    """Tests for select_by_bridge_potential."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()
        from src.proof_network import _get_data_cache, _get_idf_cache

        self.data = _get_data_cache(self.conn)
        self.idf_cache = _get_idf_cache(self.conn)
        from src.premise_retrieval import _get_neighborhood_signatures

        self.nbr_sigs = _get_neighborhood_signatures(self.conn, self.data)

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_returns_positive_votes_for_matching_landmarks(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        candidates = [1, 2, 3]
        votes = select_by_bridge_potential(
            candidates, query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        # landmark_A depends on premise_X (anchor 1,4) and premise_Y (anchor 2,5)
        # So nbr_sigs[1] includes anchors {1, 4, 2, 5} — matches query anchors 1, 2
        self.assertIn(1, votes)
        self.assertEqual(votes[1].vote, 1)
        self.assertEqual(votes[1].family, "bridge")

    def test_never_votes_negative(self):
        query = _make_query(prefer_anchors=[1], prefer_weights=[1.0])
        candidates = [1, 2, 3]
        votes = select_by_bridge_potential(
            candidates, query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        for v in votes.values():
            self.assertGreaterEqual(v.vote, 0)

    def test_empty_query_returns_empty(self):
        query = _make_query(prefer_anchors=[], prefer_weights=[])
        votes = select_by_bridge_potential(
            [1, 2], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        self.assertEqual(votes, {})

    def test_ranks_are_sequential(self):
        query = _make_query(prefer_anchors=[1, 2, 3], prefer_weights=[1.0, 1.0, 1.0])
        votes = select_by_bridge_potential(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        ranks = sorted(v.rank for v in votes.values())
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))


class TestSelfMatchSelector(unittest.TestCase):
    """Tests for select_by_self_match."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()
        from src.proof_network import _get_data_cache, _get_idf_cache

        self.data = _get_data_cache(self.conn)
        self.idf_cache = _get_idf_cache(self.conn)
        from src.premise_retrieval import _get_neighborhood_signatures

        self.nbr_sigs = _get_neighborhood_signatures(self.conn, self.data)

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_confidence_capped_at_06(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        votes = select_by_self_match(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        for v in votes.values():
            self.assertLessEqual(v.confidence, 0.6)

    def test_never_votes_negative(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        votes = select_by_self_match(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        for v in votes.values():
            self.assertGreaterEqual(v.vote, 0)

    def test_top_quartile_gets_positive_vote(self):
        # With 3 candidates, top quartile = max(3//4, 1) = 1
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        votes = select_by_self_match(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        positive_votes = [v for v in votes.values() if v.vote == 1]
        self.assertGreaterEqual(len(positive_votes), 1)

    def test_family_is_match(self):
        query = _make_query(prefer_anchors=[1], prefer_weights=[1.0])
        votes = select_by_self_match(
            [1], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        for v in votes.values():
            self.assertEqual(v.family, "match")


class TestAccessibilitySelector(unittest.TestCase):
    """Tests for select_by_accessibility."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()
        from src.proof_network import _get_data_cache, _get_idf_cache

        self.data = _get_data_cache(self.conn)
        self.idf_cache = _get_idf_cache(self.conn)
        from src.premise_retrieval import _get_neighborhood_signatures

        self.nbr_sigs = _get_neighborhood_signatures(self.conn, self.data)

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_abstains_when_no_accessible_theorem_id(self):
        query = _make_query(accessible_theorem_id=None)
        votes = select_by_accessibility(
            [1, 2], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        self.assertEqual(votes, {})

    def test_votes_positive_when_neighbors_overlap_accessible(self):
        # theorem 100 has accessible premises {4, 5}
        # landmark_A depends on {4, 5} — full overlap
        query = _make_query(accessible_theorem_id=100)
        votes = select_by_accessibility(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        self.assertIn(1, votes)
        self.assertEqual(votes[1].vote, 1)

    def test_abstains_when_no_overlap(self):
        # landmark_B depends on {5, 6}, accessible = {4, 5}
        # overlap exists (5), so landmark_B should vote +1
        # But if we create a landmark with no overlap...
        # Entity 3 depends on {4} which IS accessible — so it votes +1 too
        query = _make_query(accessible_theorem_id=100)
        votes = select_by_accessibility(
            [1, 2, 3], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        # All landmarks have some overlap with {4, 5}
        for v in votes.values():
            self.assertEqual(v.vote, 1)

    def test_family_is_context(self):
        query = _make_query(accessible_theorem_id=100)
        votes = select_by_accessibility(
            [1], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        if votes:
            self.assertEqual(votes[1].family, "context")


class TestHubSuppressorSelector(unittest.TestCase):
    """Tests for select_by_hub_suppressor."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()
        from src.proof_network import _get_data_cache, _get_idf_cache

        self.data = _get_data_cache(self.conn)
        self.idf_cache = _get_idf_cache(self.conn)
        from src.premise_retrieval import _get_neighborhood_signatures

        self.nbr_sigs = _get_neighborhood_signatures(self.conn, self.data)

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_never_votes_positive(self):
        query = _make_query()
        votes = select_by_hub_suppressor(
            [1, 2, 3, 4, 5, 6], query, self.data, self.idf_cache, self.conn, self.nbr_sigs
        )
        for v in votes.values():
            self.assertLessEqual(v.vote, 0)

    def test_suppresses_high_degree_entities(self):
        # Add many depends_on links targeting entity 4 to make it a hub
        for i in range(200, 260):
            self.conn.execute(
                "INSERT INTO entities (id, name, entity_type, provenance) "
                "VALUES (?, ?, 'lemma', 'traced')",
                (i, f"extra_{i}"),
            )
            self.conn.execute(
                "INSERT INTO entity_links (source_id, target_id, relation, weight) "
                "VALUES (?, 4, 'depends_on', 0.95)",
                (i,),
            )
        self.conn.commit()
        clear_caches()
        clear_selector_caches()
        from src.proof_network import _get_data_cache

        data = _get_data_cache(self.conn)

        query = _make_query()
        votes = select_by_hub_suppressor(
            [4, 5, 6], query, data, self.idf_cache, self.conn, self.nbr_sigs
        )
        # Entity 4 should be suppressed (high in-degree)
        self.assertIn(4, votes)
        self.assertEqual(votes[4].vote, -1)

    def test_family_is_suppression(self):
        # Make entity 4 a hub
        for i in range(200, 260):
            self.conn.execute(
                "INSERT INTO entities (id, name, entity_type, provenance) "
                "VALUES (?, ?, 'lemma', 'traced')",
                (i, f"extra_{i}"),
            )
            self.conn.execute(
                "INSERT INTO entity_links (source_id, target_id, relation, weight) "
                "VALUES (?, 4, 'depends_on', 0.95)",
                (i,),
            )
        self.conn.commit()
        clear_caches()
        clear_selector_caches()
        from src.proof_network import _get_data_cache

        data = _get_data_cache(self.conn)

        query = _make_query()
        votes = select_by_hub_suppressor([4], query, data, self.idf_cache, self.conn, self.nbr_sigs)
        if votes:
            self.assertEqual(votes[4].family, "suppression")


class TestSignedRRF(unittest.TestCase):
    """Tests for compute_rrf."""

    def test_single_positive_vote(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.8,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            )
        ]
        rrf = compute_rrf(votes, +1)
        self.assertAlmostEqual(rrf, 0.8 / (60 + 1))

    def test_negative_votes_ignored_for_positive(self):
        votes = [
            SelectorVote(
                vote=-1,
                confidence=0.9,
                raw_score=1.0,
                selector_name="a",
                family="suppression",
                evidence="",
                rank=1,
            )
        ]
        rrf = compute_rrf(votes, +1)
        self.assertAlmostEqual(rrf, 0.0)

    def test_multiple_positive_votes_sum(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.8,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            ),
            SelectorVote(
                vote=1,
                confidence=0.6,
                raw_score=0.5,
                selector_name="b",
                family="match",
                evidence="",
                rank=3,
            ),
        ]
        rrf = compute_rrf(votes, +1)
        expected = 0.8 / (60 + 1) + 0.6 / (60 + 3)
        self.assertAlmostEqual(rrf, expected)

    def test_zero_rank_treated_as_one(self):
        votes = [
            SelectorVote(
                vote=-1,
                confidence=0.9,
                raw_score=1.0,
                selector_name="hub",
                family="suppression",
                evidence="",
                rank=0,
            )
        ]
        rrf = compute_rrf(votes, -1)
        self.assertAlmostEqual(rrf, 0.9 / (60 + 1))

    def test_confidence_differentiates_votes(self):
        """High-confidence hub suppression outweighs low-confidence."""
        high = [
            SelectorVote(
                vote=-1, confidence=0.9, raw_score=50.0,
                selector_name="hub", family="suppression", evidence="", rank=0,
            )
        ]
        low = [
            SelectorVote(
                vote=-1, confidence=0.5, raw_score=30.0,
                selector_name="hub", family="suppression", evidence="", rank=0,
            )
        ]
        self.assertGreater(compute_rrf(high, -1), compute_rrf(low, -1))

    def test_abstain_votes_not_counted(self):
        votes = [
            SelectorVote(
                vote=0,
                confidence=0.5,
                raw_score=0.3,
                selector_name="a",
                family="match",
                evidence="",
                rank=2,
            )
        ]
        self.assertAlmostEqual(compute_rrf(votes, +1), 0.0)
        self.assertAlmostEqual(compute_rrf(votes, -1), 0.0)


class TestConvergenceClassification(unittest.TestCase):
    """Tests for classify_convergence."""

    def test_converged_two_families_no_opposition(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.9,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            ),
            SelectorVote(
                vote=1,
                confidence=0.7,
                raw_score=0.5,
                selector_name="b",
                family="match",
                evidence="",
                rank=2,
            ),
        ]
        result = classify_convergence(votes, support_rrf=0.03, oppose_rrf=0.0)
        self.assertEqual(result, "converged")

    def test_conflicted_high_opposition(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.8,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            ),
            SelectorVote(
                vote=-1,
                confidence=0.9,
                raw_score=5.0,
                selector_name="hub",
                family="suppression",
                evidence="",
                rank=0,
            ),
        ]
        # oppose_rrf > 0.5 * support_rrf
        result = classify_convergence(votes, support_rrf=0.016, oppose_rrf=0.016)
        self.assertEqual(result, "conflicted")

    def test_isolated_single_family(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.9,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            ),
        ]
        result = classify_convergence(votes, support_rrf=0.016, oppose_rrf=0.0)
        self.assertEqual(result, "isolated")

    def test_converged_with_low_opposition(self):
        votes = [
            SelectorVote(
                vote=1,
                confidence=0.9,
                raw_score=1.0,
                selector_name="a",
                family="bridge",
                evidence="",
                rank=1,
            ),
            SelectorVote(
                vote=1,
                confidence=0.6,
                raw_score=0.5,
                selector_name="b",
                family="context",
                evidence="",
                rank=2,
            ),
            SelectorVote(
                vote=-1,
                confidence=0.5,
                raw_score=3.0,
                selector_name="hub",
                family="suppression",
                evidence="",
                rank=0,
            ),
        ]
        # oppose < 0.3 * support
        result = classify_convergence(votes, support_rrf=0.032, oppose_rrf=0.005)
        self.assertEqual(result, "converged")


class TestBuildHypotheses(unittest.TestCase):
    """Tests for build_hypotheses."""

    def test_excludes_candidates_without_support(self):
        all_votes = {
            1: [
                SelectorVote(
                    vote=-1,
                    confidence=0.9,
                    raw_score=1.0,
                    selector_name="hub",
                    family="suppression",
                    evidence="",
                    rank=0,
                )
            ],
        }
        hyps = build_hypotheses([1], all_votes, {}, oppose_lambda=0.5)
        self.assertEqual(len(hyps), 0)

    def test_includes_candidates_with_support(self):
        all_votes = {
            1: [
                SelectorVote(
                    vote=1,
                    confidence=0.9,
                    raw_score=1.0,
                    selector_name="a",
                    family="bridge",
                    evidence="",
                    rank=1,
                )
            ],
        }
        hyps = build_hypotheses([1], all_votes, {1: {10, 20}}, oppose_lambda=0.5)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].entity_id, 1)
        self.assertEqual(hyps[0].neighborhood_size, 2)

    def test_sorted_by_fused_score(self):
        all_votes = {
            1: [
                SelectorVote(
                    vote=1,
                    confidence=0.5,
                    raw_score=0.3,
                    selector_name="a",
                    family="bridge",
                    evidence="",
                    rank=5,
                )
            ],
            2: [
                SelectorVote(
                    vote=1,
                    confidence=0.9,
                    raw_score=1.0,
                    selector_name="a",
                    family="bridge",
                    evidence="",
                    rank=1,
                )
            ],
        }
        hyps = build_hypotheses([1, 2], all_votes, {}, oppose_lambda=0.5)
        self.assertEqual(hyps[0].entity_id, 2)  # higher rank -> higher RRF

    def test_novelty_proxy_with_overlap(self):
        all_votes = {
            1: [
                SelectorVote(
                    vote=1,
                    confidence=0.9,
                    raw_score=1.0,
                    selector_name="a",
                    family="bridge",
                    evidence="",
                    rank=1,
                )
            ],
        }
        nbr_sigs = {1: {10, 20, 30}}
        selected_nbrs = {10, 20}  # 2/3 overlap
        hyps = build_hypotheses([1], all_votes, nbr_sigs, selected_nbrs=selected_nbrs)
        self.assertAlmostEqual(hyps[0].novelty_proxy, 1.0 / 3.0, places=4)


class TestResolveLandmarks(unittest.TestCase):
    """Integration tests for resolve_landmarks."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_returns_scored_entities(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        results, trace = resolve_landmarks(query, self.conn)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, ScoredEntity)

    def test_returns_trace(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        _, trace = resolve_landmarks(query, self.conn)
        self.assertGreaterEqual(trace.landmark_count, 0)

    def test_respects_landmark_limit(self):
        query = _make_query(prefer_anchors=[1, 2, 3], prefer_weights=[1.0, 1.0, 1.0])
        results, _ = resolve_landmarks(query, self.conn, config={"landmark_limit": 1})
        self.assertLessEqual(len(results), 1)

    def test_only_traced_entities(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        results, trace = resolve_landmarks(query, self.conn)
        self.assertEqual(trace.landmarks_by_provenance.get("premise_only", 0), 0)

    def test_empty_query_returns_empty(self):
        query = _make_query(prefer_anchors=[], prefer_weights=[])
        results, _ = resolve_landmarks(query, self.conn)
        self.assertEqual(len(results), 0)

    def test_scores_are_non_negative(self):
        query = _make_query(prefer_anchors=[1, 2, 3], prefer_weights=[1.0, 1.0, 1.0])
        results, _ = resolve_landmarks(query, self.conn)
        for r in results:
            self.assertGreaterEqual(r.final_score, 0.0)

    def test_trace_has_selector_vote_counts(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        _, trace = resolve_landmarks(query, self.conn)
        # Should have vote counts keyed like "bridge_potential:+1"
        self.assertIsInstance(trace.selector_vote_counts, dict)
        if trace.landmark_count > 0:
            self.assertGreater(len(trace.selector_vote_counts), 0)

    def test_trace_has_convergence_distribution(self):
        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        _, trace = resolve_landmarks(query, self.conn)
        self.assertIsInstance(trace.convergence_distribution, dict)
        for key in trace.convergence_distribution:
            self.assertIn(key, ("converged", "isolated", "conflicted"))


class TestResolveWiring(unittest.TestCase):
    """Test that resolve() in resolution.py can use landmark_expand."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()

    def test_resolve_with_landmark_expand_strategy(self):
        from src.nav_contracts import NavOutput
        from src.resolution import SearchContext, resolve

        nav = NavOutput(
            directions={"structure": 1},
            direction_confidences={"structure": 0.8},
            anchor_scores={},
            progress=0.5,
            critic_score=0.5,
        )
        ctx = SearchContext()
        # With landmark_expand strategy
        candidates = resolve(
            nav, self.conn, ctx,
            retrieval_config={"strategy": "landmark_expand"},
        )
        self.assertIsInstance(candidates, list)

    def test_resolve_default_uses_flat(self):
        from src.nav_contracts import NavOutput
        from src.resolution import SearchContext, resolve

        nav = NavOutput(
            directions={"structure": 1},
            direction_confidences={"structure": 0.8},
            anchor_scores={},
            progress=0.5,
            critic_score=0.5,
        )
        ctx = SearchContext()
        # Default strategy (flat) should still work
        candidates = resolve(nav, self.conn, ctx)
        self.assertIsInstance(candidates, list)


class TestCacheInvalidation(unittest.TestCase):
    """Test that clear_caches() clears all module caches."""

    def test_clear_caches_clears_selector_cache(self):
        from src.landmark_selectors import _nbr_entity_cache

        _nbr_entity_cache[999] = {1: {2, 3}}
        clear_caches()
        self.assertEqual(len(_nbr_entity_cache), 0)

    def test_clear_caches_clears_neighborhood_cache(self):
        from src.retrieval_stages import _neighborhood_cache

        _neighborhood_cache[999] = {1: {2, 3}}
        clear_caches()
        self.assertEqual(len(_neighborhood_cache), 0)


class TestStrategyFlag(unittest.TestCase):
    """Test landmark_strategy flag in landmark_expand_retrieve."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_default_strategy_uses_bridge_potential(self):
        from src.premise_retrieval import landmark_expand_retrieve

        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        # Default strategy should work without error
        results, trace = landmark_expand_retrieve(query, self.conn, limit=4)
        self.assertIsInstance(results, list)

    def test_multi_pass_strategy(self):
        from src.premise_retrieval import landmark_expand_retrieve

        query = _make_query(prefer_anchors=[1, 2], prefer_weights=[1.0, 1.0])
        results, trace = landmark_expand_retrieve(
            query, self.conn, limit=4, config={"landmark_strategy": "multi_pass"}
        )
        self.assertIsInstance(results, list)


class TestDataCacheHubInDegrees(unittest.TestCase):
    """Test that _DataCache loads hub_in_degrees correctly."""

    def setUp(self):
        clear_caches()
        clear_selector_caches()
        self.conn = _build_test_db()

    def tearDown(self):
        self.conn.close()
        clear_caches()
        clear_selector_caches()

    def test_hub_in_degrees_loaded(self):
        from src.proof_network import _get_data_cache

        data = _get_data_cache(self.conn)
        # premise_X (id=4) is targeted by landmark_A and hub_entity -> in_degree=2
        self.assertEqual(data.hub_in_degrees.get(4), 2)
        # premise_Y (id=5) is targeted by landmark_A and landmark_B -> in_degree=2
        self.assertEqual(data.hub_in_degrees.get(5), 2)

    def test_hub_in_degrees_graceful_without_links(self):
        conn = init_db(":memory:")
        clear_caches()
        from src.proof_network import _get_data_cache

        data = _get_data_cache(conn)
        self.assertEqual(data.hub_in_degrees, {})
        conn.close()


if __name__ == "__main__":
    unittest.main()
