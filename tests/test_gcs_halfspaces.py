"""Verify compaction preserves sets, necessary overlap rows and solved path costs."""
import unittest
import numpy as np
from gcs.halfspaces import compact_halfspaces, nonimplied_rows
from scripts.planar3R_iris_regions import ReliablePolytope
from scripts.planar3R_ssm_envelopes import sample_hull

try:
    from pydrake.geometry.optimization import GraphOfConvexSets
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


class HalfspaceTests(unittest.TestCase):
    def test_scaled_axis_duplicates_keep_tightest_bound(self):
        A = np.array([[1., 0], [2, 0], [-3, 0], [0, 1], [0, -1], [1, 1], [1, 1]])
        b = np.array([2., 2, 0, 1, 0, 2, 1.5])
        small_A, small_b = compact_halfspaces(A, b)
        self.assertEqual(len(small_b), 5)
        for q in np.random.default_rng(7).uniform(-1, 3, (1000, 2)):
            self.assertEqual(bool(np.all(A@q <= b)), bool(np.all(small_A@q <= small_b)))

    def test_adjacent_box_keeps_shared_face_bound(self):
        A = np.vstack((np.eye(2), -np.eye(2)))
        source_b, target_b = np.array([1., 1, 0, 0]), np.array([2., 1, -1, 0])
        self.assertEqual(nonimplied_rows(A, source_b, A, target_b), [0])

    @unittest.skipUnless(HAS_DRAKE, "Drake/MOSEK is required")
    def test_weighted_gcs_cost_and_fixed_endpoint_are_preserved(self):
        from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
        A = np.vstack((np.eye(2), -np.eye(2), [2., 0]))
        regions = [ReliablePolytope(A.copy(), np.array(b)) for b in
                   ([1., 1, 0, 0, 2], [2., 1, -1, 0, 4], [2., 2, -1, -1, 4])]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
        start, goal = np.array([.1, .1]), np.array([1.9, 1.9])
        results = [solve_target_sets(regions, edges, start,
                   [dict(polytope=sample_hull([goal]), region=2)], [.2, .9, .4],
                   max_rounded_paths=5, max_rounding_trials=20, solver_backend="mosek",
                   candidate_region_routes=[[0, 1, 2]], compact_constraints=compact)
                   for compact in (False, True)]
        self.assertTrue(all(r is not None for r in results))
        self.assertAlmostEqual(results[0]["cost"], results[1]["cost"], places=6)
        for result in results:
            np.testing.assert_allclose(result["waypoints"][-1], goal, atol=1e-8)
