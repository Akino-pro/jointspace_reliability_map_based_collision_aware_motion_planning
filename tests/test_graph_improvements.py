"""Exact adjacency screening, optional torus seams and fair sampled terminals."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from scripts.planar3R_iris_regions import ReliablePolytope, polytope_adjacency
from scripts.periodic_graph import remap_edges, angle_error
from scripts.planar3R_ssm_envelopes import sample_hull
from scripts.sampled_goal_baseline import compare_sampled_goals

try:
    import pydrake
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def box(low, high):
    n = len(low)
    return ReliablePolytope(np.vstack((np.eye(n), -np.eye(n))), np.r_[high, -np.asarray(low)])


class AdjacencyTests(unittest.TestCase):
    def test_screening_matches_all_pairs_including_flat_contacts(self):
        regions = [box([i, 0, 0], [i+1, 1, 1]) for i in range(8)]
        stats = {}
        screened = polytope_adjacency(regions, statistics=stats)
        exhaustive = polytope_adjacency(regions, broad_phase=False)
        self.assertEqual(screened, exhaustive)
        self.assertEqual(stats["intersection_lp_count"], 7)
        self.assertEqual(stats["bounding_lp_count"], 0)
        self.assertEqual(len(screened), 14)

    def test_aabb_overlap_does_not_replace_exact_intersection(self):
        first = box([0, 0], [1, 1])
        first.A = np.vstack((first.A, [1, 1]))
        first.b = np.r_[first.b, .5]
        second = box([.4, .4], [1, 1])
        self.assertEqual(polytope_adjacency([first, second]), [])

    def test_non_axis_aligned_regions_use_bound_lps(self):
        diamond = ReliablePolytope(np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]]), np.ones(4))
        stats = {}
        self.assertEqual(polytope_adjacency([diamond, box([4, 4], [5, 5])], statistics=stats), [])
        self.assertEqual(stats["bounding_lp_count"], 4)

    def test_optional_multi_axis_seam_and_remap(self):
        regions = [box([2.9, 2.9, -.2], [np.pi, np.pi, .2]),
                   box([-np.pi, -np.pi, -.2], [-2.9, -2.9, .2])]
        self.assertEqual(polytope_adjacency(regions), [])
        edges = polytope_adjacency(regions, periodic=True)
        self.assertEqual(len(edges), 2)
        np.testing.assert_allclose(edges[0].shift, [2*np.pi, 2*np.pi, 0])
        mapped = remap_edges(edges, {0: 3, 1: 2})
        self.assertEqual(mapped[0], (3, 2))
        np.testing.assert_array_equal(mapped[0].shift, edges[0].shift)

    def test_aabb_overlap_across_seam_still_needs_exact_lp(self):
        first = box([2.9, 0], [np.pi, .2])
        first.A = np.vstack((first.A, [1, 0]))
        first.b = np.r_[first.b, 3.0]
        second = box([-np.pi, 0], [-2.9, .2])
        self.assertEqual(polytope_adjacency([first, second], periodic=True), [])


class BaselineTests(unittest.TestCase):
    def test_deduplicate_points_but_attach_all_containing_regions(self):
        regions = [box([-1, -1], [1, 1]), box([0, -1], [2, 1])]
        envelopes = [SimpleNamespace(samples=np.array([[.5, 0], [.5, 0], [4, 0], [.7, 0]]))]
        with patch("scripts.planar3R_two_stage_gcs_planner.solve_target_sets", return_value=None) as solve:
            report = compare_sampled_goals(regions, [], np.zeros(2), envelopes, [1, .4],
                dict(max_rounded_paths=200, max_rounding_trials=300),
                lambda q: q[0] != .7, lambda p: (True, len(p)), lambda q: 0)
        targets = solve.call_args.args[3]
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["regions"], [0, 1])
        self.assertEqual(report["unique_sample_count"], 3)
        self.assertEqual(report["terminal_edge_count"], 2)
        self.assertFalse(report["success"])
        self.assertEqual(solve.call_args.kwargs["max_rounding_trials"], 300)


@unittest.skipUnless(HAS_DRAKE, "Drake required")
class PeriodicSolveTests(unittest.TestCase):
    def solve_seam(self, n=3, axes=(0,)):
        from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
        lo, hi = np.full(n, -.2), np.full(n, .2)
        lo[list(axes)], hi[list(axes)] = 2.9, np.pi
        other_lo, other_hi = lo.copy(), hi.copy()
        other_lo[list(axes)], other_hi[list(axes)] = -np.pi, -2.9
        regions = [box(lo, hi), box(other_lo, other_hi)]
        start, goal = np.zeros(n), np.zeros(n)
        start[list(axes)], goal[list(axes)] = 3, -3
        target = [dict(polytope=sample_hull([goal]), region=1)]
        edges = polytope_adjacency(regions, periodic=True)
        solved = solve_target_sets(regions, edges, start, target, [1, 1],
                                   max_rounded_paths=10, max_rounding_trials=20, solver_backend="mosek")
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved["cost"], np.sqrt(len(axes))*(2*np.pi-6), places=5)
        self.assertLess(angle_error(solved["waypoints"][-1], goal), 1e-7)
        np.testing.assert_allclose(solved["waypoints"][0], start, atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(np.diff(solved["waypoints"], axis=0), axis=1).sum(),
                               solved["cost"], places=6)
        repeated = solve_target_sets(regions, edges, start, target, [1, 1],
            candidate_region_routes=[solved["route_spec"]], max_rounded_paths=2, max_rounding_trials=3,
            solver_backend="mosek")
        self.assertTrue(repeated["additional_route_results"][0]["success"])
        self.assertAlmostEqual(repeated["cost"], solved["cost"], places=6)
        return solved

    def test_single_seam_continuous_output_and_retained_route(self):
        self.solve_seam()

    def test_corner_seam(self):
        self.solve_seam(axes=(0, 1))

    def test_active_joint_slice_does_not_move_locked_joint(self):
        solved = self.solve_seam(n=2)
        full = np.full((len(solved["waypoints"]), 3), .7)
        full[:, [0, 2]] = solved["waypoints"]
        np.testing.assert_array_equal(full[:, 1], .7)

    def test_sampled_multi_goal_solution_is_physical_not_dummy_sink(self):
        region = box([-1, -1], [1, 1])
        report = compare_sampled_goals([region], [], np.zeros(2),
            [SimpleNamespace(samples=np.array([[.8, 0], [.2, 0]]))], [1],
            dict(max_rounded_paths=10, max_rounding_trials=20, solver_backend="mosek"),
            lambda q: True, lambda p: (True, len(p)), lambda q: 0)
        self.assertTrue(report["success"], report)
        self.assertAlmostEqual(report["cost"], .2, places=6)
        np.testing.assert_allclose(report["endpoint"], [.2, 0], atol=1e-7)

    def test_two_stage_pipeline_across_seam_projects_in_original_chart(self):
        from scripts.planar3R_two_stage_gcs_planner import plan_two_stage
        from scripts.planar3R_collision import link_points
        from scripts.planar3R_iris_regions import save_iris_checkpoint
        start, goal_q = np.array([3., .1, -.1]), np.array([-3., .1, -.1])
        regions = [box([2.9, -.2, -.2], [np.pi, .2, .2]),
                   box([-np.pi, -.2, -.2], [-2.9, .2, .2])]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/"seams.npz"
            save_iris_checkpoint(cache, regions, {"model": {
                "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
            with patch("scripts.planar3R_two_point_ssm_views.compute_ssms_for_target",
                       return_value=[np.tile(goal_q, (3, 1))]):
                result = plan_two_stage(start, link_points(goal_q)[-1], [], cache,
                    reliability_source="none", periodic=True, compare_sampled=True,
                    max_rounded_paths=10, max_rounding_trials=20,
                    stage2_max_rounded_paths=10, stage2_max_rounding_trials=20)
        self.assertTrue(result["success"], result)
        self.assertTrue(result["sampled_baseline"]["success"])
        self.assertAlmostEqual(result["cost"], .01*(2*np.pi-6), places=6)
        self.assertGreater(result["waypoints"][-1][0], np.pi)
        self.assertTrue(result["stage2_additional_route_results"][0]["success"])

    def test_post_failure_pipeline_crosses_seam_with_fixed_original_lock(self):
        from scripts.planar3R_post_failure_planner import plan_after_failure
        from scripts.planar3R_collision import link_points
        from scripts.planar3R_iris_regions import save_iris_checkpoint
        start, goal_q = np.array([3., .1, -.1]), np.array([-3., .1, -.1])
        regions = [box([2.9, -.2, -.2], [np.pi, .2, .2]),
                   box([-np.pi, -.2, -.2], [-2.9, .2, .2])]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/"seams.npz"
            save_iris_checkpoint(cache, regions, {"model": {
                "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
            result = plan_after_failure(start, 3, link_points(goal_q)[-1], cache, periodic=True)
        self.assertTrue(result["success"], result)
        np.testing.assert_array_equal(np.asarray(result["waypoints"])[:, 2], -.1)
        self.assertAlmostEqual(result["cost"], 2*np.pi-6, places=5)


if __name__ == "__main__":
    unittest.main()
