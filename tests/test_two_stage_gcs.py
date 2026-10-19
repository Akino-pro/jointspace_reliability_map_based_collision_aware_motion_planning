import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from planar3R_collision import link_points
from planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from planar3R_post_failure_planner import locked_joint_ik
from planar3R_ssm_envelopes import (
    build_envelopes, intersect_envelopes, project_to_local_ssm, sample_hull,
)
from planar3R_two_stage_gcs_planner import plan_two_stage, solve_target_sets

try:
    from pydrake.geometry.optimization import HPolyhedron
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def cube(lower=-np.pi, upper=np.pi):
    return ReliablePolytope(np.vstack((np.eye(3), -np.eye(3))),
                            np.r_[np.full(3, upper), np.full(3, -lower)])


def local_branch():
    reference = np.array([0.4, -0.8, 0.7])
    goal = link_points(reference)[-1]
    samples = []
    for angle in np.linspace(0.65, 0.75, 15):
        solutions, _ = locked_joint_ik(goal, 3, angle)
        samples.append(min(solutions, key=lambda q: np.linalg.norm(q - reference)))
    return goal, np.asarray(samples)


class EnvelopeTests(unittest.TestCase):
    def test_tighter_epsilon_refines_ordered_pieces_without_losing_samples(self):
        t = np.linspace(0, np.pi, 61)
        points = np.c_[np.cos(t), np.sin(t), t * 0]
        coarse = build_envelopes([points], 0.3)
        fine = build_envelopes([points], 0.02)
        self.assertGreater(len(fine), len(coarse))
        self.assertTrue(all(e.polyline_distance_bound <= 0.02 for e in fine))
        self.assertTrue(all(any(e.polytope.contains(q) for e in fine) for q in points))
        for first, second in zip(fine[:-1], fine[1:]):
            np.testing.assert_allclose(first.samples[-1], second.samples[0])

    def test_periodic_seam_is_split_before_subsampling(self):
        points = np.array([[3.0, 0, 0], [3.1, 0, 0], [-3.1, 0, 0], [-3.0, 0, 0]])
        envelopes = build_envelopes([points], 10, 2)
        self.assertEqual(len(envelopes), 2)
        self.assertTrue(all(not e.polytope.contains([0, 0, 0]) for e in envelopes))

    def test_exact_lower_dimensional_hulls_have_no_artificial_thickness(self):
        for samples, inside, outside in [
            ([[1, 2, 3]], [1, 2, 3], [1, 2, 3.01]),
            ([[0, 0, 0], [1, 0, 0]], [0.5, 0, 0], [0.5, 0.01, 0]),
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.2, 0.2, 0], [0.2, 0.2, 0.01]),
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]),
        ]:
            hull = sample_hull(samples)
            self.assertTrue(hull.contains(inside))
            self.assertFalse(hull.contains(outside))

    def test_intersection_keeps_line_contacts_and_rejects_empty_sets(self):
        envelopes = build_envelopes([np.array([[0, 0, 0], [1, 0, 0]])])
        targets = intersect_envelopes(envelopes, [cube(0, 1), cube(2, 3)])
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0]["polytope"].contains([0.5, 0, 0]))
        self.assertFalse(targets[0]["polytope"].contains([0.5, 0.1, 0]))

    def test_projection_is_continuous_ik_not_nearest_discrete_sample(self):
        goal, samples = local_branch()
        envelope = build_envelopes([samples], 10)[0]
        point = (samples[6] + samples[7]) / 2
        projected = project_to_local_ssm(point, envelope, cube(), goal, [], 0)
        self.assertIsNotNone(projected)
        self.assertLess(np.linalg.norm(link_points(projected)[-1] - goal), 1e-7)
        self.assertGreater(np.min(np.linalg.norm(samples - projected, axis=1)), 1e-4)

    def test_projection_cannot_escape_selected_free_region(self):
        goal, samples = local_branch()
        envelope = build_envelopes([samples], 10)[0]
        self.assertIsNone(project_to_local_ssm(samples[7], envelope, cube(2, 3), goal, [], 0))

    def test_invalid_resolution_is_rejected(self):
        for epsilon in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                build_envelopes([], epsilon)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache.npz"
        save_iris_checkpoint(self.cache, [cube()], {"model": {
            "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})

    def test_unreachable_goal_stops_without_sampling_or_gcs(self):
        with patch("planar3R_two_stage_gcs_planner.solve_target_sets") as solve:
            result = plan_two_stage([0, 0, 0], [10, 0], [], self.cache)
        self.assertEqual(result["status"], "goal_unreachable")
        self.assertEqual(result["gcs_solve_calls"], 0)
        solve.assert_not_called()

    def test_missing_start_coverage_stops_without_gcs(self):
        save_iris_checkpoint(self.cache, [cube(2, 3)], {"model": {
            "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
        result = plan_two_stage([0, 0, 0], [2, 0], [], self.cache, reliability_source="none")
        self.assertEqual(result["status"], "no_path_in_iris_coverage")
        self.assertEqual(result["gcs_solve_calls"], 0)

    def run_mocked(self, failure=None):
        goal, samples = local_branch()
        calls = []

        def solve(regions, edges, start, targets, weights, rounding):
            calls.append((regions, edges, start.copy(), targets))
            if len(calls) == 1:
                if failure == "stage1":
                    return None
                return {"waypoints": np.array([start, samples.mean(axis=0)]),
                        "target": 0, "cost": 1.0}
            goal_q = targets[0]["polytope"].b[:3]
            # Point hull SVD can choose an arbitrary orthogonal frame: recover
            # the singleton from its first three equality normals.
            goal_q = np.linalg.solve(targets[0]["polytope"].A[:3], goal_q)
            return {"waypoints": np.array([start, goal_q]), "target": 0,
                    "cost": float(np.linalg.norm(goal_q - start)), "solver_cost": 0.1}

        with patch("planar3R_two_point_ssm_views.compute_ssms_for_target", return_value=[samples]), \
                patch("planar3R_two_stage_gcs_planner.solve_target_sets", side_effect=solve):
            result = plan_two_stage([0.3, -0.9, 0.6], goal, [], self.cache,
                                    geometric_epsilon=10, reliability_source="none")
        return result, calls

    def test_two_solves_use_original_start_and_fresh_fixed_goal_graph(self):
        result, calls = self.run_mocked()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], 2)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(len(result["waypoints"]), 2)
        np.testing.assert_allclose(result["waypoints"][-1], result["estimated_goal"])
        self.assertLess(result["endpoint_error"], 1e-7)

    def test_solver_failure_stops_without_second_solve(self):
        result, calls = self.run_mocked("stage1")
        self.assertEqual(result["status"], "solver_failed")
        self.assertEqual(len(calls), 1)

    def test_projection_failure_stops_without_second_solve(self):
        with patch("planar3R_two_stage_gcs_planner.project_to_local_ssm", return_value=None):
            result, calls = self.run_mocked()
        self.assertEqual(result["status"], "projection_failed")
        self.assertEqual(result["gcs_solve_calls"], 1)
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(HAS_DRAKE, "Requires robotics Drake bindings")
class TwoStageDrakeTests(unittest.TestCase):
    def test_variable_target_cost_has_no_free_sink_motion(self):
        region = cube(-1, 3)
        target = sample_hull([[1, 0, 0], [2, 0, 0]])
        result = solve_target_sets([region], [], np.zeros(3),
                                   [{"polytope": target, "region": 0}], [0.4])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["waypoints"][-1], [1, 0, 0], atol=1e-5)
        self.assertAlmostEqual(result["cost"], 0.4, places=5)
        self.assertAlmostEqual(result["solver_cost"], 0.4, places=5)

    def test_full_two_stage_pipeline_with_real_solver_and_local_goal_curve(self):
        goal, samples = local_branch()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.npz"
            save_iris_checkpoint(cache, [cube()], {"model": {
                "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
            with patch("planar3R_two_point_ssm_views.compute_ssms_for_target", return_value=[samples]):
                result = plan_two_stage([0.3, -0.9, 0.6], goal, [], cache,
                                        reliability_source="none", geometric_epsilon=0.01)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], 2)
        self.assertLess(result["endpoint_error"], 1e-7)
        self.assertLess(result["fixed_goal_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
