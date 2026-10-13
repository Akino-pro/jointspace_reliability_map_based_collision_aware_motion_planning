import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from planar3R_collision import link_points
from planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from planar3R_post_failure_planner import locked_joint_ik, plan_after_failure, slice_regions

try:
    from pydrake.geometry.optimization import Point
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def cube(lower=None, upper=None):
    lower = np.full(3, -np.pi) if lower is None else np.asarray(lower)
    upper = np.full(3, np.pi) if upper is None else np.asarray(upper)
    return ReliablePolytope(np.vstack((np.eye(3), -np.eye(3))), np.concatenate((upper, -lower)))


class PostFailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "regions.npz"
        self.save([cube()])

    def save(self, regions, obstacles=None):
        save_iris_checkpoint(self.cache, regions, {"model": {
            "link_lengths": [1, 1, 1], "link_radius": 0.0,
            "obstacles": [] if obstacles is None else obstacles,
        }})

    def test_analytic_ik_for_each_failed_joint_matches_original_3r(self):
        reference = np.array([0.4, -0.8, 0.7])
        goal = link_points(reference)[-1]
        for joint in (1, 2, 3):
            with self.subTest(joint=joint):
                solutions, degenerate = locked_joint_ik(goal, joint, reference[joint - 1])
                self.assertFalse(degenerate)
                self.assertEqual(len(solutions), 2)
                self.assertTrue(any(np.linalg.norm(q - reference) < 1e-8 for q in solutions))
                for q in solutions:
                    self.assertEqual(q[joint - 1], reference[joint - 1])
                    np.testing.assert_allclose(link_points(q)[-1], goal, atol=1e-8)

    def test_singular_ik_solutions_merge_instead_of_duplicate(self):
        solutions, degenerate = locked_joint_ik([3, 0], 3, 0)
        self.assertFalse(degenerate)
        self.assertEqual(len(solutions), 1)

    def test_slice_preserves_original_slanted_constraints(self):
        region = cube()
        region.A = np.vstack((region.A, [1, 1, 0]))
        region.b = np.append(region.b, 0.25)
        slices = slice_regions([region], 1, 0.5)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].A.shape[1], 2)
        self.assertTrue(slices[0].contains([-0.4, 0]))
        self.assertFalse(slices[0].contains([0, 0]))
        self.assertEqual(len(slice_regions([cube()], 1, np.pi)), 1)
        self.assertEqual(slice_regions([cube()], 1, 4), [])

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_unreachable_stops_before_gcs(self, solve):
        report = plan_after_failure([0.4, -0.8, 0.7], 2, [10, 10], self.cache)
        self.assertEqual(report["status"], "unreachable")
        self.assertEqual(report["gcs_solve_calls"], 0)
        solve.assert_not_called()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_collision_at_failure_configuration_stops_before_gcs(self, solve):
        self.save([cube()], [{"type": "circle", "center": [0.5, 0], "radius": 0.1}])
        report = plan_after_failure([0, 0, 0], 3, [1, 1], self.cache)
        self.assertEqual(report["status"], "start_in_collision")
        solve.assert_not_called()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_all_colliding_goals_stop_before_gcs(self, solve):
        goal = link_points([0.3, 0.6, 0])[-1]
        self.save([cube()], [{"type": "circle", "center": goal.tolist(), "radius": 0.02}])
        report = plan_after_failure([np.pi / 2, 0, 0], 3, goal, self.cache)
        self.assertEqual(report["status"], "all_goals_in_collision")
        solve.assert_not_called()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_missing_coverage_stops_without_generating_or_solving(self, solve):
        q = np.array([0.4, -0.8, 0.7])
        self.save([cube(q - 0.02, q + 0.02)])
        report = plan_after_failure(q, 3, link_points([0.9, -0.5, 0.7])[-1], self.cache)
        self.assertEqual(report["status"], "no_path_in_iris_coverage")
        self.assertEqual(report["gcs_solve_calls"], 0)
        solve.assert_not_called()

    @patch("planar3R_post_failure_planner.solve_to_any_goal", return_value=(None, None))
    def test_solver_failure_is_not_retried(self, solve):
        report = plan_after_failure([0.4, -0.8, 0.7], 3,
                                    link_points([0.9, -0.5, 0.7])[-1], self.cache)
        self.assertEqual(report["status"], "solver_failed")
        self.assertEqual(report["gcs_solve_calls"], 1)
        solve.assert_called_once()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_invalid_returned_path_stops_without_retry(self, solve):
        solve.return_value = (np.array([[0.4, -0.8], [0.5, -0.8]]), 0.1)
        report = plan_after_failure([0.4, -0.8, 0.7], 3,
                                    link_points([0.9, -0.5, 0.7])[-1], self.cache)
        self.assertEqual(report["status"], "path_validation_failed")
        solve.assert_called_once()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_degenerate_continuous_ik_is_not_misreported_as_unreachable(self, solve):
        report = plan_after_failure([0.3, 0.4, np.pi], 3, [1, 0], self.cache)
        self.assertEqual(report["status"], "unsupported_degenerate_ik")
        solve.assert_not_called()

    @patch("planar3R_post_failure_planner.solve_to_any_goal")
    def test_already_at_goal_needs_no_solver(self, solve):
        q = [0.4, -0.8, 0.7]
        report = plan_after_failure(q, 3, link_points(q)[-1], self.cache)
        self.assertTrue(report["success"])
        self.assertEqual(report["path_length"], 0)
        solve.assert_not_called()

    def test_more_than_one_failure_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_after_failure([0, 0, 0], [1, 2], [1, 1], self.cache)

    @unittest.skipUnless(HAS_DRAKE, "Requires robotics Drake bindings")
    def test_one_gcs_solve_selects_nearest_goal_with_unit_length_cost(self):
        start = np.array([0.4, -0.8, 0.7])
        for joint in (1, 2, 3):
            with self.subTest(joint=joint):
                active = [i for i in range(3) if i != joint - 1]
                desired = start.copy()
                desired[active] += [0.1, 0.05]
                goal = link_points(desired)[-1]
                solutions, _ = locked_joint_ik(goal, joint, start[joint - 1])
                expected = min(np.linalg.norm(q[active] - start[active]) for q in solutions)
                report = plan_after_failure(start, joint, goal, self.cache)
                self.assertTrue(report["success"], report)
                self.assertEqual(report["gcs_solve_calls"], 1)
                self.assertEqual(report["connected_goal_count"], 2)
                path = np.asarray(report["waypoints"])
                np.testing.assert_array_equal(path[:, joint - 1], np.full(len(path), start[joint - 1]))
                self.assertAlmostEqual(report["cost"], expected, places=5)
                self.assertAlmostEqual(report["path_length"], expected, places=5)


if __name__ == "__main__":
    unittest.main()
