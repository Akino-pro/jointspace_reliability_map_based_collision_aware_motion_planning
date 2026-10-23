import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import spatial4R_model as model
from scripts.spatial4R_iris import scene_metadata
from scripts.planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from scripts.spatial4R_post_failure_planner import chart_configuration, locked_joint_ik, plan_after_failure, slice_regions

try:
    from pydrake.geometry.optimization import Point
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def cube(center, radius):
    center = np.asarray(center)
    return ReliablePolytope(np.vstack((np.eye(4), -np.eye(4))), np.r_[center+radius, -center+radius])


def root_report(singular=0):
    return dict(proven_unreachable=False, singular_solution_count=singular)


class SpatialFailureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.checkpoint = Path(self.folder.name)/"regions.npz"
        self.goal_q = np.array([.8, -.4, .2, .4])
        self.goal = model.position(self.goal_q)
        save_iris_checkpoint(self.checkpoint, [cube(np.zeros(4), np.pi)],
                             dict(model=scene_metadata([]), generation_seconds=0))

    def failure_q(self, joint):
        q = self.goal_q+np.array([.02, -.03, .01, -.02])
        q[joint-1] = self.goal_q[joint-1]
        return q

    def test_ik_returns_fk_valid_locked_roots_for_all_four_joints(self):
        for joint in range(1, 5):
            failure = self.failure_q(joint)
            roots, report = locked_joint_ik(self.goal, joint, failure[joint-1], failure, max_starts=32)
            self.assertGreater(len(roots), 0)
            self.assertEqual(report["starts_attempted"], 32)
            self.assertFalse(report["completeness_certified"])
            self.assertEqual(report["singular_solution_count"], 0)
            self.assertTrue(any(np.linalg.norm(model.wrap(q-self.goal_q)) < 1e-6 for q in roots))
            for q in roots:
                self.assertEqual(q[joint-1], failure[joint-1])
                self.assertLess(np.linalg.norm(model.position(q)-self.goal), 1e-8)

    def test_ik_stops_at_four_periodically_distinct_physical_configurations(self):
        roots = [np.array([-.9+.5*i, 0., 0.]) for i in range(4)]
        # Controlled IK fixture checks enumeration, not physical root existence.
        with patch('scripts.spatial4R_post_failure_planner.least_squares',
                   side_effect=[SimpleNamespace(x=u) for u in roots]) as solve, \
                patch.object(model, "position", return_value=np.zeros(3)), \
                patch.object(model, "jacobian", return_value=np.column_stack((np.eye(3), np.zeros(3)))):
            solutions, report = locked_joint_ik(np.zeros(3), 4, .2, max_starts=20)
        self.assertEqual(len(solutions), 4)
        self.assertEqual(solve.call_count, 4)
        self.assertEqual(report["physical_solution_limit"], 4)
        self.assertEqual(report["physical_solution_count"], 4)
        self.assertEqual(report["termination_reason"], "physical_solution_limit_reached")

    def test_slice_substitution_matches_original_halfspaces_for_each_joint(self):
        parent = cube(np.zeros(4), 2)
        parent.A = np.vstack((parent.A, [1, 2, -1, .5]))
        parent.b = np.r_[parent.b, .8]
        for joint in range(1, 5):
            active = [i for i in range(4) if i != joint-1]
            slices = slice_regions([parent], joint, .3)
            self.assertEqual(len(slices), 1)
            self.assertEqual(slices[0].A.shape[1], 3)
            for u in np.random.default_rng(joint).uniform(-3, 3, (100, 3)):
                q = np.zeros(4)
                q[joint-1] = .3
                q[active] = u
                self.assertEqual(parent.contains(q), slices[0].contains(u))
            self.assertEqual(slice_regions([parent], joint, 3), [])

    def test_conservative_unreachable_goal_stops_without_loading_regions(self):
        with patch('scripts.spatial4R_post_failure_planner.load_iris_checkpoint') as load:
            report = plan_after_failure(self.failure_q(4), 4, [100, 0, 0], "missing.npz")
        self.assertEqual(report["status"], "unreachable")
        self.assertEqual(report["gcs_solve_calls"], 0)
        load.assert_not_called()

    def test_missing_numerical_ik_is_inconclusive_and_stops_without_gcs(self):
        with patch('scripts.spatial4R_post_failure_planner.locked_joint_ik', return_value=([], root_report())), \
                patch('scripts.spatial4R_post_failure_planner.solve_target_sets') as solve:
            report = plan_after_failure(self.failure_q(4), 4, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "ik_search_failed")
        self.assertEqual(report["gcs_solve_calls"], 0)
        solve.assert_not_called()

    def test_singular_goal_is_reported_without_assuming_a_continuous_set(self):
        with patch('scripts.spatial4R_post_failure_planner.locked_joint_ik', return_value=([self.goal_q], root_report(1))), \
                patch('scripts.spatial4R_post_failure_planner.solve_target_sets') as solve:
            report = plan_after_failure(self.failure_q(4), 4, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "unsupported_singular_ik")
        solve.assert_not_called()

    def test_colliding_found_goals_stop_before_gcs(self):
        failure = self.failure_q(4)
        with patch('scripts.spatial4R_post_failure_planner.locked_joint_ik', return_value=([self.goal_q], root_report())), \
                patch.object(model, "collision_free", side_effect=lambda q, *args: np.linalg.norm(q-failure) < 1e-10), \
                patch('scripts.spatial4R_post_failure_planner.solve_target_sets') as solve:
            report = plan_after_failure(failure, 4, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "all_found_goals_in_collision")
        self.assertEqual(report["gcs_solve_calls"], 0)
        solve.assert_not_called()

    def test_missing_slice_connection_stops_without_gcs_or_generating_regions(self):
        failure = self.failure_q(4)
        save_iris_checkpoint(self.checkpoint, [cube(failure, .002), cube(self.goal_q, .002)],
                             dict(model=scene_metadata([]), generation_seconds=0))
        with patch('scripts.spatial4R_post_failure_planner.locked_joint_ik', return_value=([self.goal_q], root_report())), \
                patch('scripts.spatial4R_post_failure_planner.solve_target_sets') as solve:
            report = plan_after_failure(failure, 4, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "no_path_in_iris_coverage")
        self.assertEqual(report["connected_goal_count"], 0)
        solve.assert_not_called()

    def test_already_at_goal_returns_supplied_windings_and_skips_ik(self):
        q = self.goal_q+2*np.pi*np.array([1, -2, 3, -1])
        with patch('scripts.spatial4R_post_failure_planner.locked_joint_ik') as ik:
            report = plan_after_failure(q, 4, self.goal, self.checkpoint)
        self.assertTrue(report["success"])
        self.assertEqual(report["cost"], 0)
        np.testing.assert_array_equal(report["waypoints"][0], q)
        ik.assert_not_called()

    def test_source_collision_stops_before_ik(self):
        with patch.object(model, "collision_free", return_value=False), \
                patch('scripts.spatial4R_post_failure_planner.locked_joint_ik') as ik:
            report = plan_after_failure(self.failure_q(4), 4, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "start_in_collision")
        ik.assert_not_called()

    def test_invalid_inputs(self):
        for q, joint, goal in (([0]*3, 1, self.goal), ([0]*4, 5, self.goal),
                                ([np.nan]*4, 1, self.goal), ([0]*4, 1, [0, 0])):
            with self.assertRaises(ValueError):
                plan_after_failure(q, joint, goal, self.checkpoint)

    def test_chart_preserves_positive_pi_and_nonzero_windings(self):
        supplied = np.array([3*np.pi, -5*np.pi, 4*np.pi+.1, -2*np.pi-.2])
        chart, offsets = chart_configuration(supplied)
        self.assertEqual(chart[0], np.pi)
        np.testing.assert_allclose(chart+offsets, supplied)


    @unittest.skipUnless(HAS_DRAKE, "Drake is required")
    def test_full_gcs_for_each_lock_preserves_windings_and_uses_unweighted_length(self):
        for joint in range(1, 5):
            supplied = self.failure_q(joint)+2*np.pi*np.array([1, -2, 3, -1])
            report = plan_after_failure(supplied, joint, self.goal, self.checkpoint,
                                        ik_max_starts=32, solver_backend="mosek")
            self.assertTrue(report["success"], report)
            path = np.asarray(report["waypoints"])
            np.testing.assert_array_equal(path[:, joint-1], np.full(len(path), supplied[joint-1]))
            np.testing.assert_allclose(path[0], supplied, atol=1e-8)
            self.assertLess(report["endpoint_error"], 1e-8)
            self.assertEqual(report["gcs_solve_calls"], 1)
            self.assertFalse(report["reliability_used"])
            self.assertEqual(report["solver_settings"]["max_rounded_paths"], 200)
            self.assertEqual(report["solver_settings"]["max_rounding_trials"], 300)
            self.assertAlmostEqual(report["cost"], np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), places=7)
            self.assertAlmostEqual(report["cost"], np.linalg.norm(supplied-(self.goal_q+2*np.pi*np.array([1, -2, 3, -1]))), places=6)


if __name__ == "__main__":
    unittest.main()
