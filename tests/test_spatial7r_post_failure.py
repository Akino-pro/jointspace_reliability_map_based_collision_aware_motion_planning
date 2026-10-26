import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import spatial7R_model as model
from scripts.spatial7R_iris import scene_metadata
from scripts.planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from scripts import spatial7R_post_failure_planner as planner

try:
    from pydrake.geometry.optimization import Point
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def cube(center, radius):
    center = np.asarray(center)
    return ReliablePolytope(np.vstack((np.eye(7), -np.eye(7))),
                            np.r_[center+radius, -center+radius], reliability=.03)


class Spatial7RFailureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.checkpoint = Path(self.folder.name)/"regions.npz"
        self.goal_q = model.GOAL_REFERENCE_CONFIGURATION.copy()
        self.goal = model.pose(self.goal_q)
        save_iris_checkpoint(self.checkpoint, [cube(self.goal_q, .25)],
                             dict(model=scene_metadata([]), generation_seconds=0))

    def failure(self, joint):
        q = self.goal_q+np.array([.02, -.03, .01, -.02, .01, .02, -.01])
        q[joint-1] = self.goal_q[joint-1]
        return q

    def test_full_pose_ik_for_every_locked_joint(self):
        for joint in range(1, 8):
            q = self.failure(joint)
            roots, report = planner.locked_joint_ik(self.goal, joint, q[joint-1], q, max_starts=8)
            self.assertEqual(report["starts_attempted"], 8)
            self.assertFalse(report["completeness_certified"])
            self.assertTrue(roots, (joint, report))
            self.assertEqual(report["singular_solution_count"], 0)
            self.assertTrue(any(np.linalg.norm(model.wrap(p-self.goal_q)) < 1e-6 for p in roots))
            for p in roots:
                self.assertEqual(p[joint-1], q[joint-1])
                self.assertLess(np.linalg.norm(model.pose_residual(p, self.goal)), 1e-8)

    def test_ik_does_not_inherit_four_root_cap(self):
        values = [np.full(6, -.8+.2*i) for i in range(6)]
        with patch.object(planner, "least_squares", side_effect=[SimpleNamespace(x=x) for x in values]), \
                patch.object(model, "pose_residual", return_value=np.zeros(6)), \
                patch.object(model, "jacobian", return_value=np.column_stack((np.eye(6), np.zeros(6)))):
            roots, report = planner.locked_joint_ik(self.goal, 7, .2, max_starts=6)
        self.assertEqual(len(roots), 6)
        self.assertEqual(report["physical_solution_count"], 6)

    def test_slice_preserves_all_original_halfspaces_for_every_lock(self):
        parent = cube(np.zeros(7), 2)
        parent.A = np.vstack((parent.A, [1, 2, -1, .5, -.2, .7, -.8]))
        parent.b = np.r_[parent.b, .8]
        for joint in range(1, 8):
            active = [i for i in range(7) if i != joint-1]
            sliced = planner.slice_regions([parent], joint, .3)
            self.assertEqual(len(sliced), 1)
            self.assertEqual(sliced[0].reliability, 1.)
            for u in np.random.default_rng(joint).uniform(-3, 3, (50, 6)):
                q = np.zeros(7)
                q[joint-1], q[active] = .3, u
                self.assertEqual(parent.contains(q), sliced[0].contains(u))
            self.assertEqual(planner.slice_regions([parent], joint, 3), [])

    def test_unreachable_stops_before_loading_checkpoint(self):
        goal = self.goal.copy()
        goal[:3, 3] = [100, 0, 0]
        with patch.object(planner, "load_iris_checkpoint") as load:
            report = planner.plan_after_failure(self.failure(7), 7, goal, "missing.npz")
        self.assertEqual(report["status"], "unreachable")
        load.assert_not_called()

    def test_ik_failure_singular_and_collision_stop_before_gcs(self):
        for roots, singular, collision, status in (
                ([], 0, False, "ik_search_failed"),
                ([self.goal_q], 1, False, "unsupported_singular_ik"),
                ([self.goal_q], 0, True, "all_found_goals_in_collision")):
            failure = self.failure(7)
            with patch.object(planner, "locked_joint_ik", return_value=(roots, dict(singular_solution_count=singular))), \
                    patch.object(planner, "solve_target_sets") as solve, \
                    patch.object(model, "collision_free", side_effect=lambda q, *args: not collision or np.linalg.norm(q-failure) < 1e-10):
                report = planner.plan_after_failure(failure, 7, self.goal, self.checkpoint)
            self.assertEqual(report["status"], status)
            solve.assert_not_called()

    def test_missing_slice_connection_is_not_unreachable(self):
        q = self.failure(7)
        save_iris_checkpoint(self.checkpoint, [cube(q, .001), cube(self.goal_q, .001)],
                             dict(model=scene_metadata([]), generation_seconds=0))
        with patch.object(planner, "locked_joint_ik", return_value=([self.goal_q], dict(singular_solution_count=0))):
            report = planner.plan_after_failure(q, 7, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "no_path_in_iris_coverage")
        self.assertEqual(report["gcs_solve_calls"], 0)

    def test_already_at_pose_preserves_windings(self):
        q = self.goal_q+2*np.pi*np.array([1, -2, 3, -1, 2, -1, 1])
        with patch.object(planner, "locked_joint_ik") as ik:
            report = planner.plan_after_failure(q, 7, self.goal, self.checkpoint)
        self.assertTrue(report["success"])
        self.assertEqual(report["cost"], 0.)
        np.testing.assert_array_equal(report["waypoints"][0], q)
        ik.assert_not_called()

    def test_invalid_inputs_and_scene(self):
        for q, joint, goal in (([0]*6, 1, self.goal), ([0]*7, 8, self.goal),
                               ([np.nan]*7, 1, self.goal), ([0]*7, 1, np.zeros((4, 4)))):
            with self.assertRaises(ValueError):
                planner.plan_after_failure(q, joint, goal, self.checkpoint)
        scene = scene_metadata([])
        scene["dh"] = dict(scene["dh"], a=[0]*7)
        save_iris_checkpoint(self.checkpoint, [cube(self.goal_q, .25)], dict(model=scene))
        with self.assertRaisesRegex(ValueError, "DH geometry"):
            planner.plan_after_failure(self.failure(7), 7, self.goal, self.checkpoint)

    def test_source_collision_stops_before_ik(self):
        with patch.object(model, "collision_free", return_value=False), \
                patch.object(planner, "locked_joint_ik") as ik:
            report = planner.plan_after_failure(self.failure(7), 7, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "start_in_collision")
        ik.assert_not_called()

    @unittest.skipUnless(HAS_DRAKE, "Drake is required")
    def test_real_gcs_every_lock_uses_unweighted_length_and_preserves_windings(self):
        offsets = 2*np.pi*np.array([1, -2, 3, -1, 2, -1, 1])
        for joint in range(1, 8):
            q = self.failure(joint)+offsets
            report = planner.plan_after_failure(q, joint, self.goal, self.checkpoint, ik_max_starts=8)
            self.assertTrue(report["success"], report)
            path = np.asarray(report["waypoints"])
            np.testing.assert_array_equal(path[:, joint-1], np.full(len(path), q[joint-1]))
            self.assertLess(report["orientation_error"], 1e-8)
            self.assertLess(report["endpoint_error"], 1e-8)
            self.assertFalse(report["reliability_used"])
            self.assertTrue(report["graph_membership_valid"])
            self.assertAlmostEqual(report["cost"], np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), places=7)
            self.assertAlmostEqual(report["cost"], np.linalg.norm(q-self.goal_q-offsets), places=6)

    @unittest.skipUnless(HAS_DRAKE, "Drake is required")
    def test_multiple_targets_choose_shorter_path_and_reject_wrong_target(self):
        q = self.failure(7)
        near, far = q.copy(), q.copy()
        near[0] += .01
        far[0] += .10
        # Controlled task endpoints isolate graph choice from physical IK search.
        with patch.object(planner, "locked_joint_ik", return_value=([far, near], dict(singular_solution_count=0))), \
                patch.object(model, "pose_residual", side_effect=lambda p, goal: np.ones(6) if np.linalg.norm(p-q) < 1e-10 else np.zeros(6)):
            report = planner.plan_after_failure(q, 7, self.goal, self.checkpoint)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["connected_goal_count"], 2)
        self.assertEqual(report["selected_ik_solution_index"], 1)
        self.assertAlmostEqual(report["cost"], .01, places=7)
        # A geometrically plausible route ending at a different goal is invalid.
        bad = dict(waypoints=[q[:6], near[:6]], target=0, region_route=[0],
                   cost=.01, solver_name="fixture", rounded_path_count=1, quality={})
        with patch.object(planner, "locked_joint_ik", return_value=([far, near], dict(singular_solution_count=0))), \
                patch.object(planner, "solve_target_sets", return_value=bad):
            report = planner.plan_after_failure(q, 7, self.goal, self.checkpoint)
        self.assertEqual(report["status"], "path_validation_failed")


if __name__ == "__main__":
    unittest.main()
