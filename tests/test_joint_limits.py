import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from planar3R_collision import link_points
from planar3R_iris_gcs_planner import PlanningStopped, plan_with_iris_gcs
from planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from planar3R_joint_limits import (
    clip_regions_to_joint_limits, find_limited_ik, validate_joint_limits,
    within_joint_limits, workspace_bounds,
)
from planar3R_post_failure_planner import locked_joint_ik, plan_after_failure

try:
    from pydrake.geometry.optimization import Point
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


class JointLimitTests(unittest.TestCase):
    def test_invalid_and_partial_limits_are_rejected(self):
        for lower, upper in (([-1] * 3, None), ([0] * 3, [0] * 3),
                             ([-4] * 3, [1] * 3), ([np.nan] * 3, [1] * 3)):
            with self.assertRaises(ValueError):
                validate_joint_limits(lower, upper)

    def test_cartesian_enclosure_contains_sampled_forward_kinematics(self):
        lower, upper = np.array([-2, -0.4, 0.2]), np.array([0.3, 1.2, 2])
        lo, hi = workspace_bounds(lower, upper)
        for q in np.random.default_rng(17).uniform(lower, upper, size=(100, 3)):
            point = link_points(q)[-1]
            self.assertTrue(np.all(point >= lo - 1e-10))
            self.assertTrue(np.all(point <= hi + 1e-10))

    def test_outside_artificial_workspace_is_rejected_without_ssm_or_gcs(self):
        # (-3,0) is physically reachable only with q1=+/-pi, q2=q3=0.
        with patch("planar3R_gcs_planner.ssm_candidate_points") as candidates:
            with self.assertRaises(PlanningStopped) as stopped:
                plan_with_iris_gcs([0.5, 0.5], [-3, 0], [], "unused.npz",
                                   artificial_lower=[-3] * 3, artificial_upper=[3] * 3)
            self.assertEqual(stopped.exception.report["status"], "goal_unreachable")
            self.assertEqual(stopped.exception.report["gcs_solve_calls"], 0)
            candidates.assert_not_called()

    def test_limited_ik_is_within_limits_and_reaches_the_target(self):
        lower, upper = validate_joint_limits([-1] * 3, [1] * 3)
        target = link_points([0.6, -0.4, 0.2])[-1]
        q, status = find_limited_ik(target, lower, upper)
        self.assertEqual(status, "reachable")
        self.assertTrue(within_joint_limits(q, lower, upper))
        np.testing.assert_allclose(link_points(q)[-1], target, atol=1e-8)

    def test_exhausted_search_is_not_a_false_unreachable_claim(self):
        with patch("planar3R_joint_limits.least_squares") as solve:
            solve.return_value.x = np.zeros(3)
            q, status = find_limited_ik([0, 0], np.full(3, -1), np.full(3, 1), max_boxes=0)
        self.assertIsNone(q)
        self.assertEqual(status, "unresolved")

    def test_clipping_preserves_slanted_boundaries(self):
        A = np.vstack((np.eye(3), -np.eye(3), [1, 1, 0]))
        region = ReliablePolytope(A, np.array([3, 3, 3, 3, 3, 3, 0.2]))
        clipped = clip_regions_to_joint_limits([region], np.full(3, -0.5), np.full(3, 0.5))
        self.assertEqual(len(clipped), 1)
        self.assertTrue(clipped[0].contains([0, 0, 0]))
        self.assertFalse(clipped[0].contains([0.4, 0.4, 0]))
        self.assertFalse(clipped[0].contains([0, 0, 0.6]))

    @unittest.skipUnless(HAS_DRAKE, "Requires robotics Drake bindings")
    def test_prefailure_path_stays_inside_box_and_postfailure_releases_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.npz"
            full = ReliablePolytope(np.vstack((np.eye(3), -np.eye(3))), np.full(6, np.pi))
            save_iris_checkpoint(path, [full], {"model": {
                "link_lengths": [1, 1, 1], "link_radius": 0.0, "obstacles": [],
            }})
            start, goal_q = np.array([0.4, -0.8, 0.7]), np.array([0.7, -0.5, 0.3])
            with patch("planar3R_gcs_planner.ssm_candidate_points", side_effect=[[start], [goal_q]]):
                result = plan_with_iris_gcs(link_points(start)[-1], link_points(goal_q)[-1], [], path,
                                            reliability_source="none", candidate_limit=1,
                                            artificial_lower=[-1] * 3, artificial_upper=[1] * 3)
            self.assertTrue(within_joint_limits(result["waypoints"], np.full(3, -1), np.full(3, 1)))
            failure = np.array([0.4, -0.4, 0.3])
            target = link_points([0.9, -0.3, 0.3])[-1]
            goals, _ = locked_joint_ik(target, 3, failure[2])
            self.assertTrue(all(not within_joint_limits(q, np.full(3, -0.5), np.full(3, 0.5)) for q in goals))
            after = plan_after_failure(failure, 3, target, path)
            self.assertTrue(after["success"], after)
            self.assertTrue(after["artificial_limits_released"])
            self.assertFalse(within_joint_limits(after["waypoints"][-1], np.full(3, -0.5), np.full(3, 0.5)))


if __name__ == "__main__":
    unittest.main()
