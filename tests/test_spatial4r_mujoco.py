"""Verify DH geometry and contacts against the planning model, including both robots."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from scripts import spatial4R_model as planning
from scripts.planar3R_mujoco_viewer import pose_at, set_pose, trajectory
from scripts.spatial4R_mujoco_viewer import (
    PROJECT, SpatialReplayScene, collision_free, kinematics, load_scene, make_mjcf,
)


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is required")
class SpatialMujocoTests(unittest.TestCase):
    def test_locked_joint_replay_validation_and_red_marker(self):
        saved = json.loads((PROJECT/'results/spatial4r/spatial4R_ftwmax_paths200_trials300.json').read_text())
        saved["failed_joint"] = 4
        for q in saved["waypoints"]:
            q[3] = -.3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"post_failure.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            scene = load_scene(path)
            self.assertEqual(scene.failed_joint, 4)
            model = mujoco.MjModel.from_xml_string(make_mjcf(scene))
            marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "joint_marker4")
            np.testing.assert_allclose(model.geom_rgba[marker], [1, .15, .15, 1])
            points, times = trajectory(scene)
            self.assertEqual(pose_at(points, times, times[-1]/2)[3], -.3)
            saved["waypoints"][-1][3] += .1
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scene(path)

    def test_both_dh_models_match_planner_and_mujoco_at_random_poses(self):
        original = planning.ROBOT_PARAMETER_SET
        try:
            for robot in ("ftwmax", "rcmax"):
                planning.select_robot(robot)
                scene = SpatialReplayScene(
                    np.array([[0., 0., 0., 0.], [1., .5, -.5, .4]]),
                    planning.A.copy(), planning.D.copy(), planning.ALPHA.copy(),
                    planning.LEGACY_OBSTACLES, planning.LINK_RADIUS, np.zeros(3))
                model = mujoco.MjModel.from_xml_string(make_mjcf(scene))
                data = mujoco.MjData(model)
                self.assertEqual(model.nq, 4)
                self.assertFalse(np.any(model.jnt_limited))
                tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
                hits = free = 0
                for q in np.random.default_rng(19).uniform(-8, 8, (300, 4)):
                    set_pose(model, data, q)
                    expected = planning.kinematics(q)
                    actual, segments = kinematics(scene, q)
                    np.testing.assert_allclose(actual, expected[0], atol=1e-12)
                    np.testing.assert_allclose(segments, expected[3], atol=1e-12)
                    np.testing.assert_allclose(data.site_xpos[tip], expected[0][:3, 3], atol=1e-12)
                    np.testing.assert_allclose(data.site_xmat[tip].reshape(3, 3), expected[0][:3, :3], atol=1e-12)
                    # Check every capsule's actual world endpoints, not only tool FK.
                    for i, (first, second) in enumerate(segments):
                        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"arm_link{i//2+1}_{i%2}")
                        center = data.geom_xpos[geom]
                        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
                        half = model.geom_size[geom, 1]
                        np.testing.assert_allclose(center, (first+second)/2, atol=1e-12)
                        if np.linalg.norm(first-second) > 1e-12:
                            error = min(np.linalg.norm(center-half*axis-first), np.linalg.norm(center+half*axis-first))
                            self.assertLess(error, 1e-12)
                    colliding = any(c.dist < 0 for c in data.contact)
                    self.assertEqual(colliding, not planning.collision_free(q, scene.obstacles))
                    self.assertEqual(collision_free(scene, q), planning.collision_free(q, scene.obstacles))
                    hits += colliding
                    free += not colliding
                self.assertGreater(hits, 0)
                self.assertGreater(free, 0)
        finally:
            planning.select_robot(original)

    def test_saved_path_uses_its_own_dh_parameters(self):
        scene = load_scene(PROJECT/'results/spatial4r/spatial4R_ftwmax_paths200_trials300.json')
        np.testing.assert_allclose(scene.a, [.5, .48, .76, .95])
        np.testing.assert_allclose(kinematics(scene, scene.waypoints[-1])[0][:3, 3], scene.goal, atol=1e-12)
        scene.waypoints = np.array([[-3., 0, 0, 3.], [3., 0, 0, -3.]])
        points, times = trajectory(scene)
        np.testing.assert_allclose(pose_at(points, times, times[-1]/2), np.zeros(4), atol=1e-12)

    def test_invalid_reports_are_rejected(self):
        saved = json.loads((PROJECT/'results/spatial4r/spatial4R_ftwmax_paths200_trials300.json').read_text())
        variants = [dict(success=False), dict(waypoints=[[0, 0, 0]]),
                    dict(obstacles=[dict(type="circle", center=[0, 0], radius=.1)]),
                    dict(robot=dict(a=[1, 1], d=[0]*4, alpha=[0]*4)), dict(link_radius=-1)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"path.json"
            for change in variants:
                path.write_text(json.dumps({**saved, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_scene(path)


if __name__ == "__main__":
    unittest.main()
