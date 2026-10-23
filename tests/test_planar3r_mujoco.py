import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.planar3R_collision import is_collision_free, link_points
from scripts.planar3R_mujoco_viewer import ReplayScene, load_scene, make_mjcf, pose_at, set_pose, trajectory

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def scene(points=None, lengths=None):
    return ReplayScene(
        np.array([[0., 0., 0.], [1., 0.5, -0.5]] if points is None else points),
        np.array([1., 1., 1.] if lengths is None else lengths),
        [dict(type="circle", center=[0.75, 0.75], radius=0.15)], 0.02, np.array([1.5, 1.5]))


class ReplayTests(unittest.TestCase):
    def test_large_joint_rotation_uses_planned_segment_not_periodic_shortcut(self):
        points, times = trajectory(scene([[-3., 0., 0.], [3., 0., 0.]]))
        np.testing.assert_allclose(pose_at(points, times, times[-1]/2), [0, 0, 0])
        self.assertAlmostEqual(times[-1], 6/0.65)

    def test_duplicate_points_and_singleton_paths_are_supported(self):
        for poses in ([[1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]):
            points, times = trajectory(scene(poses))
            np.testing.assert_allclose(pose_at(points, times, 20), [1, 2, 3])

    def test_saved_post_failure_path_preserves_locked_joint(self):
        saved = dict(success=True, waypoints=[[0.1, 1.2, 0.3], [0.4, 1.2, 0.6]],
                     failed_joint=2, link_lengths=[1, 1, 1], link_radius=0.02)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"path.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            replay = load_scene(path)
            points, times = trajectory(replay)
            self.assertEqual(replay.failed_joint, 2)
            self.assertEqual(pose_at(points, times, times[-1]/2)[1], 1.2)
            saved["waypoints"][-1][1] = 1.3
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scene(path)

    def test_spatial_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"path.json"
            path.write_text(json.dumps(dict(waypoints=[[0, 0, 0, 0]])), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scene(path)


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is required")
class MujocoModelTests(unittest.TestCase):
    def test_tool_position_matches_analytic_fk_for_source_link_lengths(self):
        replay = scene(lengths=[0.4, 0.6, 0.8])
        model = mujoco.MjModel.from_xml_string(make_mjcf(replay))
        data = mujoco.MjData(model)
        self.assertEqual(model.nq, 3)
        self.assertFalse(np.any(model.jnt_limited))
        tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        for q in np.random.default_rng(8).uniform(-8, 8, (50, 3)):
            set_pose(model, data, q)
            np.testing.assert_allclose(data.site_xpos[tip][:2], link_points(q, replay.lengths)[-1], atol=1e-12)

    def test_circle_contacts_match_analytic_checks_without_self_collision(self):
        replay = scene()
        model = mujoco.MjModel.from_xml_string(make_mjcf(replay))
        data = mujoco.MjData(model)
        collisions, free = 0, 0
        for q in np.random.default_rng(9).uniform(-np.pi, np.pi, (150, 3)):
            set_pose(model, data, q)
            colliding = any(contact.dist < 0 for contact in data.contact)
            expected = not is_collision_free(q, replay.obstacles, replay.lengths, replay.link_radius)
            self.assertEqual(colliding, expected)
            collisions += colliding
            free += not colliding
        self.assertGreater(collisions, 0)
        self.assertGreater(free, 0)


if __name__ == "__main__":
    unittest.main()
