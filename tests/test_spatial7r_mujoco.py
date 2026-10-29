"""Verify seven-joint MuJoCo geometry against the planning DH model."""
import unittest
import numpy as np
try:
    import mujoco
except ImportError:
    mujoco = None

from scripts import spatial7R_model as planning
from scripts.planar3R_mujoco_viewer import set_pose
from scripts.spatial4R_mujoco_viewer import PROJECT, SpatialReplayScene, collision_free, kinematics
from scripts.spatial7R_mujoco_viewer import load_scene, make_mjcf


@unittest.skipIf(mujoco is None, "MuJoCo is required")
class Spatial7RMujocoTests(unittest.TestCase):
    def test_saved_scene_and_random_configurations_match_planning_geometry(self):
        scene = load_scene(PROJECT / "results/spatial7r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json")
        model = mujoco.MjModel.from_xml_string(make_mjcf(scene))
        data = mujoco.MjData(model)
        tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        self.assertEqual(model.nq, 7)
        np.testing.assert_allclose(scene.waypoints[0], planning.START_CONFIGURATION)
        np.testing.assert_allclose(kinematics(scene, scene.waypoints[-1])[0], scene.goal_pose, atol=1e-12)
        for q in np.random.default_rng(29).uniform(-np.pi, np.pi, (50, 7)):
            set_pose(model, data, q)
            expected = planning.pose(q)
            np.testing.assert_allclose(data.site_xpos[tip], expected[:3, 3], atol=1e-12)
            np.testing.assert_allclose(data.site_xmat[tip].reshape(3, 3), expected[:3, :3], atol=1e-12)
            self.assertEqual(collision_free(scene, q), planning.collision_free(q, scene.obstacles))
            self.assertEqual(any(c.dist < 0 for c in data.contact), not collision_free(scene, q))

    def test_new_pose_preview_has_both_frames_and_no_unplanned_trace(self):
        scene = SpatialReplayScene(np.array([planning.START_CONFIGURATION, planning.GOAL_REFERENCE_CONFIGURATION]),
                                   planning.A, planning.D, planning.ALPHA, planning.OBSTACLES,
                                   planning.LINK_RADIUS, planning.position(planning.GOAL_REFERENCE_CONFIGURATION))
        scene.goal_pose = planning.pose(planning.GOAL_REFERENCE_CONFIGURATION)
        scene.pose_preview = True
        model = mujoco.MjModel.from_xml_string(make_mjcf(scene))
        self.assertEqual(model.nq, 7)
        for name in ("START_X", "START_Y", "START_Z", "GOAL_X", "GOAL_Y", "GOAL_Z"):
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name), 0)
        for i in range(model.ngeom):
            self.assertFalse((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("trace"))
        np.testing.assert_allclose(scene.obstacles[0]["center"],
                                   (planning.position(scene.waypoints[0])+scene.goal)/2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
