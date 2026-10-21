import ast
import itertools
import unittest

import numpy as np

import spatial4R_model as model
from spatial4R_iris import build_plant
from spatial4R_two_stage_gcs_planner import plan_before_failure
from planar3R_bbox_reliability_partition import partition_multiple_branches, merge_equal_reliability_boxes
from planar3R_iris_regions import ReliablePolytope, split_iris_regions
from planar3R_ssm_envelopes import build_envelopes, intersect_envelopes, sample_hull
from planar3R_two_stage_gcs_planner import solve_target_sets

try:
    from pydrake.geometry.optimization import HPolyhedron
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


class SpatialModelTests(unittest.TestCase):
    def test_reference_uses_final_active_rcmax_parameters(self):
        parameters = model.reference_parameters(parameter_set="rcmax")
        self.assertEqual(parameters["reliabilities"], [0.9, 0.7, 0.5, 0.3])
        self.assertAlmostEqual(parameters["a"][0], 0.04732540449946272)
        self.assertAlmostEqual(parameters["alpha"][0], 1.1599121046989387)

    def test_first_ftwmax_robot_is_selected_without_running_reference_script(self):
        parameters = model.reference_parameters(parameter_set="ftwmax")
        np.testing.assert_allclose(parameters["a"], [0.5, 0.48, 0.76, 0.95])
        np.testing.assert_allclose(parameters["d"], [-0.29, 0, 0.05, 1])
        np.testing.assert_allclose(parameters["alpha"], np.deg2rad([85, -53, -89, 68]))
        self.assertEqual(model.ROBOT_PARAMETER_SET, "ftwmax")

    def test_position_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        for q in rng.uniform(-2, 2, (10, 4)):
            numeric = np.column_stack([(model.position(q+np.eye(4)[i]*1e-6)
                                        - model.position(q-np.eye(4)[i]*1e-6))/2e-6
                                       for i in range(4)])
            np.testing.assert_allclose(model.jacobian(q), numeric, atol=1e-8)

    def test_reliability_matches_all_fifteen_reference_formulas(self):
        source = ast.parse(model.REFERENCE_FILE.read_text(encoding="utf-8"))
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                        and n.name == "reliability_computation")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(model.REFERENCE_FILE), "exec"), namespace)
        reference = namespace["reliability_computation"](*model.JOINT_RELIABILITIES)
        lookup = model.reliability_lookup()
        actual = [lookup[frozenset(indices)] for size in range(1, 5)
                  for indices in itertools.combinations(range(1, 5), size)]
        np.testing.assert_allclose(actual, reference, atol=1e-14)

    def test_four_dimensional_partition_merge_preserves_total_volume(self):
        bounds = [np.tile([-2., 0.], (4, 1)), np.tile([-1., 1.], (4, 1))]
        raw = partition_multiple_branches(bounds, model.reliability_lookup())
        merged = merge_equal_reliability_boxes(raw)
        self.assertEqual(len(raw), 625)
        self.assertLess(len(merged), len(raw))
        self.assertAlmostEqual(sum(np.prod(c["upper"]-c["lower"]) for c in merged), (2*np.pi)**4)

    def test_four_dimensional_hulls_and_intersections(self):
        points = np.array([[0., 0., 0., 0.], [1., 0., 0., 0.]])
        envelopes = build_envelopes([points], arc_spacing=0.2)
        region = ReliablePolytope(np.vstack((np.eye(4), -np.eye(4))), np.ones(8))
        targets = intersect_envelopes(envelopes, [region])
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0]["polytope"].contains([0.4, 0, 0, 0]))
        self.assertFalse(targets[0]["polytope"].contains([0.4, 0, 0, 0.01]))

    def test_box_implied_halfspaces_are_removed_without_changing_intersection(self):
        parent = ReliablePolytope(np.vstack((np.eye(4), -np.eye(4), [1, 1, 0, 0])),
                                  np.r_[np.full(8, 3.0), 0.5])
        cells = [dict(lower=np.zeros(4), upper=np.ones(4), reliability=0.7)]
        children = split_iris_regions([parent], cells)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].A.shape[0], 9)
        rng = np.random.default_rng(12)
        for q in rng.uniform(-1, 2, (200, 4)):
            expected = parent.contains(q) and np.all(q >= 0) and np.all(q <= 1)
            self.assertEqual(children[0].contains(q), expected)

    def test_trace_closes_on_the_same_position_manifold(self):
        q = model.GOAL_REFERENCE_CONFIGURATION
        goal = model.position(q)
        branch = model.trace_ssm(q, goal)
        self.assertIsNotNone(branch)
        self.assertLess(np.max([np.linalg.norm(model.position(p)-goal) for p in branch]), 1e-8)
        self.assertLess(np.linalg.norm(model.wrap(branch[0]-branch[-1])), 1e-8)

    def test_capsule_collision_checker_detects_a_blocked_link(self):
        first, second = model.kinematics(model.START_CONFIGURATION)[3][-1]
        obstacle = dict(type="sphere", center=((first+second)/2).tolist(), radius=0.02)
        self.assertFalse(model.collision_free(model.START_CONFIGURATION, [obstacle]))
        self.assertTrue(model.collision_free(model.START_CONFIGURATION, []))

    def test_unreachable_goal_stops_before_region_loading(self):
        result = plan_before_failure(goal=[100, 0, 0], checkpoint="does-not-exist.npz")
        self.assertEqual(result["status"], "goal_unreachable")
        self.assertEqual(result["gcs_solve_calls"], 0)


@unittest.skipUnless(HAS_DRAKE, "Drake is required")
class SpatialDrakeTests(unittest.TestCase):
    def test_drake_tool_pose_and_capsules_match_analytic_dh_model(self):
        diagram, plant, scene = build_plant()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_context = scene.GetMyContextFromRoot(context)
        rng = np.random.default_rng(5)
        for q in rng.uniform(-np.pi, np.pi, (100, 4)):
            plant.SetPositions(plant_context, q)
            actual = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                                 plant.GetFrameByName("frame4")).GetAsMatrix4()
            np.testing.assert_allclose(actual, model.kinematics(q)[0], atol=1e-12)
            collision = scene.get_query_output_port().Eval(scene_context).HasCollisions()
            self.assertEqual(not collision, model.collision_free(q))

    def test_four_dimensional_gcs_charges_full_joint_length(self):
        region = ReliablePolytope(np.vstack((np.eye(4), -np.eye(4))), np.ones(8), reliability=0.6)
        start, goal = np.zeros(4), np.full(4, 0.2)
        solved = solve_target_sets([region], [], start, [dict(polytope=sample_hull([goal]), region=0)],
                                   [0.4], candidate_region_routes=[[0]],
                                   max_rounded_paths=5, max_rounding_trials=10)
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved["cost"], 0.4*np.linalg.norm(goal-start), places=6)
        np.testing.assert_allclose(solved["waypoints"][-1], goal, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
