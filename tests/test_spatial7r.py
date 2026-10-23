"""SE(3) geometry, eight-branch aggregation and 7D Drake/GCS integration."""
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import spatial7R_model as model
from scripts.helper_functions import compute_reliability_by_f_list
from scripts.spatial7R_iris import build_plant
from scripts.spatial7R_two_stage_gcs_planner import (
    ResourceLimit, local_reliability_regions, path_is_in_graph, plan_before_failure,
)
from scripts.planar3R_iris_regions import ReliablePolytope
from scripts.planar3R_ssm_envelopes import build_envelopes, intersect_envelopes, sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets

try:
    from pydrake.geometry.optimization import Point
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def box(lower, upper):
    return ReliablePolytope(np.vstack((np.eye(7), -np.eye(7))), np.r_[upper, -np.asarray(lower)])


class Spatial7RTests(unittest.TestCase):
    def test_default_task_has_30cm_translation_and_20_degree_orientation_change(self):
        start = model.pose(model.START_CONFIGURATION)
        goal = model.pose(model.GOAL_REFERENCE_CONFIGURATION)
        self.assertAlmostEqual(np.linalg.norm(goal[:3, 3]-start[:3, 3]), .30, places=12)
        angle = np.linalg.norm(Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec())
        self.assertAlmostEqual(np.rad2deg(angle), 20., places=10)
        self.assertTrue(model.collision_free(model.START_CONFIGURATION, model.OBSTACLES))
        self.assertTrue(model.collision_free(model.GOAL_REFERENCE_CONFIGURATION, model.OBSTACLES))

    def test_reference_constants_are_read_without_executing_the_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.py"
            path.write_text(model.REFERENCE_FILE.read_text(encoding="utf-8-sig")+"\nraise AssertionError('Must not execute')\n", encoding="utf-8")
            parameters = model.reference_parameters(path)
        np.testing.assert_allclose(parameters["a"], [.4, .8, .2, 1, .6, .4, 0])
        np.testing.assert_allclose(parameters["d"], [-.4, -.6, .2, .6, -.8, .2, 0])
        np.testing.assert_allclose(parameters["alpha"], np.deg2rad([-62, -79, 90, 29, 81, -80, 0]))
        self.assertEqual(parameters["reliabilities"], [.2, .3, .4, .5, .6, .7, .8])

    def test_full_pose_residual_jacobian_matches_finite_differences(self):
        goal = model.pose(model.GOAL_REFERENCE_CONFIGURATION)
        for q in np.random.default_rng(4).uniform(-2, 2, (12, 7)):
            numeric = np.column_stack([(model.pose_residual(q+np.eye(7)[i]*1e-6, goal)
                                        -model.pose_residual(q-np.eye(7)[i]*1e-6, goal))/2e-6 for i in range(7)])
            np.testing.assert_allclose(numeric, model.residual_jacobian(q, goal), atol=1e-7)

    def test_last_zero_offset_joint_changes_orientation_but_not_position(self):
        q = model.GOAL_REFERENCE_CONFIGURATION.copy()
        first = model.pose(q)
        q[-1] += .8
        second = model.pose(q)
        np.testing.assert_allclose(first[:3, 3], second[:3, 3], atol=1e-12)
        self.assertGreater(np.linalg.norm(model.pose_residual(q, first)[3:]), .79)
        np.testing.assert_allclose(model.jacobian(q)[:3, -1], 0, atol=1e-12)

    def test_half_turn_orientation_error_does_not_disappear(self):
        goal = model.pose(model.GOAL_REFERENCE_CONFIGURATION)
        goal[:3, :3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ goal[:3, :3]
        self.assertAlmostEqual(np.linalg.norm(model.pose_residual(model.GOAL_REFERENCE_CONFIGURATION, goal)[3:]), np.pi)

    def test_actual_regular_ssm_closes_and_preserves_position_and_orientation(self):
        q = model.GOAL_REFERENCE_CONFIGURATION
        goal = model.pose(q)
        branch = model.trace_ssm(q, goal)
        self.assertIsNotNone(branch)
        self.assertGreater(len(branch), 30)
        self.assertLess(max(np.linalg.norm(model.pose_residual(p, goal)) for p in branch), 1e-8)
        np.testing.assert_allclose(model.wrap(branch[0]-branch[-1]), 0, atol=1e-12)

    def test_eight_branch_cap_and_search_diagnostics(self):
        starts = [np.array([-1.75+.5*i, 0, 0, 0, 0, 0, 0]) for i in range(8)]
        # Controlled branch-search fixture, not eight physical robot SSMs.
        with patch.object(model, "least_squares", side_effect=[SimpleNamespace(x=q, fun=np.zeros(6)) for q in starts]) as ik, \
                patch.object(model, "refine_ik", side_effect=lambda q, goal: q), \
                patch.object(model, "trace_ssm", side_effect=lambda q, goal: np.array([q, q+[0, 0, 0, 0, 0, 0, .1], q])):
            branches, report = model.compute_ssms(np.eye(4), attempts=20)
        self.assertEqual(len(branches), 8)
        self.assertEqual(ik.call_count, 8)
        self.assertEqual(report["branch_limit"], 8)
        self.assertEqual(report["termination_reason"], "branch_limit_reached")
        self.assertFalse(report["all_branches_certified"])

    def test_reliability_matches_reference_conditional_formula_for_all_128_sets(self):
        lookup = model.reliability_lookup()
        self.assertEqual(len(lookup), 128)
        for subset, actual in lookup.items():
            f_list = [0, *subset] if subset else []
            self.assertAlmostEqual(actual, compute_reliability_by_f_list(f_list, model.JOINT_RELIABILITIES), places=13)

    def test_eight_branches_all_contribute_to_local_refinement_and_target_pool(self):
        parent = box(np.r_[-1, np.full(6, -.1)], np.r_[1, np.full(6, .1)])
        branches = [np.array([np.r_[center-.03, np.full(6, -.2)], np.r_[center+.03, np.full(6, .2)]])
                    for center in np.linspace(-.7, .7, 8)]
        regions, report = local_reliability_regions([parent], branches)
        self.assertEqual(report["local_partition_cell_count"], 17)
        for center in np.linspace(-.7, .7, 8):
            point = np.r_[center, np.zeros(6)]
            self.assertTrue(any(r.contains(point) and r.reliability == 1 for r in regions))
        point = np.r_[.95, np.zeros(6)]
        expected = model.reliability_lookup()[frozenset(range(2, 8))]
        self.assertTrue(any(r.contains(point) and abs(r.reliability-expected) < 1e-12 for r in regions))
        report = {}
        envelopes = build_envelopes(branches, arc_spacing=.044, sampling_report=report)
        targets = intersect_envelopes(envelopes, [parent])
        self.assertEqual({envelopes[t["envelope"]].branch for t in targets}, set(range(8)))
        self.assertEqual(len(report["ssm_branch_sample_counts"]), 8)

    def test_branch_recovery_sets_are_unioned_and_iris_halfspaces_remain(self):
        parent = box(np.full(7, -.2), np.full(7, .2))
        parent.A = np.vstack((parent.A, [1, 1, 0, 0, 0, 0, 0]))
        parent.b = np.r_[parent.b, .08]
        bounds = [np.tile([1., 2.], (7, 1)) for _ in range(2)]
        bounds[0][0] = [-.1, .1]
        bounds[1][1] = [-.1, .1]
        branches = [np.array([b[:, 0], b[:, 1]]) for b in bounds]
        regions, _ = local_reliability_regions([parent], branches)
        point = np.zeros(7)
        actual = max(r.reliability for r in regions if r.contains(point))
        lookup = model.reliability_lookup()
        self.assertEqual(actual, lookup[frozenset({1, 2})])
        self.assertGreater(actual, max(lookup[frozenset({1})], lookup[frozenset({2})]))
        for point in np.random.default_rng(7).uniform(-.25, .25, (200, 7)):
            self.assertEqual(parent.contains(point), any(r.contains(point) for r in regions))

    def test_eight_branches_union_all_seven_failures_without_duplicate_credit(self):
        parent = box(np.full(7, -.02), np.full(7, .02))
        bounds = [np.tile([1., 2.], (7, 1)) for _ in range(8)]
        for i, bound in enumerate(bounds):
            bound[i % 7] = [-.1, .1]
        branches = [np.array([b[:, 0], b[:, 1]]) for b in bounds]
        regions, report = local_reliability_regions([parent], branches)
        self.assertEqual(report["reliability_rule"], "union_of_branch_recovery_sets")
        self.assertEqual(report["local_partition_cell_count"], 1)
        self.assertEqual(len(regions), 1)
        self.assertTrue(regions[0].contains(np.zeros(7)))
        self.assertEqual(regions[0].reliability, 1.0)

    def test_budget_stops_before_allocating_large_partition(self):
        parent = box(np.full(7, -.2), np.full(7, .2))
        branch = np.array([np.full(7, -.1), np.full(7, .1)])
        with self.assertRaises(ResourceLimit):
            local_reliability_regions([parent], [branch], max_cells=1)

    def test_radial_unreachability_stops_before_loading_or_solving(self):
        goal = np.eye(4)
        goal[:3, 3] = [100, 0, 0]
        with patch("scripts.spatial7R_two_stage_gcs_planner.load_iris_checkpoint") as load:
            report = plan_before_failure(goal_pose=goal, checkpoint="missing.npz")
        self.assertEqual(report["status"], "goal_unreachable")
        self.assertEqual(report["gcs_solve_calls"], 0)
        load.assert_not_called()

    def test_pose_and_branch_limit_validation(self):
        for goal in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                model.validate_pose(goal)
        with self.assertRaises(ValueError):
            model.compute_ssms(np.eye(4), max_branches=9)

    def test_graph_validation_rejects_target_or_region_violations(self):
        region = box(np.full(7, -1), np.full(7, 1))
        start, goal = np.zeros(7), np.full(7, .2)
        solved = dict(waypoints=np.array([start, goal]), region_route=[0])
        self.assertTrue(path_is_in_graph(solved, [region], start, sample_hull([goal])))
        self.assertFalse(path_is_in_graph(solved, [region], start, sample_hull([goal+.01])))
        self.assertFalse(path_is_in_graph(solved, [box(np.full(7, -.1), np.full(7, .1))], start, sample_hull([goal])))


@unittest.skipUnless(HAS_DRAKE, "Drake is required")
class Spatial7RDrakeTests(unittest.TestCase):
    def test_seven_joint_plant_pose_and_collisions_match_analytic_dh(self):
        q = np.zeros(7)
        q[0] = .3
        first, second = model.kinematics(q)[3][1]
        obstacles = [dict(type="sphere", center=((first+second)/2).tolist(), radius=.05)]
        diagram, plant, scene = build_plant(obstacles)
        self.assertEqual(plant.num_positions(), 7)
        context = diagram.CreateDefaultContext()
        pc = plant.GetMyMutableContextFromRoot(context)
        sc = scene.GetMyContextFromRoot(context)
        collisions = free = 0
        for q in np.random.default_rng(9).uniform(-np.pi, np.pi, (100, 7)):
            plant.SetPositions(pc, q)
            actual = plant.CalcRelativeTransform(pc, plant.world_frame(), plant.GetFrameByName("frame7")).GetAsMatrix4()
            np.testing.assert_allclose(actual, model.pose(q), atol=1e-12)
            colliding = scene.get_query_output_port().Eval(sc).HasCollisions()
            self.assertEqual(colliding, not model.collision_free(q, obstacles))
            collisions += colliding
            free += not colliding
        self.assertGreater(collisions, 0)
        self.assertGreater(free, 0)

    def test_gcs_charges_all_seven_joint_coordinates(self):
        region = box(np.full(7, -1), np.full(7, 1))
        start, goal = np.zeros(7), np.full(7, .2)
        solved = solve_target_sets([region], [], start, [dict(polytope=sample_hull([goal]), region=0)],
                                   [.4], max_rounded_paths=5, max_rounding_trials=10, solver_backend="mosek")
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved["cost"], .4*np.linalg.norm(goal), places=7)


if __name__ == "__main__":
    unittest.main()
