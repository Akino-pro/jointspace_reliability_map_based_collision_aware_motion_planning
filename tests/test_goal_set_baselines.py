import unittest
from types import SimpleNamespace

import numpy as np

from scripts.goal_set_baselines import compare_goal_sets, baseline_seconds, task_evaluator, planar_position_jacobian
from scripts.planar3R_iris_regions import ReliablePolytope
from scripts.planar3R_ssm_envelopes import sample_hull

try:
    from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def box(dimension, lower=-3., upper=3.):
    return ReliablePolytope(np.vstack((np.eye(dimension), -np.eye(dimension))),
                            np.r_[np.full(dimension, upper), np.full(dimension, -lower)])


@unittest.skipUnless(HAS_DRAKE, "Drake required")
class GoalSetTests(unittest.TestCase):
    def setUp(self):
        self.regions = [box(2)]
        self.settings = dict(max_rounded_paths=20, max_rounding_trials=30, solver_backend="mosek")
        self.task = dict(id="x=1", dimension=1, residual=lambda q: np.array([q[0]-1]),
                         jacobian=lambda q: np.array([[1., 0.]]))
        self.envelopes = [SimpleNamespace(branch=0), SimpleNamespace(branch=1)]
        self.targets = [dict(polytope=sample_hull([[2., 0.]]), region=0, envelope=0),
                        dict(polytope=sample_hull([[1., -.2], [1., .2]]), region=0, envelope=1)]

    def run_comparisons(self, task=None, **kwargs):
        return compare_goal_sets(self.regions, [], np.zeros(2), self.targets, self.envelopes,
                                 [1.], self.settings, self.settings, task or self.task,
                                 lambda p, e, r: p.copy(), lambda p: (True, len(p)), **kwargs)

    def test_union_selects_near_convex_set_and_runs_two_stages(self):
        reports = self.run_comparisons(union_hulls=True)
        result = reports["union_hull_baseline"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["selected_hull_target"], 1)
        self.assertEqual(result["stage1_region_route"], [0])
        self.assertEqual(result["stage1_route_spec"]["regions"], [0])
        self.assertEqual(result["selected_ssm_branch"], 1)
        self.assertEqual(result["gcs_solve_calls"], 2)
        self.assertAlmostEqual(result["cost"], 1., places=6)
        self.assertTrue(result["graph_membership_valid"])
        np.testing.assert_allclose(result["endpoint"], [1., 0.], atol=1e-5)

    def test_native_nonlinear_goal_solves_true_curve_outside_hull_targets(self):
        task = dict(id="circle", dimension=1,
                    residual=lambda q: np.array([(q[0]-2)**2+q[1]**2-1]),
                    jacobian=lambda q: np.array([[2*(q[0]-2), 2*q[1]]]))
        result = self.run_comparisons(task, nonlinear_goal=True)["nonlinear_goal_baseline"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], 1)
        self.assertEqual(result["solver_name"], "SNOPT")
        self.assertLess(result["task_error"], 1e-6)
        self.assertAlmostEqual(result["cost"], 1., places=5)
        self.assertEqual(result["goal_surrogate"], "all_connected_free_regions_without_task_constraint")

    def test_nonlinear_full_pose_on_real_seven_joint_model(self):
        from scripts import spatial7R_model as model
        q_goal = model.GOAL_REFERENCE_CONFIGURATION
        goal = model.pose(q_goal)
        center = q_goal.copy()
        region = ReliablePolytope(np.vstack((np.eye(7), -np.eye(7))), np.r_[center+.2, -center+.2])
        task = dict(id=dict(type="SE3", goal=goal.tolist()), dimension=6,
                    residual=lambda q: model.pose_residual(q, goal),
                    jacobian=lambda q: model.residual_jacobian(q, goal))
        result = compare_goal_sets([region], [], center+.01, [], [], [.37], self.settings, self.settings,
                                   task, None, lambda p: model.validate_path(p, []),
                                   nonlinear_goal=True)["nonlinear_goal_baseline"]
        self.assertTrue(result["success"], result)
        self.assertLess(result["task_error"], 1e-6)
        self.assertTrue(result["cost_valid"])

    def test_autodiff_uses_full_pose_and_planar_analytic_jacobians(self):
        from scripts import spatial7R_model as model
        q = model.START_CONFIGURATION
        goal = model.pose(model.GOAL_REFERENCE_CONFIGURATION)
        task = dict(residual=lambda q: model.pose_residual(q, goal),
                    jacobian=lambda q: model.residual_jacobian(q, goal))
        np.testing.assert_allclose(ExtractGradient(task_evaluator(task)(InitializeAutoDiff(q))),
                                   model.residual_jacobian(q, goal), atol=1e-12)
        from scripts.planar3R_collision import link_points, L
        q = np.array([.3, -.2, .4])
        numeric = np.column_stack([(link_points(q+np.eye(3)[i]*1e-6)[-1]
                                    -link_points(q-np.eye(3)[i]*1e-6)[-1])/2e-6 for i in range(3)])
        np.testing.assert_allclose(planar_position_jacobian(q, np.asarray(L)), numeric, atol=1e-8)

    def test_nonlinear_position_tasks_on_real_planar_and_spatial_models(self):
        from scripts import spatial4R_model as spatial
        from scripts.planar3R_collision import link_points, L
        for goal_q, position, jacobian in (
                (np.array([.6, -.3, .7]), lambda q: link_points(q)[-1],
                 lambda q: planar_position_jacobian(q, np.asarray(L))),
                (spatial.GOAL_REFERENCE_CONFIGURATION.copy(), spatial.position, spatial.jacobian)):
            dimension = len(goal_q)
            region = ReliablePolytope(np.vstack((np.eye(dimension), -np.eye(dimension))),
                                      np.r_[goal_q+.15, -goal_q+.15])
            goal = position(goal_q)
            task = dict(id=dict(position=goal.tolist()), dimension=len(goal),
                        residual=lambda q: position(q)-goal, jacobian=jacobian)
            result = compare_goal_sets([region], [], goal_q+.01, [], [], [.2],
                                       self.settings, self.settings, task, None,
                                       lambda p: (True, len(p)), nonlinear_goal=True)["nonlinear_goal_baseline"]
            self.assertTrue(result["success"], result)
            self.assertLess(result["task_error"], 1e-6)

    def test_nonlinear_failure_and_unsupported_exact_mode_are_explicit(self):
        task = dict(id="impossible", dimension=1, residual=lambda q: np.ones(1),
                    jacobian=lambda q: np.zeros((1, 2)))
        result = self.run_comparisons(task, nonlinear_goal=True)["nonlinear_goal_baseline"]
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "solver_failed")
        result = self.run_comparisons(nonlinear_goal=True, rounding=False)["nonlinear_goal_baseline"]
        self.assertEqual(result["status"], "unsupported_solve_mode")
        self.assertEqual(result["gcs_solve_calls"], 0)

    def test_timings_exclude_all_comparisons_and_windings_are_restored(self):
        outputs = self.run_comparisons(union_hulls=True, offsets=[2*np.pi, -4*np.pi])
        result = outputs["union_hull_baseline"]
        self.assertTrue(result["success"], result)
        np.testing.assert_allclose(result["waypoints"][0], [2*np.pi, -4*np.pi], atol=1e-8)
        outputs["sampled_baseline"] = dict(seconds=3.)
        self.assertAlmostEqual(baseline_seconds(outputs), 3+result["seconds"])

    def test_union_hulls_preserves_periodic_route_and_physical_cost(self):
        from scripts.planar3R_iris_regions import polytope_adjacency
        identity = np.vstack((np.eye(2), -np.eye(2)))
        regions = [ReliablePolytope(identity, np.array([np.pi, .2, -2.8, .2])),
                   ReliablePolytope(identity, np.array([-2.8, .2, np.pi, .2]))]
        edges = polytope_adjacency(regions, periodic=True)
        task = dict(id="periodic_sine", dimension=1,
                    residual=lambda q: np.array([np.sin(q[0])-np.sin(-3.)]),
                    jacobian=lambda q: np.array([[np.cos(q[0]), 0.]]))
        targets = [dict(polytope=sample_hull([[-3., 0.]]), region=1, envelope=0)]
        result = compare_goal_sets(regions, edges, np.array([3., 0.]), targets, self.envelopes,
                                   [1., 1.], self.settings, self.settings, task,
                                   lambda p, e, r: p.copy(), lambda p: (True, len(p)),
                                   union_hulls=True)["union_hull_baseline"]
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["cost"], 2*np.pi-6, places=6)
        self.assertGreater(result["endpoint"][0], np.pi)


if __name__ == "__main__":
    unittest.main()
