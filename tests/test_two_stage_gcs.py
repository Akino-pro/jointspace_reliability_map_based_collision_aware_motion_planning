import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.planar3R_collision import link_points
from scripts.planar3R_iris_regions import ReliablePolytope, save_iris_checkpoint
from scripts.planar3R_post_failure_planner import locked_joint_ik
from scripts.planar3R_ssm_envelopes import (
    build_envelopes, intersect_envelopes, project_to_local_ssm, sample_hull,
)
from scripts.planar3R_two_stage_gcs_planner import (
    MixedIntegerSolverUnavailable, plan_two_stage, solve_target_sets,
)

try:
    from pydrake.geometry.optimization import HPolyhedron
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def cube(lower=-np.pi, upper=np.pi):
    return ReliablePolytope(np.vstack((np.eye(3), -np.eye(3))),
                            np.r_[np.full(3, upper), np.full(3, -lower)])


def local_branch():
    reference = np.array([0.4, -0.8, 0.7])
    goal = link_points(reference)[-1]
    samples = []
    for angle in np.linspace(0.65, 0.75, 15):
        solutions, _ = locked_joint_ik(goal, 3, angle)
        samples.append(min(solutions, key=lambda q: np.linalg.norm(q - reference)))
    return goal, np.asarray(samples)


class EnvelopeTests(unittest.TestCase):
    def test_tighter_epsilon_refines_ordered_pieces_without_losing_samples(self):
        t = np.linspace(0, np.pi, 61)
        points = np.c_[np.cos(t), np.sin(t), t * 0]
        coarse = build_envelopes([points], 0.3, samples_per_branch=300)
        fine = build_envelopes([points], 0.02, samples_per_branch=300)
        self.assertGreater(len(fine), len(coarse))
        self.assertTrue(all(e.polyline_distance_bound <= 0.02 for e in fine))
        self.assertTrue(all(any(e.polytope.contains(q) for e in fine) for q in points))
        for first, second in zip(fine[:-1], fine[1:]):
            np.testing.assert_allclose(first.samples[-1], second.samples[0])

    def test_periodic_seam_is_split_before_subsampling(self):
        points = np.array([[3.0, 0, 0], [3.1, 0, 0], [-3.1, 0, 0], [-3.0, 0, 0]])
        envelopes = build_envelopes([points], 10, 2)
        self.assertEqual(len(envelopes), 2)
        self.assertTrue(all(not e.polytope.contains([0, 0, 0]) for e in envelopes))

    def test_exact_lower_dimensional_hulls_have_no_artificial_thickness(self):
        for samples, inside, outside in [
            ([[1, 2, 3]], [1, 2, 3], [1, 2, 3.01]),
            ([[0, 0, 0], [1, 0, 0]], [0.5, 0, 0], [0.5, 0.01, 0]),
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.2, 0.2, 0], [0.2, 0.2, 0.01]),
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]),
        ]:
            hull = sample_hull(samples)
            self.assertTrue(hull.contains(inside))
            self.assertFalse(hull.contains(outside))

    def test_intersection_keeps_line_contacts_and_rejects_empty_sets(self):
        envelopes = build_envelopes([np.array([[0, 0, 0], [1, 0, 0]])])
        targets = intersect_envelopes(envelopes, [cube(0, 1), cube(2, 3)])
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0]["polytope"].contains([0.5, 0, 0]))
        self.assertFalse(targets[0]["polytope"].contains([0.5, 0.1, 0]))

    def test_projection_is_continuous_ik_not_nearest_discrete_sample(self):
        goal, samples = local_branch()
        envelope = build_envelopes([samples], 10)[0]
        point = (samples[6] + samples[7]) / 2
        projected = project_to_local_ssm(point, envelope, cube(), goal, [], 0)
        self.assertIsNotNone(projected)
        self.assertLess(np.linalg.norm(link_points(projected)[-1] - goal), 1e-7)
        self.assertGreater(np.min(np.linalg.norm(samples - projected, axis=1)), 1e-4)

    def test_projection_cannot_escape_selected_free_region(self):
        goal, samples = local_branch()
        envelope = build_envelopes([samples], 10)[0]
        self.assertIsNone(project_to_local_ssm(samples[7], envelope, cube(2, 3), goal, [], 0))

    def test_invalid_resolution_is_rejected(self):
        for epsilon in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                build_envelopes([], epsilon)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache.npz"
        save_iris_checkpoint(self.cache, [cube()], {"model": {
            "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})

    def test_unreachable_goal_stops_without_sampling_or_gcs(self):
        with patch('scripts.planar3R_two_stage_gcs_planner.solve_target_sets') as solve:
            result = plan_two_stage([0, 0, 0], [10, 0], [], self.cache)
        self.assertEqual(result["status"], "goal_unreachable")
        self.assertEqual(result["gcs_solve_calls"], 0)
        solve.assert_not_called()

    def test_missing_start_coverage_stops_without_gcs(self):
        save_iris_checkpoint(self.cache, [cube(2, 3)], {"model": {
            "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
        result = plan_two_stage([0, 0, 0], [2, 0], [], self.cache, reliability_source="none")
        self.assertEqual(result["status"], "no_path_in_iris_coverage")
        self.assertEqual(result["gcs_solve_calls"], 0)

    def test_exact_mode_without_solver_stops_before_geometry_and_gcs(self):
        with patch('scripts.planar3R_two_stage_gcs_planner.require_mixed_integer_solver',
                   side_effect=MixedIntegerSolverUnavailable("No enabled MISOCP solver")), \
                patch('scripts.planar3R_two_stage_gcs_planner.solve_target_sets') as solve:
            result = plan_two_stage([0, 0, 0], [2, 0], [], self.cache, rounding=False)
        self.assertEqual(result["status"], "mixed_integer_solver_unavailable")
        self.assertEqual(result["gcs_solve_calls"], 0)
        solve.assert_not_called()

    def run_mocked(self, failure=None, candidate_costs=(1.0,), **planner_kwargs):
        goal, samples = local_branch()
        calls = []

        def solve(regions, edges, start, targets, weights, rounding,
                  candidate_region_routes=None, **solve_settings):
            calls.append((regions, edges, start.copy(), targets, candidate_region_routes,
                          dict(solve_settings)))
            if len(calls) <= len(candidate_costs):
                candidate_cost = candidate_costs[len(calls) - 1]
                if failure == "stage1" or candidate_cost is None:
                    return None
                return {"waypoints": np.array([start, samples.mean(axis=0)]),
                        "target": 0, "cost": candidate_cost, "region_route": [0]}
            goal_q = targets[0]["polytope"].b[:3]
            # Point hull SVD can choose an arbitrary orthogonal frame: recover
            # the singleton from its first three equality normals.
            goal_q = np.linalg.solve(targets[0]["polytope"].A[:3], goal_q)
            return {"waypoints": np.array([start, goal_q]), "target": 0,
                    "cost": float(np.linalg.norm(goal_q - start)), "solver_cost": 0.1}

        def make_targets(envelopes, regions):
            target = intersect_envelopes(envelopes, regions)[0]
            return [dict(target) for _ in candidate_costs]

        with patch('scripts.planar3R_two_point_ssm_views.compute_ssms_for_target', return_value=[samples]), \
                patch('scripts.planar3R_two_stage_gcs_planner.intersect_envelopes', side_effect=make_targets), \
                patch('scripts.planar3R_two_stage_gcs_planner.solve_target_sets', side_effect=solve):
            result = plan_two_stage([0.3, -0.9, 0.6], goal, [], self.cache,
                                    geometric_epsilon=10, reliability_source="none", **planner_kwargs)
        return result, calls

    def test_total_budget_is_conserved_and_stage_two_remains_independent(self):
        result, calls = self.run_mocked(candidate_costs=(4., 3., 2., 1.),
                                        stage1_total_paths=7, stage1_total_trials=11,
                                        stage2_max_rounded_paths=200, stage2_max_rounding_trials=300)
        self.assertTrue(result["success"], result)
        self.assertEqual([call[-1]["max_rounded_paths"] for call in calls[:-1]], [2, 2, 2, 1])
        self.assertEqual([call[-1]["max_rounding_trials"] for call in calls[:-1]], [3, 3, 3, 2])
        self.assertEqual(calls[-1][-1]["max_rounded_paths"], 200)
        self.assertEqual(calls[-1][-1]["max_rounding_trials"], 300)
        self.assertEqual(result["stage1_budget"]["total_max_rounded_paths"], 7)
        self.assertEqual(result["stage1_budget_mode"], "shared_total")

    def test_insufficient_total_budget_does_not_drop_targets_or_call_gcs(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.run_mocked(candidate_costs=(3., 2., 1.), stage1_total_paths=2, stage1_total_trials=4)

    def test_shared_budget_requires_both_caps_and_rounding(self):
        for values in (dict(stage1_total_paths=20), dict(stage1_total_trials=30),
                       dict(stage1_total_paths=20, stage1_total_trials=30, rounding=False)):
            with self.assertRaises(ValueError):
                plan_two_stage([0, 0, 0], [1, 1], [], self.cache, **values)

    def test_joint_comparison_receives_the_same_total_budget(self):
        with patch('scripts.planar3R_two_stage_gcs_planner.compare_goal_sets', return_value={}) as compare:
            result, _ = self.run_mocked(candidate_costs=(3., 2., 1.),
                                        stage1_total_paths=7, stage1_total_trials=11,
                                        compare_union_hulls=True, stage2_max_rounded_paths=200,
                                        stage2_max_rounding_trials=300)
        self.assertTrue(result["success"], result)
        settings, second_settings = compare.call_args.args[6:8]
        self.assertEqual(settings["max_rounded_paths"], 7)
        self.assertEqual(settings["max_rounding_trials"], 11)
        self.assertEqual(second_settings["max_rounded_paths"], 200)
        self.assertEqual(second_settings["max_rounding_trials"], 300)

    def test_joint_only_skips_all_primary_solves(self):
        comparison = dict(success=True, status="success", cost=.4, seconds=0., gcs_solve_calls=2)
        with patch('scripts.planar3R_two_stage_gcs_planner.compare_goal_sets',
                   return_value={"union_hull_baseline": comparison}) as compare:
            result, calls = self.run_mocked(union_hulls_only=True)
        self.assertTrue(result["success"], result)
        self.assertFalse(result["primary_method_executed"])
        self.assertEqual(calls, [])
        self.assertTrue(compare.call_args.kwargs["union_hulls"])

    def test_joint_only_rejects_other_baselines(self):
        for flags in (dict(compare_sampled=True), dict(compare_nonlinear_goal=True)):
            with self.assertRaisesRegex(ValueError, "excludes"):
                plan_two_stage([0, 0, 0], [1, 1], [], self.cache, union_hulls_only=True, **flags)

    def test_each_intersection_gets_a_separate_solve_and_best_is_selected(self):
        result, calls = self.run_mocked(candidate_costs=(3.0, 1.0, 2.0))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], 4)
        self.assertEqual(result["stage1_gcs_solve_calls"], 3)
        self.assertEqual(result["stage2_gcs_solve_calls"], 1)
        self.assertEqual(result["selected_target"], 1)
        self.assertEqual(result["stage1_cost"], 1.0)
        self.assertEqual([x["cost"] for x in result["stage1_candidates"]], [3.0, 1.0, 2.0])
        self.assertTrue(all(len(call[3]) == 1 for call in calls))
        for call in calls:
            np.testing.assert_allclose(call[2], calls[0][2])
        self.assertEqual(result["projection_connection_baseline"]["stage1_cost"], 1.0)
        self.assertTrue(all(call[4] is None for call in calls[:-1]))
        self.assertEqual(calls[-1][4], [[0]])
        for call in calls[:-1]:
            self.assertEqual(call[5]["max_rounded_paths"], 100)
            self.assertEqual(call[5]["max_rounding_trials"], 150)
        self.assertEqual(calls[-1][5]["max_rounded_paths"], 100)
        self.assertEqual(calls[-1][5]["max_rounding_trials"], 150)

    def test_failed_target_does_not_hide_other_target_results(self):
        result, calls = self.run_mocked(candidate_costs=(None, 2.0, 1.0))
        self.assertTrue(result["success"], result)
        self.assertEqual(len(calls), 4)
        self.assertFalse(result["stage1_candidates"][0]["success"])
        self.assertEqual(result["selected_target"], 2)

    def test_all_target_failures_stop_before_projection_and_final_solve(self):
        result, calls = self.run_mocked(candidate_costs=(None, None, None))
        self.assertEqual(result["status"], "solver_failed")
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["stage2_gcs_solve_calls"], 0)

    def test_two_solves_use_original_start_and_fresh_fixed_goal_graph(self):
        result, calls = self.run_mocked()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], 2)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(len(result["waypoints"]), 2)
        np.testing.assert_allclose(result["waypoints"][-1], result["estimated_goal"])
        self.assertLess(result["endpoint_error"], 1e-7)

    def test_projection_connection_comparison_preserves_stage1_route_and_weighted_cost(self):
        result, calls = self.run_mocked()
        comparison = result["projection_connection_baseline"]
        self.assertTrue(comparison["success"], comparison)
        self.assertTrue(comparison["connector_in_selected_region"])
        self.assertEqual(comparison["gcs_solve_calls"], 0)
        self.assertEqual(len(calls), 2)
        # With reliability_source="none", R=1 and the actual weight is 0.01,
        # not unit weight. The first-stage mock has a known reported cost of 1.
        self.assertAlmostEqual(comparison["connector_weight"], 0.01)
        connection = np.asarray(comparison["waypoints"])
        distance = np.linalg.norm(connection[-1] - connection[-2])
        self.assertGreater(distance, 0)
        self.assertAlmostEqual(comparison["connector_cost"], 0.01 * distance)
        self.assertAlmostEqual(comparison["cost"], 1.0 + 0.01 * distance)
        np.testing.assert_allclose(connection[-1], result["estimated_goal"])
        self.assertEqual(len(connection), 3)
        self.assertEqual(len(result["waypoints"]), 2)
        self.assertAlmostEqual(result["stage2_minus_projection_connection_cost"],
                               result["cost"] - comparison["cost"])

    def test_invalid_connection_is_not_reported_as_feasible(self):
        # Only the diagnostic's collision check fails; stage two stays valid.
        with patch('scripts.planar3R_two_stage_gcs_planner.validate_joint_path',
                   side_effect=[(False, 2), (True, 3)]):
            result, calls = self.run_mocked()
        self.assertTrue(result["success"])
        self.assertFalse(result["projection_connection_baseline"]["success"])
        self.assertNotIn("stage2_minus_projection_connection_cost", result)
        self.assertEqual(len(calls), 2)

    def test_solver_failure_stops_without_second_solve(self):
        result, calls = self.run_mocked("stage1")
        self.assertEqual(result["status"], "solver_failed")
        self.assertEqual(len(calls), 1)

    def test_projection_failure_stops_without_second_solve(self):
        with patch('scripts.planar3R_two_stage_gcs_planner.project_to_local_ssm', return_value=None):
            result, calls = self.run_mocked()
        self.assertEqual(result["status"], "projection_failed")
        self.assertEqual(result["gcs_solve_calls"], 1)
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(HAS_DRAKE, "Requires robotics Drake bindings")
class TwoStageDrakeTests(unittest.TestCase):
    def test_mip_capability_error_is_not_misreported_as_invalid_task(self):
        with patch("pydrake.solvers.ChooseBestSolver", side_effect=ValueError("no conic MIP solver")):
            with self.assertRaises(MixedIntegerSolverUnavailable):
                solve_target_sets([cube()], [], np.zeros(3),
                                  [{"polytope": sample_hull([[1, 0, 0]]), "region": 0}],
                                  [1.0], rounding=False)

    def test_known_route_competes_with_native_rounding_without_appending_motion(self):
        from gcs.linear import LinearGCS
        from pydrake.geometry.optimization import GraphOfConvexSetsOptions

        def expensive_native(planner, **kwargs):
            names = [planner.source.name(), "v0", "v2", planner.target.name()]
            lookup = {(e.u().name(), e.v().name()): e for e in planner.gcs.Edges()}
            active = [lookup[pair] for pair in zip(names[:-1], names[1:])]
            result = planner.gcs.SolveConvexRestriction(active, GraphOfConvexSetsOptions())
            self.assertTrue(result.is_success())
            points = np.asarray([result.GetSolution(e.xv()) for e in active]).T
            return points, {"best_path": active, "best_result": result, "relaxation_cost": 0.2}

        with patch.object(LinearGCS, "SolvePath", expensive_native):
            result = solve_target_sets(
                [cube(-1, 3)] * 3, [(0, 1), (1, 2), (0, 2)], np.zeros(3),
                [{"polytope": sample_hull([[2, 0, 0]]), "region": 2}],
                [1.0, 0.1, 1.0], candidate_region_routes=[[0, 1, 2]])
        self.assertAlmostEqual(result["native_cost"], 2.0, places=5)
        self.assertAlmostEqual(result["cost"], 0.2, places=5)
        self.assertAlmostEqual(result["solver_cost"], 0.2, places=5)
        self.assertEqual(result["solution_source"], "stage1_route_convex_restriction")
        self.assertEqual(result["region_route"], [0, 1, 2])
        self.assertAlmostEqual(result["quality"]["upper_bound"], 0.2, places=5)
        self.assertTrue(result["quality"]["within_tolerance"])
        self.assertEqual(len(result["additional_route_results"]), 1)
        np.testing.assert_allclose(result["waypoints"][-1], [2, 0, 0], atol=1e-6)

    def test_variable_target_cost_has_no_free_sink_motion(self):
        region = cube(-1, 3)
        target = sample_hull([[1, 0, 0], [2, 0, 0]])
        result = solve_target_sets([region], [], np.zeros(3),
                                   [{"polytope": target, "region": 0}], [0.4])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["waypoints"][-1], [1, 0, 0], atol=1e-5)
        self.assertAlmostEqual(result["cost"], 0.4, places=5)
        self.assertAlmostEqual(result["solver_cost"], 0.4, places=5)
        self.assertAlmostEqual(result["quality"]["lower_bound"], 0.4, places=5)
        self.assertTrue(result["quality"]["within_tolerance"])
        self.assertEqual(result["rounded_path_count"], 1)

    def test_full_two_stage_pipeline_with_real_solver_and_local_goal_curve(self):
        goal, samples = local_branch()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.npz"
            save_iris_checkpoint(cache, [cube()], {"model": {
                "link_lengths": [1, 1, 1], "obstacles": [], "link_radius": 0.0}})
            with patch('scripts.planar3R_two_point_ssm_views.compute_ssms_for_target', return_value=[samples]):
                result = plan_two_stage([0.3, -0.9, 0.6], goal, [], cache,
                                        reliability_source="none", geometric_epsilon=0.01)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["gcs_solve_calls"], result["target_set_count"] + 1)
        self.assertEqual(result["stage1_gcs_solve_calls"], result["target_set_count"])
        self.assertEqual(result["stage2_gcs_solve_calls"], 1)
        self.assertLess(result["endpoint_error"], 1e-7)
        self.assertLess(result["fixed_goal_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
