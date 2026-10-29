import unittest

from scripts.gcs_budget import allocate_target_budgets


class BudgetAllocationTests(unittest.TestCase):
    def test_quotas_conserve_both_totals_and_cover_every_target(self):
        for count, paths, trials in ((20, 200, 300), (3, 8, 13), (7, 7, 9), (1, 200, 300)):
            quotas = allocate_target_budgets(count, paths, trials)
            self.assertEqual(len(quotas), count)
            self.assertEqual(sum(q["max_rounded_paths"] for q in quotas), paths)
            self.assertEqual(sum(q["max_rounding_trials"] for q in quotas), trials)
            self.assertTrue(all(1 <= q["max_rounded_paths"] <= q["max_rounding_trials"] for q in quotas))
            self.assertLessEqual(max(q["max_rounded_paths"] for q in quotas)
                                 - min(q["max_rounded_paths"] for q in quotas), 1)

    def test_invalid_and_insufficient_budgets_are_rejected(self):
        for values in ((0, 20, 30), (3, 2, 30), (3, 20, 2), (3, 20, 10),
                       (3, None, 30), (3, 20, None), (3, 2.5, 30), (True, 20, 30)):
            with self.assertRaises(ValueError):
                allocate_target_budgets(*values)

    def test_actual_unique_count_is_reported_separately_from_the_cap(self):
        try:
            from pydrake.geometry.optimization import Point
        except ImportError:
            self.skipTest("Drake required")
        import numpy as np
        from scripts.planar3R_iris_regions import ReliablePolytope
        from scripts.planar3R_ssm_envelopes import sample_hull
        from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
        region = ReliablePolytope(np.vstack((np.eye(2), -np.eye(2))), np.ones(4)*2)
        diagnostics = {}
        result = solve_target_sets([region], [], np.zeros(2),
                                   [dict(polytope=sample_hull([[1., 0.]]), region=0)], [1.],
                                   max_rounded_paths=7, max_rounding_trials=11,
                                   solver_backend="mosek", diagnostics=diagnostics)
        self.assertIsNotNone(result)
        self.assertEqual(diagnostics["rounded_path_count"], 1)
        self.assertEqual(diagnostics["successful_rounded_path_count"], 1)
        self.assertEqual(diagnostics["max_rounded_paths"], 7)
        self.assertIsNone(diagnostics["rounding_trials_attempted"])


if __name__ == "__main__":
    unittest.main()
