"""Dropping result history must still evaluate every candidate and keep the best."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from gcs.base import BaseGCS


def result(cost, success=True):
    answer = Mock()
    answer.is_success.return_value = success
    answer.get_optimal_cost.return_value = cost
    answer.get_solver_details.return_value = SimpleNamespace(optimizer_time=.01)
    return answer


class RoundingHistoryTests(unittest.TestCase):
    def run_candidates(self, retain, candidates):
        planner = BaseGCS.__new__(BaseGCS)
        planner.options = SimpleNamespace(max_rounded_paths=3)
        planner.source, planner.target = object(), object()
        planner.rounding_fn = []
        planner.retain_rounded_results = retain
        planner.gcs = Mock()
        planner.gcs.SolveShortestPath.return_value = result(.5)
        planner.gcs.SamplePaths.return_value = [["early-best"], ["failed"], ["later-worse"]]
        planner.gcs.SolveConvexRestriction.side_effect = candidates
        answer = planner.solveGCS(True, True, False)
        self.assertEqual(planner.gcs.SolveConvexRestriction.call_count, 3)
        return answer

    def test_bounded_history_keeps_early_best_and_exact_success_count(self):
        best = result(1.)
        path, answer, stats = self.run_candidates(False, [best, result(0., False), result(3.)])
        self.assertEqual(path, ["early-best"])
        self.assertIs(answer, best)
        self.assertEqual(stats["rounded_cost"], 1.)
        self.assertEqual(stats["rounded_path_count"], 3)
        self.assertEqual(stats["successful_rounded_path_count"], 2)
        self.assertEqual(stats["rounded_results"], [])

    def test_history_mode_keeps_all_results_with_same_selection(self):
        best = result(1.)
        candidates = [best, result(0., False), result(3.)]
        path, answer, stats = self.run_candidates(True, candidates)
        self.assertIs(answer, best)
        self.assertEqual(path, ["early-best"])
        self.assertEqual(stats["rounded_results"], candidates)
        self.assertEqual(stats["successful_rounded_path_count"], 2)

    def test_no_feasible_candidate_does_not_create_a_best_result(self):
        path, answer, stats = self.run_candidates(False, [result(0., False) for _ in range(3)])
        self.assertIsNone(path)
        self.assertIsNone(answer)
        self.assertEqual(stats["successful_rounded_path_count"], 0)


if __name__ == "__main__":
    unittest.main()
