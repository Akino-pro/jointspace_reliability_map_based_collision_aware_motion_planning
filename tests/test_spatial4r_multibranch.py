"""Four-branch fixtures test aggregation; a real two-branch DH target tests tracing."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import spatial4R_model as model
from scripts.spatial4R_two_stage_gcs_planner import reliability_regions_for_branches
from scripts.planar3R_bbox_reliability_partition import partition_multiple_branches
from scripts.planar3R_iris_regions import ReliablePolytope
from scripts.planar3R_ssm_envelopes import build_envelopes, intersect_envelopes


class SpatialMultiBranchTests(unittest.TestCase):
    def test_search_keeps_four_distinct_branches_and_reports_budget(self):
        starts = [np.array([-.9+.5*i, 0., 0., 0.]) for i in range(4)]
        # Controlled search fixture; these curves are not claimed to be physical SSMs.
        with patch.object(model, "least_squares", side_effect=[SimpleNamespace(x=q) for q in starts]) as ik, \
                patch.object(model, "position", return_value=np.zeros(3)), \
                patch.object(model, "refine_ik", side_effect=lambda q, goal: q), \
                patch.object(model, "trace_ssm", side_effect=lambda q, goal: np.array([q, q+[0, 0, 0, .2], q])):
            branches, report = model.compute_ssms(np.zeros(3), attempts=10)
        self.assertEqual(len(branches), 4)
        self.assertEqual(ik.call_count, 4)
        self.assertEqual(report["branch_limit"], 4)
        self.assertEqual(report["termination_reason"], "branch_limit_reached")
        self.assertFalse(report["all_branches_certified"])

    def test_segment_deduplication_handles_phase_periodicity_and_nearby_branches(self):
        line = np.array([[-.1, 0, 0, 0], [.1, 0, 0, 0]])
        self.assertTrue(model.is_duplicate_ssm_start([0, 0, 0, 0], [line]))
        self.assertTrue(model.is_duplicate_ssm_start([2*np.pi, 0, 0, 0], [line]))
        self.assertFalse(model.is_duplicate_ssm_start([0, .02, 0, 0], [line]))
        seam = np.array([[3.13, 0, 0, 0], [-3.13, 0, 0, 0]])
        self.assertTrue(model.is_duplicate_ssm_start([np.pi, 0, 0, 0], [seam]))
        self.assertFalse(model.is_duplicate_ssm_start([0, 0, 0, 0], [seam]))

    def test_four_branch_refinement_uses_max_and_only_convex_equal_cost_merges(self):
        bounds = [np.tile([1., 2.], (4, 1)) for _ in range(4)]
        for i, bound in enumerate(bounds):
            bound[i] = [-.4, .4]
        branches = [np.array([b[:, 0], b[:, 1]]) for b in bounds]
        raw, merged = reliability_regions_for_branches(branches)
        self.assertEqual(len(raw), 625)
        self.assertLess(len(merged), len(raw))
        self.assertAlmostEqual(sum(np.prod(c["upper"]-c["lower"]) for c in raw), (2*np.pi)**4)
        self.assertAlmostEqual(sum(np.prod(c["upper"]-c["lower"]) for c in merged), (2*np.pi)**4)
        lookup = model.reliability_lookup()
        lower, upper = np.array([c["lower"] for c in merged]), np.array([c["upper"] for c in merged])
        for cell in raw:
            point = (cell["lower"]+cell["upper"])/2
            values = [lookup[frozenset(j+1 for j in range(4) if b[j, 0] < point[j] < b[j, 1])]
                      for b in bounds]
            self.assertEqual(cell["branch_reliabilities"], tuple(values))
            self.assertEqual(cell["reliability"], max(values))
            matches = np.flatnonzero(np.all(point > lower, axis=1) & np.all(point < upper, axis=1))
            # Original cell centers may lie on a removed face of a merged box,
            # but remain strictly inside the unique box containing that cell.
            self.assertEqual(len(matches), 1)
            containing = merged[matches[0]]
            self.assertEqual(containing["reliability"], cell["reliability"])
            self.assertTrue(np.all(containing["lower"] <= cell["lower"]))
            self.assertTrue(np.all(containing["upper"] >= cell["upper"]))
        origin_cell = next(c for c in raw if np.all(c["lower"] < 0) and np.all(c["upper"] > 0))
        self.assertEqual(origin_cell["branch_F"], tuple(frozenset({i+1}) for i in range(4)))
        self.assertLess(origin_cell["reliability"], lookup[frozenset({1, 2, 3, 4})])
        for i in range(4):
            point = np.full(4, 2.5)
            point[i] = 0
            cell = next(c for c in raw if np.all(point > c["lower"]) and np.all(point < c["upper"]))
            self.assertEqual(cell["winning_branch"], i)
        def signature(cells):
            return {(*c["lower"], *c["upper"]): c["reliability"] for c in cells}
        self.assertEqual(signature(raw), signature(partition_multiple_branches(bounds[::-1], lookup)))

    def test_all_four_branches_sampled_and_intersected_without_cross_branch_hulls(self):
        # Four disjoint lines of unequal length make missing branches/density errors visible.
        branches = [np.array([[-1., -.8+.4*i, 0., 0.], [-1.+.2*(i+1), -.8+.4*i, 0., 0.]])
                    for i in range(4)]
        report = {}
        envelopes = build_envelopes(branches, arc_spacing=.044, sampling_report=report)
        full_chart = ReliablePolytope(np.vstack((np.eye(4), -np.eye(4))), np.full(8, np.pi))
        targets = intersect_envelopes(envelopes, [full_chart])
        self.assertEqual({envelopes[t["envelope"]].branch for t in targets}, {0, 1, 2, 3})
        self.assertEqual(len(report["ssm_branch_sample_counts"]), 4)
        for i, length in enumerate([.2, .4, .6, .8]):
            self.assertAlmostEqual(report["ssm_branch_arc_lengths"][i], length)
            self.assertEqual(report["ssm_branch_sample_counts"][i], int(np.ceil(length/.044))+1)
            for e in (e for e in envelopes if e.branch == i):
                np.testing.assert_allclose(e.samples[:, 1], -.8+.4*i)
        self.assertAlmostEqual(report["ssm_total_arc_length"], 2.)

    def test_real_ftwmax_two_branch_target_has_distinct_closed_ik_traces(self):
        q = np.array([-2.0291132605617523, 2.3781117581837794, 1.3056985099517648, 1.4303215263847688])
        goal = model.position(q)
        branches, report = model.compute_ssms(goal, reference=q)
        self.assertEqual(len(branches), 2)
        self.assertEqual(report["ik_attempts"], model.MAX_SSM_IK_ATTEMPTS)
        self.assertEqual(report["termination_reason"], "ik_attempt_budget_exhausted")
        self.assertFalse(report["all_branches_certified"])
        for i, branch in enumerate(branches):
            self.assertLess(np.max([np.linalg.norm(model.position(p)-goal) for p in branch]), 1e-8)
            np.testing.assert_allclose(model.wrap(branch[0]-branch[-1]), 0, atol=1e-12)
            self.assertFalse(model.is_duplicate_ssm_start(branch[0], branches[:i]))

    def test_invalid_search_limits_are_rejected(self):
        for options in (dict(max_branches=5), dict(max_branches=0), dict(attempts=0)):
            with self.assertRaises(ValueError):
                model.compute_ssms(np.zeros(3), **options)


if __name__ == "__main__":
    unittest.main()
