import itertools
import unittest
from unittest.mock import patch

import numpy as np

from planar3R_bbox_reliability_partition import (
    build_reliability_lookup, partition_multiple_branches, partition_single_branch,
    merge_equal_reliability_boxes,
)
from planar3R_gcs_planner import reliability_region_sets_for_workspace_point


class MultiBranchReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.bounds = [np.tile([-2.0, 0.0], (3, 1)), np.tile([-1.0, 1.0], (3, 1))]
        self.lookup = build_reliability_lookup(0.7, 0.5, 0.3)

    def test_single_branch_preserves_existing_partition(self):
        expected = partition_single_branch(self.bounds[0], self.lookup)
        actual = partition_multiple_branches(self.bounds[:1], self.lookup)
        self.assertEqual(len(actual), 27)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(before["lower"], after["lower"])
            np.testing.assert_array_equal(before["upper"], after["upper"])
            self.assertEqual(before["reliability"], after["reliability"])

    def test_common_refinement_covers_chart_with_pointwise_maximum(self):
        cells = partition_multiple_branches(self.bounds, self.lookup)
        self.assertEqual(len(cells), 125)
        volume = sum(np.prod(cell["upper"] - cell["lower"]) for cell in cells)
        self.assertAlmostEqual(volume, (2 * np.pi) ** 3)
        for cell in cells:
            point = (cell["lower"] + cell["upper"]) / 2
            values = []
            for bounds in self.bounds:
                active = frozenset(i + 1 for i in range(3)
                                   if bounds[i, 0] < point[i] < bounds[i, 1])
                values.append(self.lookup[active])
            self.assertEqual(cell["branch_reliabilities"], tuple(values))
            self.assertEqual(cell["reliability"], max(values))
            matches = [other for other in cells if np.all(point > other["lower"])
                       and np.all(point < other["upper"])]
            self.assertEqual(len(matches), 1)

    def test_different_branch_joint_conditions_are_not_unioned(self):
        point = np.array([-1.5, 0.5, 2.0])
        cell = next(cell for cell in partition_multiple_branches(self.bounds, self.lookup)
                    if np.all(point > cell["lower"]) and np.all(point < cell["upper"]))
        self.assertEqual(cell["branch_F"], (frozenset({1}), frozenset({2})))
        self.assertEqual(cell["reliability"], max(self.lookup[frozenset({1})],
                                                 self.lookup[frozenset({2})]))
        self.assertLess(cell["reliability"], self.lookup[frozenset({1, 2})])

    def test_branch_order_does_not_change_geometry_or_reliability(self):
        def signature(cells):
            return {(*cell["lower"], *cell["upper"]): cell["reliability"] for cell in cells}
        self.assertEqual(signature(partition_multiple_branches(self.bounds, self.lookup)),
                         signature(partition_multiple_branches(self.bounds[::-1], self.lookup)))

    def test_duplicate_bounds_do_not_increase_cell_count(self):
        cells = partition_multiple_branches([self.bounds[0], self.bounds[0]], self.lookup)
        self.assertEqual(len(cells), 27)

    def test_planner_receives_one_map_for_both_detected_branches(self):
        branches = [np.array(list(itertools.product(*bounds))) for bounds in self.bounds]
        with patch("planar3R_gcs_planner.compute_ssms_for_target", return_value=branches):
            maps = reliability_region_sets_for_workspace_point(0.5, 0.5)
        self.assertEqual(len(maps), 1)
        self.assertEqual(maps[0]["branch_count"], 2)
        self.assertEqual(maps[0]["reliability_rule"], "maximum_across_branches")
        self.assertEqual(maps[0]["unmerged_partition_count"], 125)
        self.assertLess(len(maps[0]["regions"]), 125)

    def test_merged_boxes_preserve_every_original_cell_and_reliability(self):
        raw = partition_multiple_branches(self.bounds, self.lookup)
        merged = merge_equal_reliability_boxes(raw)
        self.assertLess(len(merged), len(raw))
        self.assertAlmostEqual(sum(np.prod(c["upper"] - c["lower"]) for c in merged),
                               (2 * np.pi) ** 3)
        for cell in raw:
            point = (cell["lower"] + cell["upper"]) / 2
            matches = [c for c in merged if np.all(c["lower"] <= cell["lower"])
                       and np.all(c["upper"] >= cell["upper"])]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["reliability"], cell["reliability"])
        self.assertEqual(len(merge_equal_reliability_boxes(merged)), len(merged))

    def test_merge_keeps_gaps_nonconvex_unions_and_different_costs(self):
        def box(lower, upper, reliability=0.5):
            return dict(lower=np.array(lower, dtype=float), upper=np.array(upper, dtype=float),
                        reliability=reliability)
        base = box([0, 0, 0], [1, 1, 1])
        for other in [box([1, 0, 0], [2, 1, 1], 0.6),
                      box([1.000000000001, 0, 0], [2, 1, 1]),
                      box([1, 0, 0], [2, 0.5, 1]),
                      box([1, 1, 0], [2, 2, 1])]:
            self.assertEqual(len(merge_equal_reliability_boxes([base, other])), 2)

    def test_full_face_merges_repeat_across_axes(self):
        cells = [dict(lower=np.array([x, y, 0.0]), upper=np.array([x+1, y+1, 1.0]),
                      reliability=0.5) for x in (0, 1) for y in (0, 1)]
        merged = merge_equal_reliability_boxes(cells)
        self.assertEqual(len(merged), 1)
        np.testing.assert_array_equal(merged[0]["lower"], [0, 0, 0])
        np.testing.assert_array_equal(merged[0]["upper"], [2, 2, 1])
        self.assertEqual(len(merged[0]["source_cells"]), 4)


if __name__ == "__main__":
    unittest.main()
