import unittest
from unittest.mock import patch

import numpy as np

from scripts.gcs_representative_pruning import prune_representative_paths
from scripts.planar3R_iris_regions import ReliablePolytope


def box(x, y):
    return ReliablePolytope(np.vstack((np.eye(2), -np.eye(2))),
                            np.array([x+1, y+1, -x, -y]))


class RepresentativePruningTests(unittest.TestCase):
    def test_target_mapping_and_directed_path_survive(self):
        regions = [box(9, 9), box(0, 0), box(1, 0), box(2, 0)]
        edges = [(1, 2), (2, 1), (2, 3), (3, 2)]
        targets = [dict(polytope=regions[3], region=3, regions=[3], envelope=8)]
        rr, ee, tt, stats = prune_representative_paths(
            regions, edges, [.1, .5], targets, [1]*4, alpha=1, trials=5)
        self.assertEqual(stats["retained_original_region_indices"], [1, 2, 3])
        self.assertEqual(ee, [(0, 1), (1, 2)])
        self.assertEqual(tt[0]["region"], 2)
        self.assertEqual(tt[0]["envelope"], 8)
        self.assertIs(rr[2], regions[3])

    def test_each_target_gets_a_path_even_if_one_is_expensive(self):
        regions = [box(0, 0), box(1, 0), box(0, 1)]
        targets = [dict(polytope=regions[i], region=i) for i in (1, 2)]
        _, ee, tt, stats = prune_representative_paths(
            regions, [(0, 1), (0, 2)], [.5, .5], targets, [1, 1, 100], trials=2)
        self.assertEqual(set(ee), {(0, 1), (0, 2)})
        self.assertEqual(len(tt), 2)
        self.assertTrue(all(t["reachable"] for t in stats["targets"]))

    def test_unreachable_target_stays_indexed_and_is_reported(self):
        regions = [box(0, 0), box(5, 0)]
        _, ee, tt, stats = prune_representative_paths(
            regions, [], [.5, .5], [dict(polytope=regions[1], region=1)], [1, 1], trials=1)
        self.assertFalse(stats["targets"][0]["reachable"])
        self.assertEqual(len(tt), 1)
        self.assertEqual(ee, [])

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            prune_representative_paths([], [], [], [], [], alpha=.9)

    def test_feasible_point_target_when_radius_lp_fails(self):
        region = box(0, 0)
        point = ReliablePolytope(np.vstack((np.eye(2), -np.eye(2))), np.array([.5, .5, -.5, -.5]))
        with patch("scripts.gcs_representative_pruning.interior_point",
                   side_effect=[np.array([.5, .5]), None]):
            _, _, _, stats = prune_representative_paths(
                [region], [], [.1, .1], [dict(polytope=point, region=0)], [1], trials=1)
        self.assertTrue(stats["targets"][0]["reachable"])
        self.assertEqual(stats["targets"][0]["representative_method"], "feasibility_lp")


if __name__ == "__main__":
    unittest.main()
