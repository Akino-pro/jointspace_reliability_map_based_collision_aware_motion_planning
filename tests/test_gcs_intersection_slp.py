import unittest
import tempfile
import itertools
from pathlib import Path

import numpy as np

from scripts.gcs_intersection_slp import build_representative, collective_cost, erase_region_loops, optimize_representatives, prune_intersection_slp
from scripts.planar3R_iris_regions import ReliablePolytope


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return ReliablePolytope(np.vstack((np.eye(len(lo)), -np.eye(len(lo)))), np.r_[hi, -lo])


class IntersectionSlpTests(unittest.TestCase):
    def test_representatives_lie_in_both_regions_and_edges_have_shared_region(self):
        regions = [box([0, 0], [2, 1]), box([1, 0], [3, 1]), box([2, 0], [4, 1])]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        pairs, links, points, stats = build_representative(regions, edges, [1, 1, 1], max_iterations=30)
        self.assertEqual(pairs.tolist(), [[0, 1], [1, 2]])
        self.assertEqual(links.tolist(), [[0, 1, 1]])
        for pair, point in zip(pairs, points):
            self.assertTrue(all(regions[i].contains(point, tol=2e-7) for i in pair))
        self.assertLess(stats["optimization"]["final_cost"], stats["optimization"]["initial_cost"])
        self.assertLess(np.linalg.norm(points[0]-points[1]), 1e-3)

    def test_pruned_routes_remap_to_valid_original_directed_region_edges(self):
        regions = [box([9, 9], [10, 10]), box([0, 0], [1, 1]),
                   box([1, 0], [2, 1]), box([2, 0], [3, 1])]
        edges = [(1, 2), (2, 1), (2, 3), (3, 2)]
        target = dict(polytope=box([2.8, .4], [2.9, .6]), region=3, envelope=4)
        reduced, retained, targets, stats = prune_intersection_slp(
            regions, edges, [.1, .5], [target], [1]*4, trials=10, max_iterations=10)
        self.assertEqual(stats["retained_original_region_indices"], [1, 2, 3])
        self.assertEqual(retained, [(0, 1), (1, 2)])
        self.assertEqual(targets[0]["region"], 2)
        self.assertEqual(targets[0]["envelope"], 4)
        self.assertTrue(stats["targets"][0]["reachable"])

    def test_direct_path_without_any_intersections(self):
        region = box([0, 0], [1, 1])
        _, edges, _, stats = prune_intersection_slp([region], [], [.1, .1],
            [dict(polytope=box([.8, .8], [.9, .9]), region=0)], [1], trials=2)
        self.assertEqual(edges, [])
        self.assertTrue(stats["targets"][0]["reachable"])

    def test_region_loop_erasure(self):
        self.assertEqual(erase_region_loops([0, 0, 1, 2, 1, 3]), [0, 1, 3])

    def test_dense_graph_is_rejected_before_intersection_allocation(self):
        regions = [box([0, 0], [1, 1])]*318
        edges = list(itertools.combinations(range(318), 2))
        with self.assertRaisesRegex(ValueError, "bounded-memory"):
            build_representative(regions, edges, [1]*318)

    def test_offline_cache_reuses_geometry_but_rejects_different_weights(self):
        regions = [box([0, 0], [2, 1]), box([1, 0], [3, 1])]
        edges = [(0, 1), (1, 0)]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/"representative.npz"
            first = build_representative(regions, edges, [1, 1], cache, max_iterations=0)
            second = build_representative(regions, edges, [1, 1], cache, max_iterations=0)
            np.testing.assert_array_equal(first[2], second[2])
            self.assertTrue(second[3]["cache_hit"])
            with self.assertRaises(ValueError):
                build_representative(regions, edges, [1, 2], cache, max_iterations=0)


if __name__ == "__main__":
    unittest.main()
