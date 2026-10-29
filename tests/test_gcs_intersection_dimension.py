"""Affine intersection tests must distinguish faces from lower-dimensional contacts."""
import unittest
import numpy as np
from scripts.planar3R_iris_regions import ReliablePolytope
from scripts.gcs_intersection_dimension import filter_intersection_edges

try:
    from pydrake.geometry.optimization import GraphOfConvexSets
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False


def box(lower, upper):
    n = len(lower)
    return ReliablePolytope(np.vstack((np.eye(n), -np.eye(n))), np.r_[upper, -np.array(lower)])


@unittest.skipUnless(HAS_DRAKE, "Drake/MOSEK is required")
class IntersectionDimensionTests(unittest.TestCase):
    def test_seven_dimensional_face_kept_corner_removed_and_overlap_kept(self):
        regions = [box(np.zeros(7), np.ones(7)),
                   box([1, 0, 0, 0, 0, 0, 0], [2, 1, 1, 1, 1, 1, 1]),
                   box([1, 1, 0, 0, 0, 0, 0], [2, 2, 1, 1, 1, 1, 1]),
                   box(np.full(7, .5), np.full(7, 1.5))]
        edges = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]
        stats = {}
        retained = filter_intersection_edges(regions, edges, 6, statistics=stats)
        self.assertEqual(retained, [(0, 1), (1, 0), (0, 3), (3, 0)])
        self.assertEqual(stats["coordinate_screened_pairs"], 1)
        self.assertEqual(stats["tested_dimension_histogram"], {"6": 1, "7": 1})

    def test_oblique_face_and_point_need_actual_affine_test(self):
        first = box([0, 0], [1, 1])
        first.A, first.b = np.vstack((first.A, [1, 1])), np.r_[first.b, 1]
        second = box([0, 0], [1, 1])
        second.A, second.b = np.vstack((second.A, [-1, -1])), np.r_[second.b, -1]
        third = box([0, 0], [1, 1])
        third.A, third.b = np.vstack((third.A, [-1, 1])), np.r_[third.b, -1]
        stats = {}
        retained = filter_intersection_edges([first, second, third], [(0, 1), (1, 0), (0, 2), (2, 0)],
                                             1, statistics=stats)
        self.assertEqual(retained, [(0, 1), (1, 0)])
        self.assertEqual(stats["affine_dimension_test_pairs"], 2)
