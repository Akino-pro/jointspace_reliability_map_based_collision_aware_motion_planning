import unittest

import numpy as np

from scripts.planar3R_collision import link_points
from scripts.planar3R_ssm_envelopes import build_envelopes, resample_by_arclength
from scripts.planar3R_two_point_ssm_views import refine_ik_configuration
from scripts.planar3R_post_failure_planner import locked_joint_ik


class ArcLengthSamplingTests(unittest.TestCase):
    def test_nonuniform_input_is_resampled_uniformly(self):
        points = np.array([[0, 0, 0], [0.001, 0, 0], [0.1, 0, 0], [1, 0, 0]])
        samples = resample_by_arclength(points, 0.2)
        np.testing.assert_allclose(samples[:, 0], np.linspace(0, 1, 6))
        np.testing.assert_allclose(samples[[0, -1]], points[[0, -1]])

    def test_two_branch_counts_follow_lengths_at_same_spacing(self):
        branches = [np.array([[0, 0, 0], [1, 0, 0]]),
                    np.array([[0, 1, 0], [2, 1, 0]])]
        report = {}
        envelopes = build_envelopes(branches, arc_spacing=0.2, sampling_report=report)
        self.assertEqual(report["ssm_branch_sample_counts"], [6, 11])
        self.assertEqual(report["ssm_branch_arc_lengths"], [1.0, 2.0])
        self.assertEqual(report["ssm_total_arc_length"], 3.0)
        self.assertEqual({e.branch for e in envelopes}, {0, 1})

    def test_periodic_seam_does_not_create_long_chord(self):
        points = np.array([[3.0, 0, 0], [3.1, 0, 0], [-3.1, 0, 0], [-3.0, 0, 0]])
        report = {}
        envelopes = build_envelopes([points], arc_spacing=0.05, sampling_report=report)
        self.assertAlmostEqual(report["ssm_total_arc_length"], 2 * np.pi - 6)
        self.assertEqual({e.piece for e in envelopes}, {0, 1})
        self.assertTrue(all(not e.polytope.contains([0, 0, 0]) for e in envelopes))

    def test_singleton_and_duplicate_points_are_supported(self):
        samples = resample_by_arclength([[1, 2, 3], [1, 2, 3]], 0.1)
        np.testing.assert_array_equal(samples, [[1, 2, 3]])

    def test_invalid_spacing_is_rejected(self):
        for spacing in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                build_envelopes([], arc_spacing=spacing)

    def test_interpolated_samples_are_refined_back_to_actual_goal_ssm(self):
        reference = np.array([0.4, -0.8, 0.7])
        goal = link_points(reference)[-1]
        points = [min(locked_joint_ik(goal, 3, angle)[0],
                      key=lambda q: np.linalg.norm(q - reference)) for angle in (0.6, 0.8)]
        envelopes = build_envelopes(
            [np.array(points)], arc_spacing=0.01,
            refine_sample=lambda q: refine_ik_configuration(q, *goal))
        self.assertTrue(envelopes)
        for envelope in envelopes:
            for q in envelope.samples:
                self.assertLess(np.linalg.norm(link_points(q)[-1] - goal), 1e-8)

    def test_failed_refinement_does_not_bridge_missing_portion(self):
        report = {}
        envelopes = build_envelopes(
            [np.array([[0, 0, 0], [1, 0, 0]])], arc_spacing=0.25,
            refine_sample=lambda q: None if 0.4 < q[0] < 0.6 else q,
            sampling_report=report)
        self.assertEqual(report["sample_refinement_failures"], 1)
        self.assertTrue(all(not e.polytope.contains([0.5, 0, 0]) for e in envelopes))


if __name__ == "__main__":
    unittest.main()
