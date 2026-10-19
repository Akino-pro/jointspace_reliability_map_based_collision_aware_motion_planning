import unittest

from gcs.quality import bound_quality


class QualityTests(unittest.TestCase):
    def test_gap_uses_official_lower_bound_denominator(self):
        q = bound_quality(0.3618987263, 0.3850930626)
        self.assertAlmostEqual(q["relative_gap"], 0.064090685, places=6)
        self.assertFalse(q["within_tolerance"])
        self.assertEqual(q["status"], "gap_exceeds_tolerance")

    def test_requested_tolerance_changes_quality_not_feasibility(self):
        self.assertTrue(bound_quality(100, 100.5, 0.01)["within_tolerance"])
        self.assertFalse(bound_quality(100, 100.5, 0.001)["within_tolerance"])

    def test_missing_lower_bound_cannot_certify_a_good_path(self):
        q = bound_quality(None, 0.1)
        self.assertFalse(q["within_tolerance"])
        self.assertEqual(q["status"], "bound_unavailable")

    def test_zero_lower_bound_is_handled_without_division_by_zero(self):
        self.assertIsNone(bound_quality(0, 1)["relative_gap"])
        self.assertTrue(bound_quality(0, 0)["within_tolerance"])

    def test_inconsistent_bounds_are_not_silently_certified(self):
        q = bound_quality(2, 1)
        self.assertEqual(q["status"], "inconsistent_bounds")
        self.assertFalse(q["within_tolerance"])

    def test_invalid_tolerance_is_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bound_quality(1, 1, value)


if __name__ == "__main__":
    unittest.main()
