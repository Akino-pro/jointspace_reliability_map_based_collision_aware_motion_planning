"""Primary CLI policy must not silently run the per-target benchmark."""
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import planar3R_two_stage_gcs_planner as workflow


class PrimaryCliTests(unittest.TestCase):
    def settings(self, extra):
        signature = inspect.signature(workflow.plan_two_stage)
        with tempfile.TemporaryDirectory() as directory:
            argv = ["planner", "--q", "0", "0", "0", "--output", str(Path(directory)/"report.json"), *extra]
            with patch("sys.argv", argv), patch.object(workflow, "plan_two_stage", return_value={"success": True}) as solve, patch("builtins.print"):
                workflow.main()
            return signature.bind(*solve.call_args.args, **solve.call_args.kwargs).arguments

    def test_default_union_hull_800_1200_both_stages(self):
        settings = self.settings([])
        self.assertTrue(settings["union_hulls_only"])
        self.assertEqual((settings["max_rounded_paths"], settings["max_rounding_trials"]), (800, 1200))
        self.assertEqual((settings["stage2_max_rounded_paths"], settings["stage2_max_rounding_trials"]), (800, 1200))

    def test_per_target_is_explicit_and_keeps_historical_defaults(self):
        settings = self.settings(["--method", "per-target"])
        self.assertFalse(settings["union_hulls_only"])
        self.assertEqual(settings["max_rounded_paths"], workflow.STAGE1_MAX_ROUNDED_PATHS)


if __name__ == "__main__":
    unittest.main()
