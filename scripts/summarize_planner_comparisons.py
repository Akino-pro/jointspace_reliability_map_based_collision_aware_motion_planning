"""Summarize saved paired runs without rerunning or changing their measurements."""
import argparse
import json
from pathlib import Path


def summarize(paths):
    rows = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        baseline = report.get("sampled_baseline", {})
        common = report.get("geometry_seconds")
        row = dict(source=str(path), planner=report.get("planner"),
                   reliability_rule=report.get("reliability_rule", "unspecified_historical_rule"),
                   two_stage_success=report.get("success", False),
                   baseline_success=baseline.get("success", False),
                   two_stage_cost=report.get("cost"), sampled_cost=baseline.get("cost"),
                   shared_preprocessing_seconds=common,
                   two_stage_total_seconds=report.get("two_stage_total_seconds"),
                   sampled_incremental_seconds=baseline.get("seconds"),
                   sampled_total_with_shared_preprocessing_seconds=(common+baseline["seconds"]
                       if common is not None and "seconds" in baseline else None),
                   samples=report.get("actual_sample_count"),
                   covered_sampled_targets=baseline.get("target_count"),
                   effective_hull_targets=report.get("target_set_count"),
                   adjacency=report.get("adjacency_statistics"),
                   periodic=report.get("periodic_connections", False),
                   baseline_input_sha256=baseline.get("input_sha256"),
                   baseline_solver_settings=baseline.get("solver_settings"),
                   two_stage_solver_settings=report.get("stage1_solver_settings", report.get("solver_settings")),
                   stage1_quality=report.get("stage1_quality"),
                   stage2_quality=report.get("stage2_quality"),
                   sampled_quality=baseline.get("quality"),
                   global_optimality_certified=False)
        if row["two_stage_success"] and row["baseline_success"]:
            row["two_stage_minus_sampled_cost"] = row["two_stage_cost"]-row["sampled_cost"]
        for key in ("union_hull_baseline", "nonlinear_goal_baseline"):
            if key not in report:
                continue
            comparison = report[key]
            row[key] = dict(success=comparison.get("success", False), status=comparison.get("status"),
                            cost=comparison.get("cost"), incremental_seconds=comparison.get("seconds"),
                            total_with_shared_preprocessing_seconds=(common+comparison["seconds"]
                                if common is not None and "seconds" in comparison else None),
                            quality=comparison.get("quality"), task_error=comparison.get("task_error"),
                            gcs_solve_calls=comparison.get("gcs_solve_calls"),
                            input_sha256=comparison.get("input_sha256"),
                            solver_settings=comparison.get("solver_settings"),
                            restriction_solver=comparison.get("restriction_solver"))
            if report.get("success") and comparison.get("success"):
                row[key]["two_stage_minus_baseline_cost"] = report["cost"]-comparison["cost"]
        rows.append(row)
    return dict(comparisons=rows,
                timing_note="Cached runs; shared preprocessing is counted once for each method. "
                            "Measured sequentially, with no claim of repeated-run statistics.",
                interpretation="These are heuristic GCS results for different terminal models, "
                               "not ground truth or continuous-SSM optimality certificates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="results/diagnostics/planner_sampled_comparisons.json")
    args = parser.parse_args()
    result = summarize(args.reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
