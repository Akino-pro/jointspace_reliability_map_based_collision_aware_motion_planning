"""Run only union-hull with configurable matching budgets in both stages."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def worker(robot, paths, folder, discard_rounding_history=False):
    trials = paths*3//2
    if robot == "3r":
        if discard_rounding_history:
            raise ValueError("Discard-history experiment currently supports 4R only")
        from scripts.planar3R_gcs_planner import GOAL_POINT, OBSTACLES
        from scripts.planar3R_two_stage_gcs_planner import plan_two_stage
        start = json.loads(Path("results/planar3r/iris_path.json").read_text())["waypoints"][0]
        report = plan_two_stage(start, GOAL_POINT, OBSTACLES,
                                max_rounded_paths=paths, max_rounding_trials=trials,
                                stage2_max_rounded_paths=paths, stage2_max_rounding_trials=trials,
                                union_hulls_only=True)
    else:
        from scripts.spatial4R_two_stage_gcs_planner import plan_before_failure
        report = plan_before_failure(max_paths=paths, max_trials=trials,
                                     stage2_max_paths=paths, stage2_max_trials=trials,
                                     union_hulls_only=True, retain_rounded_results=not discard_rounding_history)
    assert report["primary_method_executed"] is False
    assert "sampled_baseline" not in report and "nonlinear_goal_baseline" not in report
    (folder / f"{robot}_paths{paths}.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/diagnostics/union_hull_large_budget_20261030")
    parser.add_argument("--worker", choices=["3r", "4r"])
    parser.add_argument("--paths", type=int)
    parser.add_argument("--robots", nargs="+", choices=["3r", "4r"], default=["3r", "4r"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[800, 1600, 3200])
    parser.add_argument("--discard-rounding-history", action="store_true",
                        help="4R only: keep best restriction result and counts, not all result objects")
    args = parser.parse_args()
    if any(p < 1 or p % 2 for p in args.budgets):
        parser.error("Budgets must be positive even path counts for exact 1.5x trial caps")
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.paths, folder, args.discard_rounding_history)
        return
    references = {
        "3r": "results/planar3r/per_target_budget_sweep_20261029/paths200_trials300.json",
        "4r": "results/spatial4r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json",
    }
    summary = dict(status="running", first_method_rerun=False, seed=0,
                   retain_rounded_results=not args.discard_rounding_history,
                   budgets=[[p, p*3//2] for p in args.budgets],
                   stage2_budget_policy="matches_stage_one", rss_limit_gib=6,
                   timing_note="Fresh sequential single runs; cached IRIS; no repeated-run timing statistics.",
                   references={}, runs=[])
    for robot in args.robots:
        filename = references[robot]
        ref = json.loads(Path(filename).read_text())
        summary["references"][robot] = dict(source=filename, cost=ref["cost"],
                                            total_seconds=ref["two_stage_total_seconds"],
                                            graph_task_sha256=ref["graph_task_sha256"],
                                            hull_targets_sha256=ref["hull_targets_sha256"])
    def save():
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        if summary["runs"]:
            with (folder / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary["runs"][0]))
                writer.writeheader()
                writer.writerows(summary["runs"])
    save()
    for robot in args.robots:
        for paths in args.budgets:
            stem = f"{robot}_paths{paths}"
            print(f"BEGIN {stem}: both stages {paths}/{paths*3//2}", flush=True)
            began, peak, stopped = time.perf_counter(), 0, False
            with (folder / f"{stem}.log").open("w") as output:
                command = [sys.executable, "-u", "-m", __spec__.name,
                                          "--worker", robot, "--paths", str(paths),
                                          "--output-dir", str(folder)]
                if args.discard_rounding_history:
                    command.append("--discard-rounding-history")
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        rss = next(int(line.split()[1])*1024 for line in status.splitlines()
                                   if line.startswith("VmRSS:"))
                        peak = max(peak, rss)
                        if rss >= 6*1024**3:
                            stopped = True
                            child.terminate()
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                    except (FileNotFoundError, StopIteration, ProcessLookupError):
                        pass
                    time.sleep(.2)
            row = dict(robot=robot, paths=paths, trials=paths*3//2, status="memory_limit" if stopped else "worker_failed",
                       success=False, cost=None, relative_cost_above_reference=None, total_seconds=None,
                       wall_seconds=time.perf_counter()-began, peak_rss_gib=peak/1024**3,
                       stage1_seconds=None, stage2_seconds=None, stage1_routes=None, stage2_routes=None,
                       task_error=None, collision_free=None, graph_valid=None, cost_valid=None,
                       selected_target=None, report=f"{stem}.json", exit_code=child.returncode)
            report_path = folder / row["report"]
            if child.returncode == 0 and report_path.exists():
                report = json.loads(report_path.read_text())
                answer = report["union_hull_baseline"]
                ref = summary["references"][robot]
                if (report["graph_task_sha256"] != ref["graph_task_sha256"] or
                        report["hull_targets_sha256"] != ref["hull_targets_sha256"]):
                    raise RuntimeError("Input graph or target geometry differs from historical first-method result")
                row.update(status=answer["status"], success=answer["success"], cost=answer.get("cost"),
                           total_seconds=report["total_seconds"],
                           stage1_seconds=answer.get("stage1_seconds"), stage2_seconds=answer.get("stage2_seconds"),
                           stage1_routes=answer.get("stage1_rounding_diagnostics", {}).get("rounded_path_count", 0),
                           stage2_routes=answer.get("stage2_rounding_diagnostics", {}).get("rounded_path_count", 0),
                           task_error=answer.get("task_error"), collision_free=answer.get("collision_free"),
                           graph_valid=answer.get("graph_membership_valid"), cost_valid=answer.get("cost_valid"),
                           selected_target=answer.get("selected_hull_target"))
                if row["cost"] is not None:
                    row["relative_cost_above_reference"] = row["cost"]/ref["cost"]-1
            summary["runs"].append(row)
            save()
            print("END "+json.dumps(row), flush=True)
    summary["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
