"""Run both real full-graph workers sequentially and record failures and memory usage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.spatial7R_full_graph_comparison import FOLDER, FACET_FOLDER, COMPACT_FOLDER

def memory_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])*1024
    except FileNotFoundError:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facet-adjacency", action="store_true")
    parser.add_argument("--compact-constraints", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--rss-limit-gib", type=float, default=6.)
    parser.add_argument("--representative-pruning", action="store_true")
    parser.add_argument("--pruning-alpha", type=float, default=1.1)
    parser.add_argument("--pruning-trials", type=int, default=100)
    parser.add_argument("--pruning-method", choices=["centers", "intersection-slp"], default="centers")
    parser.add_argument("--slp-cache")
    parser.add_argument("--slp-iterations", type=int, default=80)
    parser.add_argument("--max-rounded-paths", type=int, default=200)
    parser.add_argument("--max-rounding-trials", type=int, default=300)
    parser.add_argument("--discard-rounding-history", action="store_true")
    parser.add_argument("--mosek-solve-form", choices=["primal", "dual"])
    parser.add_argument("--mosek-threads", type=int)
    parser.add_argument("--methods", nargs="+", choices=["union_hulls", "per_target"], default=["union_hulls", "per_target"])
    args = parser.parse_args()
    if args.representative_pruning and not args.output_dir:
        parser.error("Use --output-dir for pruned experiments to preserve full-graph results")
    if args.compact_constraints and not (args.facet_adjacency or args.representative_pruning):
        parser.error("Compact full-adjacency experiments require pre-model representative pruning")
    folder = COMPACT_FOLDER if args.compact_constraints else FACET_FOLDER if args.facet_adjacency else FOLDER
    if args.output_dir:
        folder = Path(args.output_dir)
    if not 0 < args.rss_limit_gib <= 7:
        parser.error("Worker RSS limit must be positive and at most 7 GiB on this VM")
    rss_limit = int(args.rss_limit_gib*1024**3)
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(status="running", budgets=[args.max_rounded_paths, args.max_rounding_trials],
                   coverage="complete existing 2856-region reliability partition; no cropping",
                   rss_limit_bytes=rss_limit, runs=[])
    summary["adjacency_policy"] = "intersection_dimension_at_least_6" if args.facet_adjacency else "nonempty_intersection"
    summary["compact_constraints"] = args.compact_constraints
    summary["representative_pruning"] = args.representative_pruning
    preparation = json.loads(((FACET_FOLDER if args.facet_adjacency else FOLDER) / "prepare.json").read_text())
    summary.update(region_count=preparation["region_count"], edge_count=preparation["edge_count"],
                   target_count=preparation["target_set_count"],
                   shared_preprocessing_seconds=preparation["total_seconds"],
                   graph_task_sha256=preparation["graph_task_sha256"],
                   hull_targets_sha256=preparation["hull_targets_sha256"])
    (folder / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for method in args.methods:
        log = folder / f"{method}.log"
        began = time.perf_counter()
        peak = 0
        stopped = False
        print(f"BEGIN supervised {method}", flush=True)
        with log.open("w") as output:
            command = [sys.executable, "-u", "-m", "scripts.spatial7R_full_graph_comparison", method]
            command.extend(["--max-rounded-paths", str(args.max_rounded_paths),
                            "--max-rounding-trials", str(args.max_rounding_trials),
                            "--pruning-method", args.pruning_method, "--slp-iterations", str(args.slp_iterations)])
            if args.slp_cache:
                command.extend(["--slp-cache", args.slp_cache])
            if args.discard_rounding_history:
                command.append("--discard-rounding-history")
            if args.mosek_solve_form:
                command.extend(["--mosek-solve-form", args.mosek_solve_form])
            if args.mosek_threads is not None:
                command.extend(["--mosek-threads", str(args.mosek_threads)])
            if args.facet_adjacency:
                command.append("--facet-adjacency")
            if args.compact_constraints:
                command.append("--compact-constraints")
            if args.output_dir:
                command.extend(["--output-dir", str(folder)])
            if args.representative_pruning:
                command.extend(["--representative-pruning", "--pruning-alpha", str(args.pruning_alpha),
                                "--pruning-trials", str(args.pruning_trials)])
            child = subprocess.Popen(command,
                                     stdout=output, stderr=subprocess.STDOUT)
            while child.poll() is None:
                rss = memory_bytes(child.pid)
                peak = max(peak, rss)
                if rss >= rss_limit:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    stopped = True
                    break
                time.sleep(.2)
        row = dict(method=method, worker_wall_seconds=time.perf_counter()-began,
                   peak_rss_bytes=peak, exit_code=child.returncode,
                   status=f"memory_limit_{args.rss_limit_gib:g}GiB" if stopped else "worker_failed", log=log.name)
        log_text = log.read_text()
        row["last_phase"] = ("relaxation_or_rounding" if "GCS terminals attached" in log_text else
                             "attach_terminals" if "GCS region graph built" in log_text else
                             "gcs_model_construction" if "GCS model build:" in log_text else
                             "gcs_model_or_solve" if "7R representative pruning:" in log_text else "preprocessing")
        report_path = folder / f"{method}.json"
        if report_path.exists() and not stopped:
            report = json.loads(report_path.read_text())
            if (report.get("graph_task_sha256") != summary["graph_task_sha256"] or
                    report.get("hull_targets_sha256") != summary["hull_targets_sha256"]):
                raise RuntimeError("Graph/task or target geometry differs between methods")
            answer = report.get("union_hull_baseline", report)
            row.update(status=answer["status"], success=bool(answer.get("success")), last_phase="completed",
                       final_cost=answer.get("cost"), stage1_seconds=answer.get("stage1_seconds"),
                       stage2_seconds=answer.get("stage2_seconds"),
                       preprocessing_seconds=report.get("geometry_seconds"),
                       total_seconds=report["total_seconds"])
            if "representative_pruning" in report:
                row["pruning"] = report["representative_pruning"]
                row["quality_scope"] = "heuristically_pruned_graph_only"
        summary["runs"].append(row)
        (folder / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("END "+json.dumps({k: v for k, v in row.items() if k != "pruning"}), flush=True)
    summary["status"] = "complete"
    (folder / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
