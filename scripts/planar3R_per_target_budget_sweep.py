"""Five full per-target rounding budgets for the original planar 3R method."""
import argparse
import csv
import json
from pathlib import Path

from scripts.planar3R_gcs_planner import GOAL_POINT, OBSTACLES
from scripts.planar3R_two_stage_gcs_planner import plan_two_stage
from scripts.planar3R_union_hull_budget_sweep import BUDGETS


def write_summary(folder, summary, joint):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["runs"]
    if rows:
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fmt = lambda value, digits=10: "—" if value is None else f"{value:.{digits}f}"
    lines = ["# Planar 3R independent-target budget sweep", "",
             "Each target receives its full configured path/trial cap. There is no total-budget division.",
             "Only the original per-target two-stage method ran. Seed 0; second stage fixed at 200/300.",
             "The same 43 free regions, 824 directed edges and 20 target intersections are used throughout.",
             "Costs are feasible returned costs, not certified continuous-SSM optima. Times are single measurements.", "",
             "| Per-target cap | Total requested paths | Actual stage-one paths | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['max_paths']}/{r['max_trials']} | {r['total_path_cap']} | {r['actual_stage1_paths']} | "
                     f"{fmt(r['stage1_cost'])} | {fmt(r['final_cost'])} | {fmt(r['stage1_seconds'],3)} | "
                     f"{fmt(r['stage2_seconds'],3)} | {fmt(r['total_seconds'],3)} | {r['status']} |")
    lines += ["", "## Comparison with the completed joint-hull sweep", "",
              "Joint-hull measurements are reused from its separate sweep, not rerun or included in primary timings.",
              "Both methods have the same per-call stage-one cap and fixed second-stage 200/300 budget.",
              "The independent method performs 20 stage-one relaxations and gets 20 times the stage-one path cap.", "",
              "| Paths/trials per call | Independent final cost | Joint final cost | Independent total (s) | Joint total (s) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    prior = {r["max_paths"]: r for r in joint["runs"]}
    for r in rows:
        b = prior[r["max_paths"]]
        lines.append(f"| {r['max_paths']}/{r['max_trials']} | {fmt(r['final_cost'])} | {fmt(b['final_cost'])} | "
                     f"{fmt(r['total_seconds'],3)} | {fmt(b['total_seconds'],3)} |")
    lines += ["", "Total times include preprocessing once and exclude other methods or file/chart generation.",
              "Each stage-two solve additionally considers the retained first-stage route as a convex restriction.",
              "Path/trial settings are caps; actual SamplePaths trial counts are unavailable and not inferred.",
              "Graph/task and target fingerprints must match both across this sweep and the earlier joint sweep.",
              "All per-target allocations and diagnostics are preserved in the individual JSON reports.",
              "No IRIS regeneration, reference grid, sampled-goal or nonlinear-goal experiment was performed.",
              "", "Charts: cost_time.png and cost_time.svg. Full precision data: summary.json and summary.csv."]
    (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(folder, rows, joint):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    good = [r for r in rows if r["success"]]
    if not good:
        return False
    baseline = {r["max_paths"]: r for r in joint["runs"]}
    x = [r["max_paths"] for r in good]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, [r["final_cost"] for r in good], "o-", label="Independent targets")
    axes[0].plot(x, [baseline[n]["final_cost"] for n in x], "s--", label="Joint targets (previous sweep)")
    axes[0].set_ylabel("Final reliability-weighted cost")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].plot(x, [r["stage1_seconds"] for r in good], "o-", label="Independent stage one")
    axes[1].plot(x, [r["total_seconds"] for r in good], "^-", label="Independent total")
    axes[1].plot(x, [baseline[n]["total_seconds"] for n in x], "s--", label="Joint total (previous sweep)")
    axes[1].set_ylabel("Measured seconds")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xlabel("Stage-one candidate-path cap per GCS call")
        axis.set_xticks(x)
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle("Planar 3R: full per-target budgets; stage two fixed at 200/300; seed 0")
    fig.tight_layout()
    fig.savefig(folder / "cost_time.png", dpi=180)
    fig.savefig(folder / "cost_time.svg")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/planar3r/per_target_budget_sweep_20261029")
    parser.add_argument("--joint-summary", default="results/planar3r/union_hull_budget_sweep_20261029/summary.json")
    parser.add_argument("--start-path", default="results/planar3r/iris_path.json")
    parser.add_argument("--iris-regions", default="data/iris/planar3R_iris_regions.npz")
    args = parser.parse_args()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joint = json.loads(Path(args.joint_summary).read_text(encoding="utf-8"))
    start = json.loads(Path(args.start_path).read_text(encoding="utf-8"))["waypoints"][0]
    summary = dict(status="running", method="independent_target_two_stage_only", stage2_budget=[200, 300],
                   budget_scope="full_cap_per_target_not_divided", rounding_seed=0,
                   geometric_epsilon=.025, ssm_arc_spacing=.044, start=start, goal=list(GOAL_POINT),
                   iris_checkpoint=args.iris_regions, joint_comparison_source=args.joint_summary,
                   repeats_per_budget=1, actual_trial_counts_available=False, runs=[])
    for paths, trials in BUDGETS:
        print(f"BEGIN independent targets: EACH target {paths}/{trials}; stage two 200/300", flush=True)
        report = plan_two_stage(start, GOAL_POINT, OBSTACLES, args.iris_regions,
                                max_rounded_paths=paths, max_rounding_trials=trials,
                                stage2_max_rounded_paths=200, stage2_max_rounding_trials=300)
        name = f"paths{paths:03d}_trials{trials:03d}.json"
        (folder / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if any(key in report for key in ("sampled_baseline", "union_hull_baseline", "nonlinear_goal_baseline")):
            raise RuntimeError("Unexpected comparison method ran")
        if (report["graph_task_sha256"] != joint["graph_task_sha256"]
                or report["hull_targets_sha256"] != joint["hull_targets_sha256"]):
            raise RuntimeError("Graph/task or target geometry differs from the joint-hull sweep")
        quotas = report["stage1_budget"]["per_target"]
        if not all(q == dict(max_rounded_paths=paths, max_rounding_trials=trials) for q in quotas):
            raise RuntimeError("Per-target budget was unexpectedly divided")
        diagnostics = report.get("stage1_rounding_diagnostics", {})
        row = dict(max_paths=paths, max_trials=trials, target_count=report["target_set_count"],
                   total_path_cap=report["stage1_budget"]["total_max_rounded_paths"],
                   total_trial_cap=report["stage1_budget"]["total_max_rounding_trials"],
                   actual_stage1_paths=diagnostics.get("rounded_path_count", 0),
                   successful_stage1_restrictions=diagnostics.get("successful_rounded_path_count", 0),
                   feasible_targets=sum(item["success"] for item in report.get("stage1_candidates", [])),
                   stage1_cost=report.get("stage1_cost"), final_cost=report.get("cost"),
                   preprocessing_seconds=report["geometry_seconds"], stage1_seconds=report.get("stage1_seconds"),
                   projection_seconds=report.get("projection_seconds"), stage2_seconds=report.get("stage2_seconds"),
                   total_seconds=report["two_stage_total_seconds"],
                   actual_stage2_paths=report.get("stage2_rounded_path_count"),
                   additional_stage2_restrictions=report.get("stage2_additional_convex_restriction_calls"),
                   selected_target=report.get("selected_target"), endpoint_error=report.get("endpoint_error"),
                   collision_free=report.get("collision_free"), collision_samples=report.get("collision_samples"),
                   success=report["success"], status=report["status"], report=name)
        summary["runs"].append(row)
        summary.update(graph_task_sha256=report["graph_task_sha256"],
                       hull_targets_sha256=report["hull_targets_sha256"], free_regions=report["region_count"],
                       region_edges=report["edge_count"], hull_targets=report["target_set_count"])
        write_summary(folder, summary, joint)
        print("END "+json.dumps(row), flush=True)
    summary["status"] = "complete"
    summary["plot_generated"] = plot_results(folder, summary["runs"], joint)
    write_summary(folder, summary, joint)
    print("Completed all five independent-target budgets: "+str(folder / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
