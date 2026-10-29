"""Five first-stage rounding budgets for the joint-hull planar 3R method only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.planar3R_gcs_planner import GOAL_POINT, OBSTACLES
from scripts.planar3R_two_stage_gcs_planner import plan_two_stage

BUDGETS = ((50, 75), (100, 150), (200, 300), (300, 450), (400, 600))


def write_summary(folder, summary):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["runs"]
    if rows:
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Planar 3R joint-hull first-stage budget sweep", "",
             "Only the joint-hull two-stage method ran. Seed 0; second-stage budget fixed at 200/300.",
             "The first-stage graph has 43 free regions, 20 convex targets, one source and one sink (65 vertices).",
             "Times include graph construction and optimization; total adds shared preprocessing once.",
             "Each budget was measured once, in ascending order. This is not a statistical timing comparison.", "",
             "| Paths / trial cap | Stage-one actual routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total incl. preprocessing (s) | Status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    def fmt(value, places=8):
        return "—" if value is None else f"{value:.{places}f}"
    for row in rows:
        lines.append(f"| {row['max_paths']}/{row['max_trials']} | {row['actual_stage1_paths']} | "
                     f"{fmt(row['stage1_cost'], 10)} | {fmt(row['final_cost'], 10)} | "
                     f"{fmt(row['stage1_seconds'], 3)} | {fmt(row['stage2_seconds'], 3)} | "
                     f"{fmt(row['total_seconds'], 3)} | {row['status']} |")
    lines += ["", "Both graph/task and target-polytope fingerprints are checked across all runs.",
              "Each successful path passes task, collision, graph membership and weighted-cost checks.",
              "Second-stage candidate counts are reported separately; one retained-route restriction is additional to the random caps.",
              "The trial counts are upper bounds: Drake does not expose actual SamplePaths attempt counts.",
              "No primary, sampled-goal or nonlinear-goal method was run.",
              "No IRIS regions, graph edges, target hulls, objective weights or rounding seeds were changed.",
              "Final returned costs are feasible upper bounds, not certified continuous-SSM optima.",
              "", "Per-case JSON files, summary.json and summary.csv retain the unrounded measurements.",
              "The cost/time figure is saved as cost_time.png and cost_time.svg when matplotlib is available."]
    (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(folder, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    good = [row for row in rows if row["success"]]
    if not good:
        return False
    x = [row["max_paths"] for row in good]
    costs = np.array([row["final_cost"] for row in good])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, costs, "o-", label="Final feasible cost")
    margin = max(float(np.ptp(costs))*.2, abs(float(np.mean(costs)))*.001, 1e-6)
    axes[0].set_ylim(float(costs.min())-margin, float(costs.max())+margin)
    axes[0].set_ylabel("Reliability-weighted joint-space cost")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].plot(x, [row["stage1_seconds"] for row in good], "o-", label="Stage one")
    axes[1].plot(x, [row["stage2_seconds"] for row in good], "s-", label="Stage two (fixed 200/300)")
    axes[1].plot(x, [row["total_seconds"] for row in good], "^-", label="Total including preprocessing")
    axes[1].set_ylabel("Measured seconds")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Stage-one candidate-path cap")
        axis.set_xticks(x)
        axis.grid(alpha=.3)
    fig.suptitle("Planar 3R joint-hull method: one run per budget, seed 0")
    fig.tight_layout()
    fig.savefig(folder / "cost_time.png", dpi=180)
    fig.savefig(folder / "cost_time.svg")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/planar3r/union_hull_budget_sweep_20261029")
    parser.add_argument("--start-path", default="results/planar3r/iris_path.json")
    parser.add_argument("--iris-regions", default="data/iris/planar3R_iris_regions.npz")
    args = parser.parse_args()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.start_path).read_text(encoding="utf-8"))
    start = source["waypoints"][0]
    summary = dict(status="running", method="joint_hull_two_stage_only", stage2_budget=[200, 300],
                   rounding_seed=0, geometric_epsilon=.025, ssm_arc_spacing=.044,
                   start=start, goal=list(GOAL_POINT), iris_checkpoint=args.iris_regions,
                   budget_sweep_stage="stage_one_only", repeats_per_budget=1,
                   actual_trial_counts_available=False, runs=[])
    contract = None
    for paths, trials in BUDGETS:
        print(f"BEGIN joint-hull budget {paths}/{trials}; stage two fixed at 200/300", flush=True)
        report = plan_two_stage(start, GOAL_POINT, OBSTACLES, args.iris_regions,
                                max_rounded_paths=paths, max_rounding_trials=trials,
                                stage2_max_rounded_paths=200, stage2_max_rounding_trials=300,
                                union_hulls_only=True)
        case_name = f"paths{paths:03d}_trials{trials:03d}.json"
        (folder / case_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        joint = report["union_hull_baseline"]
        current = (joint["graph_task_sha256"], joint["hull_targets_sha256"])
        if contract is not None and current != contract:
            raise RuntimeError("Graph/task or hull-target geometry changed across budgets")
        contract = current
        if report["primary_method_executed"] or "sampled_baseline" in report or "nonlinear_goal_baseline" in report:
            raise RuntimeError("Unexpected method ran in the joint-hull-only sweep")
        first = joint.get("stage1_rounding_diagnostics", {})
        second = joint.get("stage2_rounding_diagnostics", {})
        row = dict(max_paths=paths, max_trials=trials, success=joint["success"], status=joint["status"],
                   stage1_cost=joint.get("stage1_cost"), final_cost=joint.get("cost"),
                   preprocessing_seconds=report["geometry_seconds"], stage1_seconds=joint.get("stage1_seconds"),
                   projection_seconds=joint.get("projection_seconds"), stage2_seconds=joint.get("stage2_seconds"),
                   total_seconds=report["geometry_seconds"]+joint["seconds"],
                   actual_stage1_paths=first.get("rounded_path_count", 0),
                   successful_stage1_restrictions=first.get("successful_rounded_path_count", 0),
                   actual_stage2_paths=second.get("rounded_path_count", 0),
                   selected_target=joint.get("selected_hull_target"), task_error=joint.get("task_error"),
                   collision_free=joint.get("collision_free"), graph_membership_valid=joint.get("graph_membership_valid"),
                   cost_valid=joint.get("cost_valid"), report=case_name)
        summary["runs"].append(row)
        summary.update(graph_task_sha256=current[0], hull_targets_sha256=current[1],
                       free_regions=report["region_count"], region_edges=report["edge_count"],
                       hull_targets=report["target_set_count"], graph_vertices=report["region_count"]+report["target_set_count"]+2)
        write_summary(folder, summary)
        print("END "+json.dumps(row), flush=True)
    summary["status"] = "complete"
    summary["plot_generated"] = plot_results(folder, summary["runs"])
    write_summary(folder, summary)
    print("Completed all five budgets: "+str(folder / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
