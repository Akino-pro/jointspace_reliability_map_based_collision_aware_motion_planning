"""Five per-target 7R budgets with matching stage-two budgets."""
import argparse
import csv
import json
from pathlib import Path

from scripts import spatial7R_model as model
from scripts.spatial7R_two_stage_gcs_planner import plan_before_failure
from scripts.planar3R_union_hull_budget_sweep import BUDGETS


def write_summary(folder, summary):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["runs"]
    if rows:
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fmt = lambda value, digits=10: "—" if value is None else f"{value:.{digits}f}"
    lines = ["# Spatial 7R independent-target budget sweep", "",
             f"Robot: {summary['robot']}. Existing midpoint scene and IRIS cache; no scene or graph modification.",
             ("Both stages use each case's path/trial budget; each first-stage target receives the full budget."
              if summary.get("stage2_budget_mode") == "match_first" else
              "Each target receives its full budget, without total-budget division. Stage two is fixed at 200/300."),
             "Only the primary per-target two-stage method runs. Rounding seed 0; SSM search seed 1.",
             "Times include shared preprocessing once and exclude plotting/file generation and other methods.",
             "Each budget is one sequential run, not a statistical timing benchmark.", "",
             "| Per-target paths/trials | Targets with returned paths | Actual stage-one routes | Stage-one cost | Final cost | Stage one (s) | Stage two (s) | Total (s) | Status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['max_paths']}/{r['max_trials']} | {r['feasible_targets']}/{r['target_count']} | "
                     f"{r['actual_stage1_paths']} | {fmt(r['stage1_cost'])} | {fmt(r['final_cost'])} | "
                     f"{fmt(r['stage1_seconds'],3)} | {fmt(r['stage2_seconds'],3)} | "
                     f"{fmt(r['total_seconds'],3)} | {r['status']} |")
    lines += ["", "Graph/task and target fingerprints are checked across budgets. Failed targets remain in the report.",
              "No returned path for a target is not proof of physical infeasibility.",
              "Actual unique routes and successful restrictions are recorded separately from requested caps.",
              "Actual native sampling-trial attempt counts are not exposed and are not inferred from caps.",
              ("Second-stage random caps match the case budget,plus one retained first-stage route restriction."
               if summary.get("stage2_budget_mode") == "match_first" else
               "Second-stage random cap is 200/300 plus one retained first-stage route restriction."),
              "Returned costs are feasible upper bounds, not globally certified continuous-goal optima.",
              "No joint-hull, sampled-goal or nonlinear-goal comparison was executed.",
              "The 7R defaults are not changed automatically by this experiment.",
              "", "Full precision: summary.json, summary.csv and each case JSON. Plot: cost_time.png and cost_time.svg."]
    (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(folder, rows, matched=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    good = [r for r in rows if r["success"]]
    if not good:
        return False
    x = [r["max_paths"] for r in good]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, [r["final_cost"] for r in good], "o-", label="Final feasible cost")
    axes[0].set_ylabel("Final reliability-weighted joint-space cost")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].plot(x, [r["stage1_seconds"] for r in good], "o-", label="Stage one")
    axes[1].plot(x, [r["stage2_seconds"] for r in good], "s-",
                 label="Stage two (matched)" if matched else "Stage two (200/300)")
    axes[1].plot(x, [r["total_seconds"] for r in good], "^-", label="Total including preprocessing")
    axes[1].set_ylabel("Measured seconds")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xlabel("Candidate cap in both stages" if matched else "Stage-one candidate-path cap per target")
        axis.set_xticks(x)
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle("Spatial 7R full-pose 60cm/base-Z90deg: full per-target budgets; seed 0")
    fig.tight_layout()
    fig.savefig(folder / "cost_time.png", dpi=180)
    fig.savefig(folder / "cost_time.svg")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/spatial7r/per_target_matched_stage_budget_sweep_60cm90deg")
    parser.add_argument("--iris-regions", default="data/iris/spatial7R_60cm90deg_iris_regions.npz")
    parser.add_argument("--matched-stage-budgets", action="store_true", default=True,
                        help="Set both stages to each case's budget instead of fixing stage two at200/300")
    args = parser.parse_args()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(status="running", method="spatial7R_independent_target_two_stage_only", robot="spatial7R",
                   budget_scope="full_cap_per_target_not_divided",
                   stage2_budget=None if args.matched_stage_budgets else [200, 300],
                   stage2_budget_mode="match_first" if args.matched_stage_budgets else "fixed_200_300",
                   rounding_seed=0, ssm_seed=1, ssm_ik_attempts=model.MAX_SSM_IK_ATTEMPTS,
                   geometric_epsilon=.025, ssm_arc_spacing=.044, iris_checkpoint=args.iris_regions,
                   repeats_per_budget=1, actual_trial_counts_available=False, runs=[])
    contract = None
    for paths, trials in BUDGETS:
        second_paths, second_trials = (paths, trials) if args.matched_stage_budgets else (200, 300)
        print(f"BEGIN 7R independent targets: EACH target {paths}/{trials}; stage two {second_paths}/{second_trials}", flush=True)
        report = plan_before_failure(start=model.START_CONFIGURATION,
                                     reference=model.GOAL_REFERENCE_CONFIGURATION,
                                     checkpoint=args.iris_regions, max_paths=paths, max_trials=trials,
                                     arc_spacing=.044,
                                     solver_backend="mosek", ssm_seed=1)
        name = f"paths{paths:03d}_trials{trials:03d}.json"
        (folder / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if any(key in report for key in ("sampled_baseline", "union_hull_baseline", "nonlinear_goal_baseline")):
            raise RuntimeError("Unexpected comparison method ran")
        current = (report["graph_task_sha256"], report["hull_targets_sha256"])
        if contract is not None and current != contract:
            raise RuntimeError("Graph/task or target geometry changed across budgets")
        contract = current
        candidates = report.get("stage1_candidates", [])
        if not all(c["solver_settings"]["max_rounded_paths"] == paths
                   and c["solver_settings"]["max_rounding_trials"] == trials for c in candidates):
            raise RuntimeError("Unexpected per-target budget")
        if (report["stage2_solver_settings"]["max_rounded_paths"] != second_paths
                or report["stage2_solver_settings"]["max_rounding_trials"] != second_trials):
            raise RuntimeError("Second-stage budget changed")
        diagnostics = report.get("stage1_rounding_diagnostics", {})
        row = dict(max_paths=paths, max_trials=trials, target_count=report["target_set_count"],
                   stage2_max_paths=second_paths, stage2_max_trials=second_trials,
                   total_path_cap=report["stage1_budget"]["total_max_rounded_paths"],
                   total_trial_cap=report["stage1_budget"]["total_max_rounding_trials"],
                   actual_stage1_paths=diagnostics.get("rounded_path_count", 0),
                   successful_stage1_restrictions=diagnostics.get("successful_rounded_path_count", 0),
                   feasible_targets=sum(c["success"] for c in candidates),
                   stage1_cost=report.get("stage1_cost"), final_cost=report.get("cost"),
                   preprocessing_seconds=report["geometry_seconds"], stage1_seconds=report.get("stage1_seconds"),
                   stage2_seconds=report.get("stage2_seconds"), total_seconds=report["two_stage_total_seconds"],
                   actual_stage2_paths=report.get("stage2_rounded_path_count"),
                   additional_stage2_restrictions=len(report.get("stage2_additional_route_results", [])),
                   selected_target=report.get("selected_target"), endpoint_error=report.get("endpoint_error"),
                   orientation_error_radians=report.get("orientation_error_radians"),
                   graph_path_valid=report.get("graph_path_valid"),
                   collision_free=report.get("collision_free"), collision_samples=report.get("collision_samples"),
                   success=report["success"], status=report["status"], report=name)
        summary["runs"].append(row)
        summary.update(graph_task_sha256=current[0], hull_targets_sha256=current[1],
                       free_regions=report["region_count"], region_edges=report["edge_count"],
                       hull_targets=report["target_set_count"], ssm_branches=report["ssm_branch_count"])
        write_summary(folder, summary)
        print("END "+json.dumps(row), flush=True)
    summary["status"] = "complete"
    summary["plot_generated"] = plot_results(folder, summary["runs"], args.matched_stage_budgets)
    write_summary(folder, summary)
    print("Completed all five 7R per-target budgets: "+str(folder / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
