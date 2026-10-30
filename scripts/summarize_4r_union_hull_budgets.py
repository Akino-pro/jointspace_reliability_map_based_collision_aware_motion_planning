"""Compare saved 4R union-hull budgets to the saved per-target reference."""
import csv
import json
from pathlib import Path


def main():
    output = Path("results/spatial4r/union_hull_high_budget_20261030")
    low = Path("results/spatial4r/union_hull_primal_20261030")
    reference_path = Path("results/spatial4r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json")
    reference = json.loads(reference_path.read_text())
    rows = []
    for paths in (800, 1600, 3200, 6400, 12800, 25600):
        path = (low if paths <= 3200 else output) / f"4r_paths{paths}.json"
        if paths == 25600 and not path.exists():
            path = Path("results/spatial4r/union_hull_high_budget_bounded_history_20261030/4r_paths25600.json")
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        assert report["primary_method_executed"] is False
        for key in ("graph_task_sha256", "hull_targets_sha256"):
            assert report[key] == reference[key], f"Different geometry: {path}"
        answer = report["union_hull_baseline"]
        rows.append(dict(paths=paths, trials=paths*3//2, success=answer["success"],
                         stage1_cost=answer.get("stage1_cost"),
                         cost=answer.get("cost"), seconds=report["total_seconds"],
                         gap_percent=(100*(answer["cost"]/reference["cost"]-1)
                                      if answer.get("cost") is not None else None),
                         selected_target=answer.get("selected_hull_target"),
                         retain_rounded_results=answer["solver_settings"].get("retain_rounded_results", True),
                         stage1_paths=answer.get("stage1_rounding_diagnostics", {}).get("rounded_path_count", 0),
                         stage2_paths=answer.get("stage2_rounding_diagnostics", {}).get("rounded_path_count", 0),
                         task_error=answer.get("task_error"), collision_free=answer.get("collision_free"),
                         graph_valid=answer.get("graph_membership_valid"), cost_valid=answer.get("cost_valid"),
                         report=str(path)))
    comparison = dict(reference=str(reference_path), reference_cost=reference["cost"],
                      reference_seconds=reference["two_stage_total_seconds"],
                      first_method_rerun=False, seed=0, both_stages_same_budget=True,
                      runs=rows, timing_note="Sequential single runs; cached IRIS; not repeated timing statistics")
    (output / "all_budgets_comparison.json").write_text(json.dumps(comparison, indent=2))
    with (output / "all_budgets_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 4R union-hull higher sampling budgets", "",
             "Only union-hull ran. Each stage uses the listed paths/trial caps, seed 0, MOSEK primal form.",
             "No graph, targets, collision geometry, objective weights or solver tolerances changed.",
             "Input graph and target fingerprints match the saved per-target reference.", "",
             f"Reference: per-target 200/300, cost {reference['cost']:.12f}, total {reference['two_stage_total_seconds']:.3f} s. It was not rerun.", "",
             "| Paths/trials in both stages | Final cost | Gap to reference | Total seconds | Actual stage 1 / stage 2 paths | Success |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        cost = "unavailable" if row["cost"] is None else f"{row['cost']:.12f}"
        gap = "unavailable" if row["gap_percent"] is None else f"{row['gap_percent']:.3f}%"
        lines.append(f"| {row['paths']}/{row['trials']} | {cost} | {gap} | {row['seconds']:.3f} | "
                     f"{row['stage1_paths']} / {row['stage2_paths']} | {row['success']} |")
    lines += ["", "Every successful final path is checked for goal, collision, graph membership and cost.",
              "The gap is against a saved feasible reference, not a certified global optimum.",
              "Trials are caps; actual attempt counts are not exposed by Drake.",
              "This combines sequential single runs, not repeated statistical measurements.",
              "Full high-budget worker RSS and outcomes are in summary.json; raw reports and logs are preserved.", "",
              "The initial 25600-path attempt hit the 6 GiB RSS guard after 255.7 s. Its failure remains in summary.json.",
              "The completed retry is in ../union_hull_high_budget_bounded_history_20261030, with full-result history disabled.",
              "Every candidate is still solved in the original order; only the best result and success counters are retained.",
              "The retry's successful-run time excludes the failed initial attempt; it is not a cumulative experiment time.",
              "The default history-retaining behavior is unchanged for other runs.", "",
              "Reproduce the high-budget runs in the configured Drake container:", "",
              "```powershell",
              "docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.union_hull_large_budget_sweep --robots 4r --budgets 6400 12800 25600 --output-dir results/spatial4r/union_hull_high_budget_repeat",
              "```", ""]
    lines += ["Retry the 25600-path case with bounded result history:", "", "```powershell",
              "docker exec -e MOSEKLM_LICENSE_FILE=/tmp/mosek.lic -w /workspace planar3r-iris-validation /usr/bin/python3 -u -m scripts.union_hull_large_budget_sweep --robots 4r --budgets 25600 --discard-rounding-history --output-dir results/spatial4r/union_hull_high_budget_bounded_repeat",
              "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [r for r in rows if r["success"]]
    x = [r["paths"] for r in good]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, [r["cost"] for r in good], "o-", label="Union-hull")
    axes[0].axhline(reference["cost"], color="tab:orange", linestyle="--", label="Per-target 200/300 reference")
    axes[0].set_ylabel("Final reliability-weighted cost")
    axes[1].plot(x, [r["seconds"] for r in good], "o-", label="Union-hull")
    axes[1].axhline(reference["two_stage_total_seconds"], color="tab:orange", linestyle="--", label="Saved per-target time")
    axes[1].set_ylabel("Total seconds (single runs)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [str(p) for p in x], rotation=25)
        axis.set_xlabel("Path cap in EACH stage (trial cap = 1.5x)")
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    fig.suptitle("Spatial 4R union-hull: seed 0, unchanged graph, MOSEK primal")
    fig.tight_layout()
    fig.savefig(output / "cost_time.png", dpi=180)
    fig.savefig(output / "cost_time.svg")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
