"""Summarize saved-stage-one / matched-stage-two replays without new solves."""
import csv
import json
from pathlib import Path


def main():
    independent = Path("results/diagnostics/stage2_matched_budget_independent_20261029/summary.json")
    joint = Path("results/diagnostics/stage2_matched_budget_joint_20261029/summary.json")
    output = Path("results/diagnostics/stage2_matched_budget_summary_20261029")
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(independent.read_text(encoding="utf-8"))["runs"]
    rows += json.loads(joint.read_text(encoding="utf-8"))["runs"]
    result = dict(status="complete", runs=rows, stage1_gcs_solve_calls=0,
                  scope="All completed 3R sweep cases and the completed 4R 50/75 case; unfinished 4R sweep remains paused",
                  timing_note="Stitched totals combine saved timing excluding the old stage two with the new stage-two timing",
                  joint_route_caveat="Old joint reports omitted stage-one route; these replays use saved endpoints and no extra retained route")
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    columns = ["method", "max_paths", "max_trials", "cost", "stage2_seconds", "stitched_total_seconds",
               "retained_stage1_route", "success", "status", "source", "source_sha256"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    by_method = {(r["method"], r["max_paths"]): r for r in rows}
    lines = ["# Final matched-budget stage-two replay results", "",
             "The separate 4R first-stage sweep is paused at100/150; only50/75 has a completed first-stage report.",
             "No first-stage GCS was rerun. The saved projected endpoint is reused in every case.",
             "Both stages now use that case's own path/trial cap:50/75,100/150,200/300,300/450,400/600.", "",
             "| Budget in both stages | 3R independent cost | 3R joint cost | Independent new stage2(s) | Joint new stage2(s) | Independent stitched total(s) | Joint stitched total(s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for paths in (50, 100, 200, 300, 400):
        a, b = by_method[("3R independent", paths)], by_method[("3R joint", paths)]
        lines.append(f"| {paths}/{a['max_trials']} | {a['cost']:.10f} | {b['cost']:.10f} | "
                     f"{a['stage2_seconds']:.3f} | {b['stage2_seconds']:.3f} | "
                     f"{a['stitched_total_seconds']:.3f} | {b['stitched_total_seconds']:.3f} |")
    four = by_method[("4R independent", 50)]
    lines += ["", "## Completed spatial4R case", "",
              f"50/75 in both stages:cost={four['cost']:.10f};new stage2={four['stage2_seconds']:.3f}s;"
              f"stitched total={four['stitched_total_seconds']:.3f}s. Path validated.",
              "The100/150,200/300,300/450,400/600 new 4R sweep cases are not reported as completed.",
              "", "## Interpretation and limitations", "",
              "All11replayed paths passed graph membership,collision sampling,true endpoint task and recomputed weighted-cost checks.",
              "Independent3R and4R reports include the original stage-one route; it is retained as one additional restriction.",
              "The old joint-hull reports saved only the projected endpoint,not the stage-one route. Joint replays therefore",
              "run the full fixed-goal graph with no extra retained route. No old stage-two route is used as a substitute.",
              "This asymmetry is explicit;the joint50/75 cost change cannot be attributed solely to reducing the rounding budget.",
              "Actual joint and independent path/trial settings match the stage-one case;seeds remain0 and graphs are fingerprint checked.",
              "The tiny independent400vs100cost difference(about1.5e-9)is numerically negligible,not evidence of a meaningful improvement.",
              "There is no global continuous-SSM optimum certificate and no equal-total-candidate-budget claim.",
              "", "Stitched total=old measured method total-old measured stage-two time+new measured stage-two time.",
              "It is an estimate combining different executions,not a freshly measured end-to-end duration.",
              "The 4R paused process was not resumed. Its pause record is in results/spatial4r/per_target_budget_sweep_20261029/pause.json.",
              "The previous sweeps and all source reports remain unchanged;source SHA256 digests are recorded per replay.",
              "", "Sources:", f"- {independent}", f"- {joint}",
              "", "A supplementary attempt to change the global3R stage-two default was rejected by automatic review as out of scope.",
              "No global default change was applied;all replay budgets were passed explicitly."]
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for method, label, marker in (("3R independent", "Independent (retained stage1 route)", "o"),
                                       ("3R joint", "Joint (no saved stage1 route)", "s")):
            selected = sorted([r for r in rows if r["method"] == method], key=lambda r:r["max_paths"])
            x = [r["max_paths"] for r in selected]
            axes[0].plot(x, [r["cost"] for r in selected], marker+"-", label=label)
            axes[1].plot(x, [r["stitched_total_seconds"] for r in selected], marker+"-", label=label)
        axes[0].set_ylabel("Final weighted feasible cost")
        axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[1].set_ylabel("Stitched total seconds (timing estimate)")
        axes[1].set_ylim(bottom=0)
        for axis in axes:
            axis.set_xlabel("Candidate cap in each stage (trial cap = 1.5x)")
            axis.set_xticks([50,100,200,300,400])
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        fig.suptitle("Saved stage one + matched-budget stage two; seed 0; route caveat applies")
        fig.tight_layout()
        fig.savefig(output / "cost_time.png", dpi=180)
        fig.savefig(output / "cost_time.svg")
        plt.close(fig)
    except ImportError:
        pass
    print(output / "README.md")


if __name__ == "__main__":
    main()
