"""Saved-input 4R intersection representative ablation; no per-target GCS runs."""
import csv
import json
from pathlib import Path

from scripts.spatial4R_two_stage_gcs_planner import plan_before_failure


def main():
    folder = Path("results/spatial4r/intersection_slp_20261030")
    folder.mkdir(parents=True, exist_ok=True)
    reference_path = Path("results/spatial4r/per_target_matched_stage_budget_sweep_20261029/paths200_trials300.json")
    reference = json.loads(reference_path.read_text())
    rows = []

    def add(report, path, name):
        if (report["graph_task_sha256"] != reference["graph_task_sha256"] or
                report["hull_targets_sha256"] != reference["hull_targets_sha256"]):
            raise RuntimeError("Input geometry differs from the saved reference")
        answer = report["union_hull_baseline"]
        pruning = report["representative_pruning"]
        representative = pruning["representative"]
        optimization = representative["optimization"]
        rows.append(dict(name=name, alpha=pruning["alpha"], slp_iterations=optimization.get("iterations", 0),
                         slp_status=optimization["status"], initial_collective_cost=optimization["initial_cost"],
                         final_collective_cost=optimization["final_cost"], representative_vertices=representative["intersection_vertices"],
                         representative_undirected_edges=representative["representative_undirected_edges"],
                         offline_cache_hit=representative["cache_hit"], offline_seconds=representative["seconds"],
                         offline_optimization_seconds=optimization["seconds"],
                         pruning_seconds=pruning["seconds"], regions=report["region_count"], edges=report["edge_count"],
                         success=answer["success"], final_cost=answer.get("cost"), total_seconds=report["total_seconds"],
                         stage1_seconds=answer.get("stage1_seconds"), stage2_seconds=answer.get("stage2_seconds"),
                         task_error=answer.get("task_error"), collision_free=answer.get("collision_free"),
                         graph_valid=answer.get("graph_membership_valid"), cost_valid=answer.get("cost_valid"), source=str(path)))
        (folder/"summary.json").write_text(json.dumps(dict(first_method_rerun=False,
            reference=str(reference_path), reference_cost=reference["cost"], reference_seconds=reference["two_stage_total_seconds"],
            trials_per_target=2000, gcs_budget_both_stages=[2000, 3000], seed=0, runs=rows), indent=2))
        with (folder/"summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("END "+json.dumps(rows[-1]), flush=True)

    original = Path("results/spatial4r/union_hull_intersection_slp_alpha110_20261030.json")
    add(json.loads(original.read_text()), original, "intersection_slp_alpha110_cold")
    for name, alpha, iterations in (("intersection_centers_alpha110", 1.1, 0),
                                     ("intersection_slp_alpha102", 1.02, 80),
                                     ("intersection_slp_alpha150", 1.5, 80)):
        print("BEGIN "+name, flush=True)
        report = plan_before_failure(union_hulls_only=True, representative_pruning=True,
            pruning_method="intersection-slp", pruning_alpha=alpha, pruning_trials=2000,
            slp_iterations=iterations, slp_cache=f"data/representative/spatial4r_intersection_slp_{iterations}.npz",
            max_paths=2000, max_trials=3000, stage2_max_paths=2000, stage2_max_trials=3000)
        path = folder/f"{name}.json"
        path.write_text(json.dumps(report, indent=2))
        add(report, path, name)


if __name__ == "__main__":
    main()
