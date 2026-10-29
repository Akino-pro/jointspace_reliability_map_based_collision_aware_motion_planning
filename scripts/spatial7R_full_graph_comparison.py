"""Time both 200/300 methods on the complete new-scene partition, without coverage cropping."""
import argparse
import json
from pathlib import Path
from scripts.spatial7R_midpoint_obstacle_experiment import CHECKPOINT
from scripts.spatial7R_two_stage_gcs_planner import plan_before_failure

FOLDER = Path("results/spatial7r/full_graph_60cm90deg_comparison")
FACET_FOLDER = Path("results/spatial7r/facet_graph_60cm90deg_comparison")
COMPACT_FOLDER = Path("results/spatial7r/facet_graph_60cm90deg_compact_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "per_target", "union_hulls"])
    parser.add_argument("--facet-adjacency", action="store_true")
    parser.add_argument("--compact-constraints", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.compact_constraints and not args.facet_adjacency:
        parser.error("Compact experiment requires --facet-adjacency")
    folder = COMPACT_FOLDER if args.compact_constraints else FACET_FOLDER if args.facet_adjacency else FOLDER
    if args.output_dir:
        folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"BEGIN full graph {args.mode}: each target 200/300; stage two 200/300", flush=True)
    report = plan_before_failure(
        checkpoint=CHECKPOINT, max_paths=200, max_trials=300,
        max_regions=4000, max_edges=2000000, max_targets=10000,
        geometry_only=args.mode == "prepare", union_hulls_only=args.mode == "union_hulls",
        adjacency_checkpoint=(FACET_FOLDER if args.facet_adjacency else FOLDER) / "adjacency.npz",
        min_intersection_dimension=6 if args.facet_adjacency else None,
        candidate_adjacency_checkpoint=FOLDER / "adjacency.npz" if args.facet_adjacency else None,
        compact_constraints=args.compact_constraints)
    (folder / f"{args.mode}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("END "+json.dumps({k: report.get(k) for k in
          ("success", "status", "message", "region_count", "edge_count", "target_set_count",
           "cost", "geometry_seconds", "stage1_seconds", "stage2_seconds", "total_seconds")}), flush=True)
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
