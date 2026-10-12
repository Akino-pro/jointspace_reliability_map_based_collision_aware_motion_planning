"""Planar 3R workflow: cached IRIS regions -> reliability partition -> GCS."""

import argparse
import json
from pathlib import Path

from planar3R_bbox_reliability_partition import (
    R1, R2, R3, branch_bounding_box, build_reliability_lookup, partition_single_branch,
)
from planar3R_collision import L, is_collision_free
from planar3R_two_point_ssm_views import compute_ssms_for_target, refine_ik_configuration


START_POINT = (0.5, 0.5)
GOAL_POINT = (1.5, 1.5)
RELIABILITY_EPSILON = 0.01
OBSTACLES = [{"type": "circle", "center": (0.75, 0.75), "radius": 0.15}]


def reliability_region_sets_for_workspace_point(x, y, seed=0):
    ssms = compute_ssms_for_target(x, y, seed=seed)
    if not ssms:
        raise ValueError(f"No SSM branch found for workspace point ({x}, {y}).")
    lookup = build_reliability_lookup(R1, R2, R3)
    return [{"label": f"branch {i + 1}",
             "regions": partition_single_branch(branch_bounding_box(ssm), lookup),
             "ssm": ssm} for i, ssm in enumerate(ssms)]


def ssm_candidate_points(x, y, obstacles, max_candidates=30, seed=0):
    candidates = []
    for ssm in compute_ssms_for_target(x, y, seed=seed):
        stride = max(1, len(ssm) // max_candidates)
        for q in ssm[::stride]:
            q = refine_ik_configuration(q, x, y)
            if q is not None and is_collision_free(q, obstacles):
                candidates.append(q)
    return candidates


def reliability_region_sets_from_source(reliability_source, start_point, goal_point, seed):
    if reliability_source == "start":
        return reliability_region_sets_for_workspace_point(*start_point, seed=seed)
    if reliability_source == "goal":
        return reliability_region_sets_for_workspace_point(*goal_point, seed=seed + 1)
    if reliability_source == "none":
        return [{"label": "no reliability partition", "regions": None, "ssm": None}]
    raise ValueError("reliability_source must be 'start', 'goal', or 'none'.")


def reliability_cost_weight(reliability, epsilon=RELIABILITY_EPSILON):
    return epsilon + 1.0 - float(reliability)


def parse_args():
    parser = argparse.ArgumentParser(description="IRIS + reliability GCS planning for planar 3R.")
    parser.add_argument("--sx", type=float, default=START_POINT[0])
    parser.add_argument("--sy", type=float, default=START_POINT[1])
    parser.add_argument("--gx", type=float, default=GOAL_POINT[0])
    parser.add_argument("--gy", type=float, default=GOAL_POINT[1])
    parser.add_argument("--iris-regions", default="planar3R_iris_regions.npz")
    parser.add_argument("--output", default="iris_path.json", help="Save waypoints and metrics as JSON")
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument("--no-rounding", action="store_true")
    parser.add_argument("--reliability-source", choices=["start", "goal", "none"], default="goal")
    parser.add_argument("--reliability-epsilon", type=float, default=RELIABILITY_EPSILON)
    parser.add_argument("--min-reliability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    from planar3R_iris_gcs_planner import plan_with_iris_gcs

    args = parse_args()
    result = plan_with_iris_gcs(
        (args.sx, args.sy), (args.gx, args.gy), OBSTACLES, args.iris_regions,
        reliability_source=args.reliability_source, min_reliability=args.min_reliability,
        seed=args.seed, candidate_limit=args.candidate_limit,
        reliability_epsilon=args.reliability_epsilon, rounding=not args.no_rounding,
    )
    print(f"safe regions: {len(result['regions'])}")
    print(f"planning seconds: {result['total_seconds']:.3f}")
    print(f"gcs cost: {result['cost']:.6f}")
    print(f"reliability branch: {result['reliability_branch']}")
    print(f"path reliability values: {result['path_reliabilities']}")
    print("joint-space waypoints:")
    for q in result["waypoints"]:
        print(f"  [{q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}]")
    report = {
        "region_source": "iris", "planner": "drake",
        "start_point": [args.sx, args.sy], "goal_point": [args.gx, args.gy],
        "link_lengths": list(L), "obstacles": OBSTACLES,
        "region_count": len(result["regions"]), "waypoints": result["waypoints"].tolist(),
        "path_reliabilities": [float(r) for r in result["path_reliabilities"]],
    }
    for name in ("cost", "statistics", "solve_seconds", "total_seconds", "collision_samples",
                 "endpoint_error", "reliability_branch", "start_candidate_count", "goal_candidate_count",
                 "iris_checkpoint", "iris_options"):
        report[name] = result[name]
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
