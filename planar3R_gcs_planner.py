import argparse
import math

import networkx as nx
import numpy as np

from planar3R_bbox_reliability_partition import (
    R1,
    R2,
    R3,
    branch_bounding_box,
    build_reliability_lookup,
    partition_single_branch,
)
from planar3R_collision import is_collision_free
from planar3R_gcs_regions import (
    adjacency_edges,
    box_to_hpolyhedron,
    build_safe_reliable_regions,
    point_in_box,
)
from planar3R_two_point_ssm_views import compute_ssms_for_target


START_POINT = (0.5, 0.5)
GOAL_POINT = (1.5, 1.5)

OBSTACLES = [
    {"type": "circle", "center": (0.75, 0.75), "radius": 0.15},
]


def reliability_regions_for_workspace_point(x, y, branch_index=0, seed=0):
    ssms = compute_ssms_for_target(x, y, seed=seed)
    if not ssms:
        raise ValueError(f"No SSM branch found for workspace point ({x}, {y}).")
    if branch_index >= len(ssms):
        raise ValueError(
            f"Requested branch {branch_index}, but only {len(ssms)} branch(es) were found."
        )

    reliability_lookup = build_reliability_lookup(R1, R2, R3)
    bounds = branch_bounding_box(ssms[branch_index])
    return partition_single_branch(bounds, reliability_lookup), ssms


def ssm_candidate_points(x, y, obstacles, max_candidates=30, seed=0):
    ssms = compute_ssms_for_target(x, y, seed=seed)
    candidates = []
    for ssm in ssms:
        stride = max(1, len(ssm) // max_candidates)
        for q in ssm[::stride]:
            if is_collision_free(q, obstacles):
                candidates.append(q)
    return candidates


def reliability_regions_from_source(reliability_source, start_point, goal_point, seed):
    if reliability_source == "start":
        reliability_regions, _ = reliability_regions_for_workspace_point(
            start_point[0], start_point[1], seed=seed
        )
    elif reliability_source == "goal":
        reliability_regions, _ = reliability_regions_for_workspace_point(
            goal_point[0], goal_point[1], seed=seed + 1
        )
    elif reliability_source == "none":
        reliability_regions = None
    else:
        raise ValueError("reliability_source must be 'start', 'goal', or 'none'.")
    return reliability_regions


def build_discrete_graph(boxes, path_length_weight=1.0, reliability_weight=1.0):
    graph = nx.DiGraph()
    for i, box in enumerate(boxes):
        graph.add_node(i)

    for i, j in adjacency_edges(boxes):
        distance = np.linalg.norm(boxes[i].center - boxes[j].center)
        reliability = max(boxes[j].reliability, 1e-9)
        penalty = -math.log(reliability)
        weight = path_length_weight * distance + reliability_weight * penalty
        graph.add_edge(i, j, weight=weight)
    return graph


def connect_candidates(graph, boxes, candidates, node_name, outgoing=True):
    connected = 0
    graph.add_node(node_name)
    for q in candidates:
        for i, box in enumerate(boxes):
            if not point_in_box(q, box):
                continue
            distance = np.linalg.norm(q - box.center)
            reliability = max(box.reliability, 1e-9)
            weight = distance - math.log(reliability)
            if outgoing:
                graph.add_edge(node_name, i, weight=weight)
            else:
                graph.add_edge(i, node_name, weight=weight)
            connected += 1
    return connected


def plan_with_discrete_fallback(
    start_point,
    goal_point,
    obstacles,
    cells_per_axis=8,
    samples_per_axis=2,
    reliability_source="goal",
    min_reliability=0.0,
    path_length_weight=1.0,
    reliability_weight=1.0,
    seed=0,
):
    reliability_regions = reliability_regions_from_source(
        reliability_source, start_point, goal_point, seed
    )

    boxes = build_safe_reliable_regions(
        cells_per_axis,
        obstacles,
        reliability_regions=reliability_regions,
        samples_per_axis=samples_per_axis,
        min_reliability=min_reliability,
    )
    graph = build_discrete_graph(boxes, path_length_weight, reliability_weight)

    start_candidates = ssm_candidate_points(
        start_point[0], start_point[1], obstacles, seed=seed
    )
    goal_candidates = ssm_candidate_points(
        goal_point[0], goal_point[1], obstacles, seed=seed + 1
    )

    source_edges = connect_candidates(graph, boxes, start_candidates, "source", outgoing=True)
    target_edges = connect_candidates(graph, boxes, goal_candidates, "target", outgoing=False)
    if source_edges == 0:
        raise ValueError("No collision-free start candidate is inside a safe box.")
    if target_edges == 0:
        raise ValueError("No collision-free goal candidate is inside a safe box.")

    node_path = nx.shortest_path(graph, "source", "target", weight="weight")
    box_path = [node for node in node_path if isinstance(node, int)]
    waypoints = np.array([boxes[i].center for i in box_path])
    path_reliabilities = [boxes[i].reliability for i in box_path]
    return {
        "boxes": boxes,
        "node_path": node_path,
        "box_path": box_path,
        "waypoints": waypoints,
        "path_reliabilities": path_reliabilities,
        "start_candidate_count": len(start_candidates),
        "goal_candidate_count": len(goal_candidates),
    }


def waypoint_reliabilities_from_boxes(waypoints, boxes):
    reliabilities = []
    for waypoint in waypoints:
        containing = [box.reliability for box in boxes if point_in_box(waypoint, box)]
        reliabilities.append(max(containing) if containing else 0.0)
    return reliabilities


def containing_box_indices(point, boxes):
    return [i for i, box in enumerate(boxes) if point_in_box(point, box, tol=1e-8)]


def plan_with_drake_gcs(
    start_point,
    goal_point,
    obstacles,
    cells_per_axis=5,
    samples_per_axis=2,
    reliability_source="goal",
    min_reliability=0.0,
    seed=0,
    candidate_limit=3,
    rounding=True,
):
    from gcs.linear import LinearGCS

    reliability_regions = reliability_regions_from_source(
        reliability_source, start_point, goal_point, seed
    )
    boxes = build_safe_reliable_regions(
        cells_per_axis,
        obstacles,
        reliability_regions=reliability_regions,
        samples_per_axis=samples_per_axis,
        min_reliability=min_reliability,
    )
    if not boxes:
        raise ValueError("No safe boxes were generated.")

    start_candidates_all = ssm_candidate_points(
        start_point[0], start_point[1], obstacles, max_candidates=30, seed=seed
    )
    goal_candidates_all = ssm_candidate_points(
        goal_point[0], goal_point[1], obstacles, max_candidates=30, seed=seed + 1
    )
    start_candidates = [
        q for q in start_candidates_all if containing_box_indices(q, boxes)
    ][:candidate_limit]
    goal_candidates = [
        q for q in goal_candidates_all if containing_box_indices(q, boxes)
    ][:candidate_limit]
    if not start_candidates:
        raise ValueError("No collision-free start candidates found.")
    if not goal_candidates:
        raise ValueError("No collision-free goal candidates found.")

    regions = build_drake_gcs_regions(boxes)
    edges = adjacency_edges(boxes)

    best = None
    failures = []
    for start_q in start_candidates:
        for goal_q in goal_candidates:
            planner = LinearGCS(regions, edges=edges)
            try:
                start_q = np.asarray(start_q).reshape(3)
                goal_q = np.asarray(goal_q).reshape(3)
                start_edges = containing_box_indices(start_q, boxes)
                goal_edges = containing_box_indices(goal_q, boxes)
                if not start_edges or not goal_edges:
                    failures.append("Candidate is not inside any safe box.")
                    continue
                planner.addSourceTarget(start_q, goal_q, edges=[start_edges, goal_edges])
                waypoints, results = planner.SolvePath(rounding=rounding, verbose=False)
            except Exception as exc:
                failures.append(str(exc))
                continue

            if waypoints is None:
                failures.append("SolvePath returned no path.")
                continue

            cost = results.get("mip_cost", np.inf)
            if best is None or cost < best["cost"]:
                best = {
                    "boxes": boxes,
                    "waypoints": waypoints.T,
                    "cost": cost,
                    "results": results,
                    "start": np.asarray(start_q).reshape(3),
                    "goal": np.asarray(goal_q).reshape(3),
                    "start_candidate_count": len(start_candidates),
                    "goal_candidate_count": len(goal_candidates),
                }

    if best is None:
        failure_summary = failures[:5]
        raise RuntimeError(f"Drake GCS failed for all candidate pairs: {failure_summary}")

    best["path_reliabilities"] = waypoint_reliabilities_from_boxes(best["waypoints"], boxes)
    return best


def build_drake_gcs_regions(boxes):
    return [box_to_hpolyhedron(box) for box in boxes]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype collision-free reliability-aware planar 3R planning."
    )
    parser.add_argument("--sx", type=float, default=START_POINT[0])
    parser.add_argument("--sy", type=float, default=START_POINT[1])
    parser.add_argument("--gx", type=float, default=GOAL_POINT[0])
    parser.add_argument("--gy", type=float, default=GOAL_POINT[1])
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--planner", choices=["fallback", "drake"], default="fallback")
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--no-rounding", action="store_true")
    parser.add_argument("--reliability-source", choices=["start", "goal", "none"], default="goal")
    parser.add_argument("--min-reliability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.planner == "drake":
        result = plan_with_drake_gcs(
            (args.sx, args.sy),
            (args.gx, args.gy),
            OBSTACLES,
            cells_per_axis=args.cells,
            samples_per_axis=args.samples,
            reliability_source=args.reliability_source,
            min_reliability=args.min_reliability,
            seed=args.seed,
            candidate_limit=args.candidate_limit,
            rounding=not args.no_rounding,
        )
    else:
        result = plan_with_discrete_fallback(
            (args.sx, args.sy),
            (args.gx, args.gy),
            OBSTACLES,
            cells_per_axis=args.cells,
            samples_per_axis=args.samples,
            reliability_source=args.reliability_source,
            min_reliability=args.min_reliability,
            seed=args.seed,
        )

    print(f"safe boxes: {len(result['boxes'])}")
    print(f"start candidates: {result['start_candidate_count']}")
    print(f"goal candidates: {result['goal_candidate_count']}")
    if "box_path" in result:
        print(f"path boxes: {len(result['box_path'])}")
    if "cost" in result:
        print(f"gcs cost: {result['cost']:.6f}")
    print(f"path reliability values: {result['path_reliabilities']}")
    print("joint-space waypoints:")
    for waypoint in result["waypoints"]:
        print(f"  [{waypoint[0]: .4f}, {waypoint[1]: .4f}, {waypoint[2]: .4f}]")


if __name__ == "__main__":
    main()
