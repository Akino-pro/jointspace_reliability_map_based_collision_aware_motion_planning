"""Fixed-start, adaptive SSM target selection followed by full GCS replanning."""

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from planar3R_collision import L, is_collision_free, link_points, wrap_to_pi
from planar3R_iris_gcs_planner import validate_joint_path
from planar3R_iris_regions import load_iris_checkpoint, polytope_adjacency, split_iris_regions
from planar3R_ssm_envelopes import build_envelopes, intersect_envelopes, project_to_local_ssm


def solve_target_sets(regions, edges, start, targets, weights, rounding=True):
    """Charge each physical segment in its departure region; sink is bookkeeping."""
    from pydrake.geometry.optimization import Point
    from pydrake.solvers import L2NormCost
    from gcs.linear import BindingCost, LinearGCS

    planner = LinearGCS([r.to_drake() for r in regions], edges=edges,
                        vertex_cost_weights=weights)
    vertices = list(planner.gcs.Vertices())
    planner.source = planner.gcs.AddVertex(Point(start), "fixed_start")
    planner.target = planner.gcs.AddVertex(Point(np.zeros(3)), "terminal_sink")
    for index, region in enumerate(regions):
        if region.contains(start):
            edge = planner.gcs.AddEdge(planner.source, vertices[index])
            for axis in range(3):
                edge.AddConstraint(edge.xu()[axis] == edge.xv()[axis])
    target_vertices = {}
    for index, target in enumerate(targets):
        vertex = planner.gcs.AddVertex(target["polytope"].to_drake(), f"local_target_{index}")
        target_vertices[vertex.name()] = index
        region_index = target["region"]
        edge = planner.gcs.AddEdge(vertices[region_index], vertex)
        matrix = weights[region_index] * np.hstack((-np.eye(3), np.eye(3)))
        edge.AddCost(BindingCost(L2NormCost(matrix, np.zeros(3)),
                                 np.append(edge.xu(), edge.xv())))
        planner.gcs.AddEdge(vertex, planner.target)
    waypoints, results = planner.SolvePath(rounding=rounding, preprocessing=True)
    if waypoints is None:
        return None
    # A target can only exit to the sink. Remove that artificial sink waypoint.
    path = results["best_path"]
    selected = target_vertices[path[-1].u().name()]
    waypoints = waypoints.T[:-1]
    # Recompute the actual feasible path cost, independently of solver reporting.
    best_result = results["best_result"]
    actual_cost = 0.0
    for edge in path:
        if edge.u().name() in planner.names:
            region_index = planner.names.index(edge.u().name())
            actual_cost += weights[region_index] * np.linalg.norm(
                best_result.GetSolution(edge.xv()) - best_result.GetSolution(edge.xu()))
    return {"waypoints": waypoints, "cost": float(actual_cost), "target": selected,
            "solver_cost": float(results["rounded_cost" if rounding else "mip_cost"])}


def plan_two_stage(start, goal, obstacles, checkpoint_path="planar3R_iris_regions.npz",
                   geometric_epsilon=0.05, samples_per_branch=300,
                   reliability_source="goal", reliability_branch=0,
                   reliability_epsilon=0.01, min_reliability=0.0, seed=0,
                   rounding=True, compare_sampled=False):
    started = time.perf_counter()
    supplied = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if (supplied.shape != (3,) or goal.shape != (2,) or not np.all(np.isfinite(supplied))
            or not np.all(np.isfinite(goal)) or not np.isfinite(reliability_epsilon)
            or reliability_epsilon <= 0 or not 0 <= min_reliability <= 1
            or not np.isfinite(geometric_epsilon) or geometric_epsilon <= 0
            or samples_per_branch < 2):
        raise ValueError("Invalid endpoints, epsilon, sample count, or reliability threshold")
    start = np.where(np.abs(supplied) > np.pi, wrap_to_pi(supplied), supplied)
    positive_boundary = ((np.abs(supplied) > np.pi) & (supplied > 0)
                         & np.isclose(start, -np.pi, atol=1e-12, rtol=0))
    start[positive_boundary] = np.pi
    offsets = supplied - start
    report = {"success": False, "planner": "two_stage_ssm_gcs", "gcs_solve_calls": 0,
              "start_configuration": supplied.tolist(), "goal_point": goal.tolist(),
              "geometric_epsilon": geometric_epsilon, "samples_per_branch": samples_per_branch,
              "reliability_epsilon": reliability_epsilon, "reliability_source": reliability_source,
              "iris_checkpoint": str(checkpoint_path), "angle_winding_offsets": offsets.tolist(),
              "continuous_coverage_certified": False, "global_optimality_certified": False}

    def finish(status, message, **values):
        report.update(status=status, message=message,
                      total_seconds=time.perf_counter() - started, **values)
        return report

    if np.linalg.norm(goal) > sum(L) + 1e-10:
        return finish("goal_unreachable", "Goal is outside radial reach; no GCS solve performed.")
    original, metadata = load_iris_checkpoint(checkpoint_path)
    model = metadata.get("model", {})
    expected_obstacles = [{"type": o.get("type", "circle"), "center": list(o["center"]),
                           "radius": o["radius"]} for o in obstacles]
    if model.get("link_lengths") != list(L) or model.get("obstacles") != expected_obstacles:
        raise ValueError("IRIS checkpoint geometry does not match the planning scene")
    link_radius = float(model["link_radius"])
    if not is_collision_free(start, obstacles, link_radius=link_radius):
        return finish("start_in_collision", "Fixed start is in collision.")
    # Imports are deferred so geometry and stop conditions work without Drake.
    from planar3R_gcs_planner import reliability_cost_weight, reliability_region_sets_from_source
    from planar3R_two_point_ssm_views import compute_ssms_for_target, refine_ik_configuration

    setup_started = time.perf_counter()
    partition_sets = reliability_region_sets_from_source(
        reliability_source, link_points(start)[-1], goal, seed)
    if not 0 <= reliability_branch < len(partition_sets):
        raise ValueError("reliability_branch is outside the available partition sets")
    partition = partition_sets[reliability_branch]
    report["reliability_branch"] = partition["label"]
    regions = split_iris_regions(original, partition["regions"], min_reliability)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    edges = polytope_adjacency(regions)
    graph.add_edges_from(edges)
    sources = [i for i, region in enumerate(regions) if region.contains(start)]
    if not sources:
        return finish("no_path_in_iris_coverage", "Fixed start lacks coverage in the selected partition.")
    keep = sorted(set().union(*(nx.node_connected_component(graph, i) for i in sources)))
    mapping = {old: new for new, old in enumerate(keep)}
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in mapping and j in mapping]
    regions = [regions[i] for i in keep]
    weights = [reliability_cost_weight(r.reliability, reliability_epsilon) for r in regions]
    branches = compute_ssms_for_target(*goal, seed=seed + 1)
    if not branches:
        return finish("ssm_sampling_failed", "No traced SSM found; this does not prove unreachability.")
    # Refine drift without deleting samples and joining previously separate portions.
    refined_branches = []
    for branch in branches:
        piece = []
        for q in branch:
            refined = refine_ik_configuration(q, *goal)
            if refined is None:
                if piece:
                    refined_branches.append(np.asarray(piece))
                    piece = []
            else:
                piece.append(refined)
        if piece:
            refined_branches.append(np.asarray(piece))
    envelopes = build_envelopes(refined_branches, geometric_epsilon, samples_per_branch)
    targets = intersect_envelopes(envelopes, regions)
    report.update(envelope_count=len(envelopes), target_set_count=len(targets),
                  actual_sample_count=len({tuple(q) for e in envelopes for q in e.samples}),
                  region_count=len(regions), edge_count=len(edges),
                  geometry_seconds=time.perf_counter() - setup_started,
                  envelope_polyline_bounds=[e.polyline_distance_bound for e in envelopes])
    if not targets:
        return finish("no_path_in_iris_coverage", "No sampled target hull intersects connected coverage.")
    stage_started = time.perf_counter()
    try:
        report["gcs_solve_calls"] += 1
        first = solve_target_sets(regions, edges, start, targets, weights, rounding)
    except ImportError as exc:
        report["gcs_solve_calls"] -= 1
        return finish("drake_unavailable", str(exc))
    except RuntimeError as exc:
        return finish("solver_failed", str(exc))
    report["stage1_seconds"] = time.perf_counter() - stage_started
    if first is None:
        return finish("solver_failed", "First-stage GCS returned no feasible rounded route.")
    target = targets[first["target"]]
    envelope = envelopes[target["envelope"]]
    p = first["waypoints"][-1]
    report.update(stage1_cost=first["cost"], selected_target=first["target"],
                  selected_envelope=target["envelope"], selected_region=target["region"],
                  selected_ssm_branch=envelope.branch, selected_chart_piece=envelope.piece,
                  stage1_endpoint=(p + offsets).tolist())
    projection_started = time.perf_counter()
    estimated = project_to_local_ssm(p, envelope, regions[target["region"]],
                                     goal, obstacles, link_radius)
    report["projection_seconds"] = time.perf_counter() - projection_started
    if estimated is None:
        return finish("projection_failed", "No valid continuous IK projection in the selected local free portion.")
    report.update(estimated_goal=(estimated + offsets).tolist(),
                  projection_distance=float(np.linalg.norm(estimated - p)))
    # Fresh graph with ALL retained free regions and adjacencies. No first route
    # is frozen and no projection segment is appended to the returned motion.
    from planar3R_ssm_envelopes import sample_hull
    fixed_targets = [{"polytope": sample_hull([estimated]), "region": i}
                     for i, region in enumerate(regions) if region.contains(estimated)]
    stage_started = time.perf_counter()
    report["gcs_solve_calls"] += 1
    try:
        second = solve_target_sets(regions, edges, start, fixed_targets, weights, rounding)
    except RuntimeError as exc:
        return finish("solver_failed", str(exc))
    report["stage2_seconds"] = time.perf_counter() - stage_started
    if second is None:
        return finish("solver_failed", "Second-stage full GCS returned no feasible route.")
    path = second["waypoints"] + offsets
    free, checked = validate_joint_path(path, obstacles, link_radius)
    endpoint_error = float(np.linalg.norm(link_points(path[-1])[-1] - goal))
    start_error = float(np.linalg.norm(path[0] - supplied))
    fixed_goal_error = float(np.linalg.norm(path[-1] - (estimated + offsets)))
    report.update(waypoints=path.tolist(), cost=second["cost"], solver_cost=second["solver_cost"],
                  collision_free=free, collision_samples=checked, endpoint_error=endpoint_error,
                  start_error=start_error, fixed_goal_error=fixed_goal_error,
                  link_lengths=list(L), obstacles=obstacles, link_radius=link_radius,
                  start_point=link_points(supplied)[-1].tolist(),
                  cost_type="region_weighted_joint_space_length")
    if not free or max(endpoint_error, start_error, fixed_goal_error) > 1e-6:
        return finish("path_validation_failed", "Independent final path check failed.")
    report["solve_seconds"] = report["stage1_seconds"] + report["stage2_seconds"]
    if compare_sampled:
        # Separate benchmark: same fixed start, regions, weights and seed; all
        # covered refined sampled endpoints compete in one graph.
        baseline_targets = []
        seen = set()
        for envelope in envelopes:
            for q in envelope.samples:
                if not is_collision_free(q, obstacles, link_radius=link_radius):
                    continue
                for i, region in enumerate(regions):
                    key = (i, tuple(q))
                    if region.contains(q) and key not in seen:
                        baseline_targets.append({"polytope": sample_hull([q]), "region": i})
                        seen.add(key)
        baseline_started = time.perf_counter()
        baseline_error = None
        try:
            baseline = solve_target_sets(regions, edges, start, baseline_targets, weights, rounding) if baseline_targets else None
        except RuntimeError as exc:
            baseline, baseline_error = None, str(exc)
        baseline_valid = False
        if baseline is not None:
            baseline_path = baseline["waypoints"] + offsets
            baseline_free, _ = validate_joint_path(baseline_path, obstacles, link_radius)
            baseline_valid = bool(baseline_free and
                np.linalg.norm(link_points(baseline_path[-1])[-1] - goal) <= 1e-6 and
                np.linalg.norm(baseline_path[0] - supplied) <= 1e-6)
        report["sampled_baseline"] = {"seconds": time.perf_counter() - baseline_started,
                                      "target_count": len(baseline_targets),
                                      "success": baseline_valid, "error": baseline_error,
                                      "cost": baseline["cost"] if baseline else None,
                                      "gcs_solve_calls": int(bool(baseline_targets)),
                                      "method": "one_graph_all_sampled_endpoints"}
    report["success"] = True
    return finish("success", "Validated second-stage path to continuous IK goal.")


def main():
    from planar3R_gcs_planner import GOAL_POINT, OBSTACLES

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--q", type=float, nargs=3, help="Fixed start configuration in radians")
    source.add_argument("--start-path", default=None, help="Read fixed start from saved path JSON")
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--gx", type=float, default=GOAL_POINT[0])
    parser.add_argument("--gy", type=float, default=GOAL_POINT[1])
    parser.add_argument("--geometric-epsilon", type=float, default=0.05)
    parser.add_argument("--ssm-samples", type=int, default=300)
    parser.add_argument("--iris-regions", default="planar3R_iris_regions.npz")
    parser.add_argument("--reliability-source", choices=["start", "goal", "none"], default="goal")
    parser.add_argument("--reliability-branch", type=int, default=0)
    parser.add_argument("--reliability-epsilon", type=float, default=0.01)
    parser.add_argument("--min-reliability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rounding", action="store_true")
    parser.add_argument("--compare-sampled", action="store_true")
    parser.add_argument("--output", default="two_stage_path.json")
    args = parser.parse_args()
    try:
        start = args.q if args.q is not None else json.loads(
            Path(args.start_path or "iris_path.json").read_text(encoding="utf-8"))["waypoints"][args.waypoint_index]
        result = plan_two_stage(start, [args.gx, args.gy], OBSTACLES, args.iris_regions,
                                args.geometric_epsilon, args.ssm_samples, args.reliability_source,
                                args.reliability_branch, args.reliability_epsilon, args.min_reliability,
                                args.seed, not args.no_rounding, args.compare_sampled)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        result = {"success": False, "status": "invalid_input", "message": str(exc)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
