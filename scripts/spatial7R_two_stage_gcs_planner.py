"""Seven-dimensional IRIS/reliability/two-stage GCS for a full-pose 7R task."""

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from scripts import spatial7R_model as model
from scripts.spatial7R_iris import generate_regions, scene_metadata
from scripts.planar3R_bbox_reliability_partition import merge_equal_reliability_boxes, aggregate_branch_recovery, RELIABILITY_RULE
from scripts.planar3R_iris_regions import load_iris_checkpoint, split_iris_regions, polytope_adjacency
from scripts.planar3R_ssm_envelopes import build_envelopes, intersect_envelopes, project_to_local_ssm, sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
from scripts.spatial4R_post_failure_planner import chart_configuration
from gcs.quality import bound_quality
from scripts.goal_set_baselines import baseline_seconds, compare_goal_sets

GEOMETRIC_EPSILON = .025
SSM_ARC_SPACING = .044
RELIABILITY_EPSILON = .01
MAX_ROUNDED_PATHS = 200
MAX_ROUNDING_TRIALS = 300
MAX_PARTITION_CELLS = 20000
MAX_FREE_REGIONS = 160
MAX_GRAPH_EDGES = 1500
MAX_TARGETS = 128


class ResourceLimit(RuntimeError):
    pass


def path_is_in_graph(solved, regions, start, target, tolerance=1e-7):
    points = np.asarray(solved["waypoints"])
    route = solved["region_route"]
    return bool(points.shape == (len(route)+1, 7) and np.all(np.isfinite(points))
                and np.linalg.norm(points[0]-start) <= tolerance
                and target.contains(points[-1], tol=tolerance)
                and all(regions[index].contains(points[i], tol=tolerance)
                        and regions[index].contains(points[i+1], tol=tolerance)
                        for i, index in enumerate(route)))


def region_bounds(region):
    """Exact LP coordinate extrema used only to restrict subdivision, not collision geometry."""
    dimension = region.A.shape[1]
    bounds = []
    for axis in np.eye(dimension):
        low = linprog(axis, A_ub=region.A, b_ub=region.b, bounds=[(None, None)]*dimension, method="highs")
        high = linprog(-axis, A_ub=region.A, b_ub=region.b, bounds=[(None, None)]*dimension, method="highs")
        if not low.success or not high.success:
            raise ValueError("Expected a nonempty bounded IRIS polytope")
        # Expand outward slightly: LP roundoff must not trim an original region.
        bounds.append([low.fun-1e-9, -high.fun+1e-9])
    return np.asarray(bounds)


def local_reliability_regions(originals, branches, max_cells=MAX_PARTITION_CELLS):
    """Exact common refinement inside each IRIS AABB, unioning branch recovery sets.

    This equals restricting the global bounding-box partition to IRIS coverage,
    without constructing its enormous out-of-coverage Cartesian grid. Equal-cost
    cells merge only when their union is exactly a box. The final intersection
    retains every original IRIS halfspace.
    """
    if not 1 <= len(branches) <= 8 or max_cells < 1:
        raise ValueError("Expected one to eight branches and a positive cell budget")
    bounds = np.array([np.column_stack((b.min(axis=0), b.max(axis=0))) for b in branches])
    lookup = model.reliability_lookup()
    regions, raw_count, merged_count = [], 0, 0
    for parent_index, parent in enumerate(originals):
        domain = region_bounds(parent)
        cuts = [np.unique(np.r_[lo, bounds[:, axis, :].flatten()[
            (bounds[:, axis, :].flatten() > lo) & (bounds[:, axis, :].flatten() < hi)], hi])
                for axis, (lo, hi) in enumerate(domain)]
        count = math.prod(len(axis)-1 for axis in cuts)
        if raw_count+count > max_cells:
            raise ResourceLimit(f"Exact local reliability refinement needs {raw_count+count} cells; limit is {max_cells}.")
        raw_count += count
        cells = []
        for intervals in itertools.product(*(range(len(axis)-1) for axis in cuts)):
            lower = np.array([cuts[j][i] for j, i in enumerate(intervals)])
            upper = np.array([cuts[j][i+1] for j, i in enumerate(intervals)])
            point = (lower+upper)/2
            f_sets = tuple(frozenset(j+1 for j in range(7) if b[j, 0] <= point[j] <= b[j, 1]) for b in bounds)
            cells.append(dict(lower=lower, upper=upper,
                              **aggregate_branch_recovery(f_sets, lookup)))
        merged = merge_equal_reliability_boxes(cells)
        merged_count += len(merged)
        children = split_iris_regions([parent], merged)
        for child in children:
            child.parent_index = parent_index
        regions.extend(children)
    return regions, dict(local_partition_cell_count=raw_count, merged_partition_cell_count=merged_count,
                          reliability_rule=RELIABILITY_RULE,
                          reliability_partition_scope="exact_refinement_restricted_to_iris_coverage")


def plan_before_failure(start=model.START_CONFIGURATION, goal_pose=None,
                        checkpoint="data/iris/spatial7R_30cm20deg_iris_regions.npz",
                        obstacles=None, generate_iris=False, geometry_only=False,
                        reference=model.GOAL_REFERENCE_CONFIGURATION, iris_radius=.16,
                        geometric_epsilon=GEOMETRIC_EPSILON, arc_spacing=SSM_ARC_SPACING,
                        max_paths=MAX_ROUNDED_PATHS, max_trials=MAX_ROUNDING_TRIALS,
                        ssm_attempts=model.MAX_SSM_IK_ATTEMPTS, ssm_seed=1,
                        max_partition_cells=MAX_PARTITION_CELLS, max_regions=MAX_FREE_REGIONS,
                        max_edges=MAX_GRAPH_EDGES, max_targets=MAX_TARGETS, solver_backend="mosek",
                        compare_sampled=False, compare_union_hulls=False,
                        compare_nonlinear_goal=False, nonlinear_solver="snopt"):
    began = time.perf_counter()
    supplied = np.asarray(start, dtype=float)
    goal = model.validate_pose(model.pose(reference) if goal_pose is None else goal_pose)
    if (supplied.shape != (7,) or not np.all(np.isfinite(supplied))
            or not np.isfinite(geometric_epsilon) or geometric_epsilon <= 0
            or not np.isfinite(arc_spacing) or arc_spacing <= 0
            or any(not isinstance(n, (int, np.integer)) or n < 1 for n in
                   (max_paths, max_trials, ssm_attempts, max_partition_cells, max_regions, max_edges, max_targets))):
        raise ValueError("Use finite seven-angle input, valid goal pose, positive resolution and integer budgets")
    start, offsets = chart_configuration(supplied)
    start_pose = model.pose(start)
    task_delta = model.pose_residual(start, goal)
    obstacles = model.default_obstacles(start, goal) if obstacles is None else obstacles
    report = dict(success=False, planner="spatial7R_two_stage_gcs", robot=model.PARAMETERS,
                  task="full_pose_SE3", task_dimension=6, ssm_regular_dimension=1, ssm_branch_limit=8,
                  start_configuration=supplied.tolist(), start_point=model.position(start).tolist(),
                  start_pose=start_pose.tolist(),
                  task_position_distance=float(np.linalg.norm(task_delta[:3])),
                  task_orientation_difference_degrees=float(np.rad2deg(np.linalg.norm(task_delta[3:]))),
                  goal_point=goal[:3, 3].tolist(), goal_pose=goal.tolist(),
                  obstacles=obstacles, link_radius=model.LINK_RADIUS, physical_joint_limits=None,
                  iris_checkpoint=str(checkpoint), geometric_epsilon=geometric_epsilon,
                  ssm_arc_spacing=arc_spacing, reliability_epsilon=RELIABILITY_EPSILON,
                  gcs_solve_calls=0, coverage_is_complete=False, continuous_coverage_certified=False,
                  global_optimality_certified=False,
                  solver_settings=dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials),
                  resource_limits=dict(partition_cells=max_partition_cells, regions=max_regions,
                                       directed_edges=max_edges, targets=max_targets))
    def finish(status, message, **values):
        report.update(status=status, message=message, total_seconds=time.perf_counter()-began, **values)
        report["two_stage_total_seconds"] = report["total_seconds"]-baseline_seconds(report)
        return report
    if np.linalg.norm(goal[:3, 3]) > np.hypot(model.A, model.D).sum()+1e-9:
        return finish("goal_unreachable", "Position exceeds the conservative radial reach bound; stopped.")
    if not model.collision_free(start, obstacles):
        return finish("start_in_collision", "Initial configuration collides; stopped before solving.")
    setup = time.perf_counter()
    if generate_iris:
        # A midpoint seed can cover both nearby endpoints without overlapping
        # endpoint regions multiplying the 7D perspective-conic graph.
        seeds = [(start+np.asarray(reference))/2, start, np.asarray(reference)]
        originals, metadata = generate_regions(checkpoint, obstacles, seeds, coverage_radius=iris_radius)
    else:
        originals, metadata = load_iris_checkpoint(checkpoint, expected_dimension=7)
    if metadata["model"] != scene_metadata(obstacles):
        raise ValueError("IRIS checkpoint does not match the supplied seven-joint DH model and scene")
    report.update(original_region_count=len(originals), iris_generation_seconds=metadata["generation_seconds"],
                  iris_coverage_radius=metadata.get("coverage_radius"))
    branches, trace = model.compute_ssms(goal, reference=reference, attempts=ssm_attempts, seed=ssm_seed)
    report.update(ssm_branch_count=len(branches), ssm_trace=trace)
    if not branches:
        return finish("ssm_sampling_failed", "No regular closed full-pose SSM found; no unreachability proof.")
    try:
        regions, partition_report = local_reliability_regions(originals, branches, max_partition_cells)
        report.update(partition_report, region_count=len(regions))
        if len(regions) > max_regions:
            raise ResourceLimit(f"Reliability intersections produced {len(regions)} regions; limit is {max_regions}.")
        adjacency_statistics = {}
        edges = polytope_adjacency(regions, statistics=adjacency_statistics)
        report["adjacency_statistics"] = adjacency_statistics
        report["edge_count"] = len(edges)
        if len(edges) > max_edges:
            raise ResourceLimit(f"Graph has {len(edges)} directed edges; limit is {max_edges}.")
    except ResourceLimit as exc:
        return finish("resource_limit", str(exc))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    sources = [i for i, r in enumerate(regions) if r.contains(start)]
    if not sources:
        return finish("no_path_in_iris_coverage", "The fixed start has no coverage; stopped before GCS.")
    keep = sorted(set().union(*(nx.node_connected_component(graph, i) for i in sources)))
    mapping = {old: new for new, old in enumerate(keep)}
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in mapping and j in mapping]
    regions = [regions[i] for i in keep]
    envelopes = build_envelopes(branches, geometric_epsilon, arc_spacing=arc_spacing,
                               refine_sample=lambda q: model.refine_ik(q, goal), sampling_report=report)
    targets = intersect_envelopes(envelopes, regions)
    report.update(region_count=len(regions), edge_count=len(edges), envelope_count=len(envelopes),
                  target_set_count=len(targets), actual_sample_count=sum(report["ssm_branch_sample_counts"]),
                  geometry_seconds=time.perf_counter()-setup,
                  ssm_branch_summaries=[dict(branch=i, arc_length=report["ssm_branch_arc_lengths"][i],
                                             sample_count=report["ssm_branch_sample_counts"][i],
                                             envelope_count=sum(e.branch == i for e in envelopes),
                                             target_set_count=sum(envelopes[t["envelope"]].branch == i for t in targets))
                                        for i in range(len(branches))])
    print(f"7R geometry: {len(branches)} branches, {len(regions)} regions, {len(edges)} edges, "
          f"{report['actual_sample_count']} samples, {len(targets)} targets", flush=True)
    if not targets:
        return finish("no_path_in_iris_coverage", "No full-pose target hull intersects connected coverage.")
    if len(targets) > max_targets:
        return finish("resource_limit", f"Found {len(targets)} targets; limit is {max_targets}.")
    if geometry_only:
        return finish("geometry_only", "Full-pose preprocessing complete; no GCS solve.", success=True)
    weights = [RELIABILITY_EPSILON+1-r.reliability for r in regions]
    settings = dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials, solver_backend=solver_backend)
    if compare_sampled:
        from scripts.sampled_goal_baseline import compare_sampled_goals
        report["sampled_baseline"] = compare_sampled_goals(
            regions, edges, start, envelopes, weights, settings,
            lambda q: model.collision_free(q, obstacles),
            lambda p: model.validate_path(p, obstacles),
            lambda q: np.linalg.norm(model.pose_residual(q, goal)))
    if compare_union_hulls or compare_nonlinear_goal:
        task = dict(id=dict(type="full_pose_SE3", goal=goal.tolist()), dimension=6,
                    residual=lambda q: model.pose_residual(q, goal),
                    jacobian=lambda q: model.residual_jacobian(q, goal))
        report.update(compare_goal_sets(
            regions, edges, start, targets, envelopes, weights, settings, settings, task,
            lambda p, e, r: project_to_local_ssm(
                p, e, r, np.zeros(6), obstacles, model.LINK_RADIUS,
                position=lambda q: model.pose_residual(q, goal),
                collision_free=lambda q: model.collision_free(q, obstacles)),
            lambda p: model.validate_path(p, obstacles), offsets,
            union_hulls=compare_union_hulls, nonlinear_goal=compare_nonlinear_goal,
            nonlinear_solver=nonlinear_solver))
    first_began = time.perf_counter()
    candidates, best = [], None
    for index, target in enumerate(targets):
        print(f"7R stage 1: target {index+1}/{len(targets)}", flush=True)
        error = None
        try:
            solved = solve_target_sets(regions, edges, start, [target], weights, **settings)
        except RuntimeError as exc:
            solved, error = None, str(exc)
        report["gcs_solve_calls"] += 1
        if solved and not path_is_in_graph(solved, regions, start, target["polytope"]):
            error = "Returned path violates source, target or region constraints"
            solved = None
        item = dict(target=index, branch=envelopes[target["envelope"]].branch, success=solved is not None,
                    error=error if error else (None if solved else "No feasible rounded path returned"))
        if solved:
            item.update(cost=solved["cost"], quality=solved["quality"], rounded_path_count=solved["rounded_path_count"])
            if best is None or solved["cost"] < best["cost"]:
                best = dict(solved, selected_target=index)
        candidates.append(item)
    report.update(stage1_seconds=time.perf_counter()-first_began, stage1_candidates=candidates)
    if best is None:
        return finish("stage1_solver_failed", "No feasible first-stage path returned; stopped.")
    lower = [c.get("quality", {}).get("lower_bound") for c in candidates]
    report["stage1_quality"] = bound_quality(min(lower) if all(v is not None for v in lower) else None,
                                            best["cost"], scope="sampled_full_pose_target_intersections_graph")
    target = targets[best["selected_target"]]
    envelope, region = envelopes[target["envelope"]], regions[target["region"]]
    projected = project_to_local_ssm(best["waypoints"][-1], envelope, region, np.zeros(6),
                                      obstacles, model.LINK_RADIUS,
                                      position=lambda q: model.pose_residual(q, goal),
                                      collision_free=lambda q: model.collision_free(q, obstacles))
    report.update(stage1_cost=best["cost"], selected_target=best["selected_target"],
                  selected_ssm_branch=envelope.branch, stage1_region_route=best["region_route"])
    if projected is None:
        return finish("projection_failed", "No feasible local full-pose IK projection; stopped.")
    report["estimated_goal"] = (projected+offsets).tolist()
    distance = float(np.linalg.norm(projected-best["waypoints"][-1]))
    baseline = np.vstack((best["waypoints"], projected))
    baseline_free, baseline_samples = model.validate_path(baseline, obstacles)
    report.update(projection_distance=distance, projection_connection_baseline=dict(
        method="comparison_only_stage1_path_plus_projection_segment",
        cost=best["cost"]+weights[target["region"]]*distance,
        connector_length=distance, connector_weight=weights[target["region"]],
        collision_free=baseline_free, collision_samples=baseline_samples,
        waypoints=(baseline+offsets).tolist()))
    indices = [i for i, r in enumerate(regions) if r.contains(projected)]
    second_began = time.perf_counter()
    second = solve_target_sets(regions, edges, start,
                               [dict(polytope=sample_hull([projected]), region=indices[0], regions=indices)],
                               weights, candidate_region_routes=[best["region_route"]], **settings)
    report.update(stage2_seconds=time.perf_counter()-second_began, gcs_solve_calls=report["gcs_solve_calls"]+1)
    if second is None:
        return finish("stage2_solver_failed", "No fixed-goal GCS path returned; stopped.")
    path = second["waypoints"]+offsets
    free, checked = model.validate_path(path, obstacles)
    error = model.pose_residual(path[-1], goal)
    position_error, orientation_error = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    start_error = float(np.linalg.norm(path[0]-supplied))
    fixed_error = float(np.linalg.norm(path[-1]-(projected+offsets)))
    graph_valid = path_is_in_graph(second, regions, start, sample_hull([projected]))
    valid = free and graph_valid and max(position_error, orientation_error, start_error, fixed_error) <= 1e-6
    report.update(waypoints=path.tolist(), cost=second["cost"], collision_free=free, collision_samples=checked,
                  endpoint_error=position_error, orientation_error_radians=orientation_error,
                  start_error=start_error, fixed_goal_error=fixed_error,
                  graph_path_valid=graph_valid,
                  stage2_quality=second["quality"], stage2_native_cost=second["native_cost"],
                  stage2_solution_source=second["solution_source"], stage2_region_route=second["region_route"],
                  stage2_additional_route_results=second["additional_route_results"],
                  stage2_rounded_path_count=second["rounded_path_count"], solver_name=second["solver_name"])
    return finish("success" if valid else "path_validation_failed",
                  "Validated collision-free seven-angle path and full goal pose." if valid else "Path validation failed.",
                  success=bool(valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, nargs=7, default=model.START_CONFIGURATION.tolist())
    parser.add_argument("--goal-position", type=float, nargs=3)
    parser.add_argument("--goal-quaternion", type=float, nargs=4, help="Goal orientation quaternion x y z w")
    parser.add_argument("--iris-regions", default="data/iris/spatial7R_30cm20deg_iris_regions.npz")
    parser.add_argument("--generate-iris", action="store_true")
    parser.add_argument("--iris-radius", type=float, default=.16)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--compare-sampled", action="store_true")
    parser.add_argument("--compare-union-hulls", action="store_true")
    parser.add_argument("--compare-nonlinear-goal", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--nonlinear-solver", choices=["snopt", "ipopt"], default="snopt")
    parser.add_argument("--ssm-ik-attempts", type=int, default=model.MAX_SSM_IK_ATTEMPTS)
    parser.add_argument("--ssm-seed", type=int, default=1)
    parser.add_argument("--geometric-epsilon", type=float, default=GEOMETRIC_EPSILON)
    parser.add_argument("--ssm-arc-spacing", type=float, default=SSM_ARC_SPACING)
    parser.add_argument("--max-rounded-paths", type=int, default=MAX_ROUNDED_PATHS)
    parser.add_argument("--max-rounding-trials", type=int, default=MAX_ROUNDING_TRIALS)
    parser.add_argument("--solver", choices=["mosek", "clarabel"], default="mosek")
    parser.add_argument("--output", default="results/spatial7r/spatial7R_30cm20deg_two_stage_path.json")
    args = parser.parse_args()
    if (args.goal_position is None) != (args.goal_quaternion is None):
        parser.error("Full-pose planning requires both --goal-position and --goal-quaternion")
    try:
        goal = None
        if args.goal_position is not None:
            goal = np.eye(4)
            goal[:3, 3] = args.goal_position
            goal[:3, :3] = Rotation.from_quat(args.goal_quaternion).as_matrix()
        result = plan_before_failure(args.q, goal, args.iris_regions, generate_iris=args.generate_iris,
                                      geometry_only=args.geometry_only, iris_radius=args.iris_radius,
                                      geometric_epsilon=args.geometric_epsilon, arc_spacing=args.ssm_arc_spacing,
                                      max_paths=args.max_rounded_paths, max_trials=args.max_rounding_trials,
                                      ssm_attempts=args.ssm_ik_attempts, ssm_seed=args.ssm_seed, solver_backend=args.solver,
                                      compare_sampled=args.compare_sampled or args.compare_all,
                                      compare_union_hulls=args.compare_union_hulls or args.compare_all,
                                      compare_nonlinear_goal=args.compare_nonlinear_goal or args.compare_all,
                                      nonlinear_solver=args.nonlinear_solver)
    except (ValueError, OSError, RuntimeError) as exc:
        result = dict(success=False, status="planning_error", message=str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
