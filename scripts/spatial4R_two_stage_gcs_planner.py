"""Spatial 4R position-task planning: IRIS, reliability, SSM targets, two GCS stages."""

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from scripts import spatial4R_model as model
from scripts.spatial4R_iris import generate_regions, scene_metadata
from scripts.planar3R_bbox_reliability_partition import partition_multiple_branches, merge_equal_reliability_boxes, RELIABILITY_RULE
from scripts.planar3R_iris_regions import load_iris_checkpoint, split_iris_regions, polytope_adjacency
from scripts.planar3R_ssm_envelopes import SSM_ARC_SPACING, build_envelopes, intersect_envelopes, project_to_local_ssm, sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
from gcs.quality import bound_quality
from scripts.goal_set_baselines import (
    baseline_seconds, compare_goal_sets, graph_task_fingerprint, hull_targets_fingerprint,
)

GEOMETRIC_EPSILON = 0.025
RELIABILITY_EPSILON = 0.01
MAX_ROUNDED_PATHS = 200
MAX_ROUNDING_TRIALS = 300


def reliability_regions_for_branches(branches):
    """Same common refinement/recovery-set union as planar 3R, in four dimensions."""
    if not 1 <= len(branches) <= model.MAX_SSM_BRANCHES:
        raise ValueError("Expected one to four spatial 4R SSM branches")
    bounds = [np.column_stack((b.min(axis=0), b.max(axis=0))) for b in branches]
    boxes = partition_multiple_branches(bounds, model.reliability_lookup())
    return boxes, merge_equal_reliability_boxes(boxes)


def plan_before_failure(start=model.START_CONFIGURATION, goal=None,
                        checkpoint=None, obstacles=None,
                        geometric_epsilon=GEOMETRIC_EPSILON, arc_spacing=SSM_ARC_SPACING,
                        max_paths=MAX_ROUNDED_PATHS, max_trials=MAX_ROUNDING_TRIALS,
                        geometry_only=False, generate_iris=False, reference=model.GOAL_REFERENCE_CONFIGURATION,
                        solver_backend="mosek", iris_radius=0.9, iris_seeds=None,
                        ssm_ik_attempts=model.MAX_SSM_IK_ATTEMPTS, ssm_seed=1, compare_sampled=False,
                        compare_union_hulls=False, compare_nonlinear_goal=False, nonlinear_solver="snopt",
                        stage2_max_paths=MAX_ROUNDED_PATHS, stage2_max_trials=MAX_ROUNDING_TRIALS):
    began = time.perf_counter()
    checkpoint = checkpoint or f"data/iris/spatial4R_{model.ROBOT_PARAMETER_SET}_iris_midpoint_regions.npz"
    supplied = np.asarray(start, dtype=float)
    goal = model.position(reference) if goal is None else np.asarray(goal, dtype=float)
    stage2_max_paths = max_paths if stage2_max_paths is None else stage2_max_paths
    stage2_max_trials = max_trials if stage2_max_trials is None else stage2_max_trials
    if (supplied.shape != (4,) or goal.shape != (3,) or not np.all(np.isfinite(supplied))
            or not np.all(np.isfinite(goal)) or geometric_epsilon <= 0 or arc_spacing <= 0
            or not np.isfinite(geometric_epsilon) or not np.isfinite(arc_spacing)
            or max_paths < 1 or max_trials < 1 or stage2_max_paths < 1 or stage2_max_trials < 1
            or not isinstance(ssm_ik_attempts, (int, np.integer)) or ssm_ik_attempts < 1):
        raise ValueError("Use a finite 4D configuration, 3D goal, positive resolution and budgets")
    outside = np.abs(supplied) > np.pi
    start = np.where(outside, model.wrap(supplied), supplied)
    positive_boundary = outside & (supplied > 0) & np.isclose(start, -np.pi, atol=1e-12, rtol=0)
    start[positive_boundary] = np.pi
    offsets = supplied-start
    obstacles = model.midpoint_obstacles(start, goal) if obstacles is None else obstacles
    report = dict(success=False, planner="spatial4R_two_stage_gcs", robot=model.PARAMETERS,
                  robot_parameter_set=model.ROBOT_PARAMETER_SET,
                  position_task_only=True, physical_joint_limits=None,
                  start_configuration=supplied.tolist(), start_point=model.position(start).tolist(),
                  goal_point=goal.tolist(), obstacles=obstacles, link_radius=model.LINK_RADIUS,
                  iris_checkpoint=str(checkpoint), geometric_epsilon=geometric_epsilon,
                  ssm_arc_spacing=arc_spacing, reliability_epsilon=RELIABILITY_EPSILON,
                  solver_settings=dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials,
                                       rounding_seed=0), gcs_solve_calls=0,
                  continuous_coverage_certified=False, global_optimality_certified=False)

    def finish(status, message, **values):
        report.update(status=status, message=message, total_seconds=time.perf_counter()-began, **values)
        report["two_stage_total_seconds"] = report["total_seconds"]-baseline_seconds(report)
        return report

    if np.linalg.norm(goal) > sum(np.hypot(model.A, model.D)) + 1e-9:
        return finish("goal_unreachable", "Goal exceeds a conservative radial reach bound; stopped.")
    if not model.collision_free(start, obstacles):
        return finish("start_in_collision", "Initial spatial configuration is in collision; stopped.")
    setup = time.perf_counter()
    if generate_iris:
        if reference is None:
            raise ValueError("IRIS generation requires explicit collision-free corridor seeds")
        seeds = ([start, np.asarray(reference)] + [(1-t)*start+t*np.asarray(reference) for t in (0.25, 0.5, 0.75)]
                 if iris_seeds is None else np.asarray(iris_seeds, dtype=float))
        seeds = np.asarray(seeds, dtype=float)
        if (seeds.ndim != 2 or seeds.shape[1] != 4 or not len(seeds)
                or not np.all(np.isfinite(seeds)) or np.any(np.abs(seeds) > np.pi)):
            raise ValueError("IRIS seeds must be finite four-angle configurations in the coordinate chart")
        report["iris_seed_configurations"] = seeds.tolist()
        originals, metadata = generate_regions(checkpoint, obstacles, seeds=seeds, coverage_radius=iris_radius)
    else:
        originals, metadata = load_iris_checkpoint(checkpoint, expected_dimension=4)
    if metadata["model"] != scene_metadata(obstacles):
        raise ValueError("Spatial checkpoint does not match DH parameters and collision scene")
    branches, trace_report = model.compute_ssms(goal, reference=reference,
                                               attempts=ssm_ik_attempts, seed=ssm_seed)
    report.update(ssm_branch_count=len(branches), ssm_trace=trace_report,
                  original_region_count=len(originals), iris_generation_seconds=metadata["generation_seconds"],
                  iris_coverage_radius=metadata.get("coverage_radius"),
                  coverage_is_complete=False)
    if not branches:
        return finish("ssm_sampling_failed", "No complete SSM trace found; this does not prove unreachability.")
    boxes, merged = reliability_regions_for_branches(branches)
    regions = split_iris_regions(originals, merged)
    adjacency_statistics = {}
    edges = polytope_adjacency(regions, statistics=adjacency_statistics)
    report["adjacency_statistics"] = adjacency_statistics
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    sources = [i for i, region in enumerate(regions) if region.contains(start)]
    if not sources:
        return finish("no_path_in_iris_coverage", "Initial configuration has no IRIS coverage.")
    keep = sorted(set().union(*(nx.node_connected_component(graph, i) for i in sources)))
    mapping = {old: new for new, old in enumerate(keep)}
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in mapping and j in mapping]
    regions = [regions[i] for i in keep]
    envelopes = build_envelopes(branches, geometric_epsilon, arc_spacing=arc_spacing,
                               refine_sample=lambda q: model.refine_ik(q, goal), sampling_report=report)
    targets = intersect_envelopes(envelopes, regions)
    weights = [RELIABILITY_EPSILON + 1-r.reliability for r in regions]
    report.update(graph_task_sha256=graph_task_fingerprint(
        regions, edges, start, weights, dict(type="position_3D", goal=goal.tolist())),
        hull_targets_sha256=hull_targets_fingerprint(targets))
    report["ssm_branch_summaries"] = [dict(
        branch=i, bounding_box=np.column_stack((b.min(axis=0), b.max(axis=0))).tolist(),
        arc_length=report["ssm_branch_arc_lengths"][i],
        sample_count=report["ssm_branch_sample_counts"][i],
        envelope_count=sum(e.branch == i for e in envelopes),
        target_set_count=sum(envelopes[t["envelope"]].branch == i for t in targets))
        for i, b in enumerate(branches)]
    report.update(unmerged_reliability_partition_count=len(boxes), reliability_partition_count=len(merged),
                  reliability_rule=RELIABILITY_RULE, region_count=len(regions), edge_count=len(edges),
                  envelope_count=len(envelopes), target_set_count=len(targets),
                  actual_sample_count=len({tuple(q) for e in envelopes for q in e.samples}),
                  geometry_seconds=time.perf_counter()-setup)
    print(f"Spatial geometry: {len(branches)} SSM branches; {len(boxes)} -> {len(merged)} boxes; "
          f"{len(regions)} free regions, {len(edges)} edges; {report['actual_sample_count']} samples; "
          f"{len(targets)} targets", flush=True)
    if not targets:
        return finish("no_path_in_iris_coverage", "No sampled target hull intersects connected IRIS coverage.")
    if geometry_only:
        return finish("geometry_only", "Spatial preprocessing complete; no GCS solve performed.", success=True)
    settings = dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials, solver_backend=solver_backend)
    second_settings = dict(settings, max_rounded_paths=stage2_max_paths, max_rounding_trials=stage2_max_trials)
    report.update(stage1_solver_settings=settings, stage2_solver_settings=second_settings,
                  stage1_budget=dict(allocation="fixed_per_target", target_count=len(targets),
                                     total_max_rounded_paths=max_paths*len(targets),
                                     total_max_rounding_trials=max_trials*len(targets)))
    if compare_sampled:
        from scripts.sampled_goal_baseline import compare_sampled_goals
        report["sampled_baseline"] = compare_sampled_goals(
            regions, edges, start, envelopes, weights, settings,
            lambda q: model.collision_free(q, obstacles),
            lambda p: model.validate_path(p, obstacles),
            lambda q: np.linalg.norm(model.position(q)-goal))
    if compare_union_hulls or compare_nonlinear_goal:
        task = dict(id=dict(type="position_3D", goal=goal.tolist()), dimension=3,
                    residual=lambda q: model.position(q)-goal, jacobian=model.jacobian)
        report.update(compare_goal_sets(
            regions, edges, start, targets, envelopes, weights, settings, second_settings, task,
            lambda p, e, r: project_to_local_ssm(
                p, e, r, goal, obstacles, model.LINK_RADIUS,
                position=model.position, collision_free=lambda q: model.collision_free(q, obstacles)),
            lambda p: model.validate_path(p, obstacles), offsets,
            union_hulls=compare_union_hulls, nonlinear_goal=compare_nonlinear_goal,
            nonlinear_solver=nonlinear_solver))
    candidates, best = [], None
    stage1_began = time.perf_counter()
    for index, target in enumerate(targets):
        candidate_began = time.perf_counter()
        print(f"Spatial stage 1: target {index+1}/{len(targets)}", flush=True)
        error = None
        diagnostics = {}
        try:
            solved = solve_target_sets(regions, edges, start, [target], weights,
                                       diagnostics=diagnostics, **settings)
        except RuntimeError as exc:
            solved, error = None, str(exc)
        item = dict(target=index, envelope=target["envelope"], region=target["region"],
                    ssm_branch=envelopes[target["envelope"]].branch, success=solved is not None,
                    seconds=time.perf_counter()-candidate_began, error=error)
        item.update(solver_settings=settings, rounding_diagnostics=diagnostics)
        report["gcs_solve_calls"] += 1
        if solved is not None:
            item.update(cost=solved["cost"], quality=solved["quality"], rounded_path_count=solved["rounded_path_count"],
                        solver_name=solved["solver_name"])
            if best is None or solved["cost"] < best["cost"]:
                best = dict(solved, selected_target=index)
        candidates.append(item)
        print(f"Spatial target {index+1}: {item.get('cost', 'no solution')}, {item['seconds']:.2f}s", flush=True)
    report.update(stage1_seconds=time.perf_counter()-stage1_began, stage1_candidates=candidates)
    report["stage1_rounding_diagnostics"] = dict(
        rounded_path_count=sum(c["rounding_diagnostics"].get("rounded_path_count", 0) for c in candidates),
        successful_rounded_path_count=sum(c["rounding_diagnostics"].get("successful_rounded_path_count", 0)
                                          for c in candidates), rounding_trials_attempted=None)
    if best is None:
        return finish("stage1_solver_failed", "No feasible first-stage path was returned.")
    lowers = [c.get("quality", {}).get("lower_bound") for c in candidates]
    report["stage1_quality"] = bound_quality(
        min(lowers) if all(v is not None for v in lowers) else None, best["cost"],
        scope="union_of_sampled_target_intersections_graph_not_continuous_ssm")
    target_index = best["selected_target"]
    target = targets[target_index]
    envelope, region = envelopes[target["envelope"]], regions[target["region"]]
    endpoint = best["waypoints"][-1]
    projected = project_to_local_ssm(
        endpoint, envelope, region, goal, obstacles, model.LINK_RADIUS,
        position=model.position, collision_free=lambda q: model.collision_free(q, obstacles))
    report.update(selected_target=target_index, selected_ssm_branch=envelope.branch,
                  stage1_cost=best["cost"], stage1_region_route=best["region_route"])
    if projected is None:
        return finish("projection_failed", "No feasible continuous IK projection in the selected local region.")
    report.update(estimated_goal=(projected+offsets).tolist(), projection_distance=float(np.linalg.norm(projected-endpoint)))
    baseline = np.vstack((best["waypoints"], projected))
    baseline_free, checked = model.validate_path(baseline, obstacles)
    report["projection_connection_baseline"] = dict(
        cost=best["cost"]+weights[target["region"]]*float(np.linalg.norm(projected-endpoint)),
        collision_free=baseline_free, collision_samples=checked,
        waypoints=(baseline+offsets).tolist())
    goal_regions = [i for i, r in enumerate(regions) if r.contains(projected)]
    stage2_began = time.perf_counter()
    print("Spatial stage 2: full fixed-goal GCS plus retained stage-one route", flush=True)
    stage2_diagnostics = {}
    solved = solve_target_sets(
        regions, edges, start, [dict(polytope=sample_hull([projected]), region=goal_regions[0], regions=goal_regions)],
        weights, candidate_region_routes=[best["region_route"]], diagnostics=stage2_diagnostics, **second_settings)
    report.update(stage2_seconds=time.perf_counter()-stage2_began, gcs_solve_calls=report["gcs_solve_calls"]+1,
                  stage2_rounding_diagnostics=stage2_diagnostics)
    if solved is None:
        return finish("stage2_solver_failed", "Fixed-goal GCS returned no feasible path.")
    path = solved["waypoints"]+offsets
    free, checked = model.validate_path(path, obstacles)
    endpoint_error = float(np.linalg.norm(model.position(path[-1])-goal))
    start_error = float(np.linalg.norm(path[0]-supplied))
    fixed_error = float(np.linalg.norm(path[-1]-(projected+offsets)))
    report.update(cost=solved["cost"], stage2_quality=solved["quality"],
                  solver_name=solved["solver_name"],
                  stage2_native_cost=solved["native_cost"], stage2_solution_source=solved["solution_source"],
                  stage2_region_route=solved["region_route"], stage2_additional_route_results=solved["additional_route_results"],
                  stage2_rounded_path_count=solved["rounded_path_count"], waypoints=path.tolist(),
                  collision_free=free, collision_samples=checked, endpoint_error=endpoint_error,
                  start_error=start_error, fixed_goal_error=fixed_error)
    valid = free and max(endpoint_error, start_error, fixed_error) <= 1e-6
    return finish("success" if valid else "path_validation_failed",
                  "Spatial fixed-goal path validated; optimality remains subject to reported bounds."
                  if valid else "Spatial path failed collision or endpoint validation.", success=bool(valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["ftwmax", "rcmax"], default="ftwmax")
    parser.add_argument("--q", type=float, nargs=4, default=model.START_CONFIGURATION.tolist())
    parser.add_argument("--goal", type=float, nargs=3)
    parser.add_argument("--iris-regions", default=None,
                        help="Defaults to a separate local checkpoint for the selected robot")
    parser.add_argument("--iris-radius", type=float, default=0.9,
                        help="Local IRIS inflation radius around seeds; coverage bound, not a physical limit")
    parser.add_argument("--generate-iris", action="store_true")
    parser.add_argument("--iris-seeds", help="JSON with waypoints used only as IRIS inflation seeds")
    parser.add_argument("--legacy-obstacle", action="store_true",
                        help="Reproduce the historical sphere at [0.5,0,0.3] instead of the task midpoint")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--compare-sampled", action="store_true")
    parser.add_argument("--compare-union-hulls", action="store_true")
    parser.add_argument("--compare-nonlinear-goal", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--nonlinear-solver", choices=["snopt", "ipopt"], default="snopt")
    parser.add_argument("--geometric-epsilon", type=float, default=GEOMETRIC_EPSILON)
    parser.add_argument("--ssm-arc-spacing", type=float, default=SSM_ARC_SPACING)
    parser.add_argument("--ssm-ik-attempts", type=int, default=model.MAX_SSM_IK_ATTEMPTS,
                        help="Multistart budget for finding up to four SSM branches; not a completeness certificate")
    parser.add_argument("--ssm-seed", type=int, default=1)
    parser.add_argument("--max-rounded-paths", type=int, default=MAX_ROUNDED_PATHS)
    parser.add_argument("--max-rounding-trials", type=int, default=MAX_ROUNDING_TRIALS)
    parser.add_argument("--stage2-max-rounded-paths", type=int, default=MAX_ROUNDED_PATHS,
                        help="Independent second-stage path cap (default: 200)")
    parser.add_argument("--stage2-max-rounding-trials", type=int, default=MAX_ROUNDING_TRIALS,
                        help="Independent second-stage trial cap (default: 300)")
    parser.add_argument("--solver", choices=["mosek", "clarabel"], default="mosek")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    model.select_robot(args.robot)
    scene_name = "local" if args.legacy_obstacle else "midpoint"
    args.iris_regions = args.iris_regions or f"data/iris/spatial4R_{args.robot}_iris_{scene_name}_regions.npz"
    args.output = args.output or f"results/spatial4r/spatial4R_{args.robot}_{scene_name}_two_stage_path.json"
    try:
        iris_seeds = (json.loads(Path(args.iris_seeds).read_text(encoding="utf-8"))["waypoints"]
                      if args.iris_seeds else None)
        result = plan_before_failure(args.q, args.goal, args.iris_regions,
                                     obstacles=model.LEGACY_OBSTACLES if args.legacy_obstacle else None,
                                     geometric_epsilon=args.geometric_epsilon, arc_spacing=args.ssm_arc_spacing,
                                     max_paths=args.max_rounded_paths, max_trials=args.max_rounding_trials,
                                     geometry_only=args.geometry_only, generate_iris=args.generate_iris,
                                     solver_backend=args.solver, iris_radius=args.iris_radius,
                                     iris_seeds=iris_seeds, ssm_ik_attempts=args.ssm_ik_attempts,
                                     ssm_seed=args.ssm_seed, compare_sampled=args.compare_sampled or args.compare_all,
                                     compare_union_hulls=args.compare_union_hulls or args.compare_all,
                                     compare_nonlinear_goal=args.compare_nonlinear_goal or args.compare_all,
                                     nonlinear_solver=args.nonlinear_solver,
                                     stage2_max_paths=args.stage2_max_rounded_paths,
                                     stage2_max_trials=args.stage2_max_rounding_trials)
    except (ValueError, OSError, RuntimeError) as exc:
        result = dict(success=False, status="planning_error", message=str(exc))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
