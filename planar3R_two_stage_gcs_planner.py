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
from planar3R_ssm_envelopes import (
    SSM_ARC_SPACING, build_envelopes, intersect_envelopes, project_to_local_ssm,
)
from gcs.quality import bound_quality


class MixedIntegerSolverUnavailable(RuntimeError):
    pass


def require_mixed_integer_solver():
    """Check solver support for the actual mixed-integer conic problem class."""
    from pydrake.solvers import ChooseBestSolver, MathematicalProgram
    probe = MathematicalProgram()
    binary = probe.NewBinaryVariables(1)[0]
    continuous = probe.NewContinuousVariables(1)[0]
    probe.AddLorentzConeConstraint(np.array([binary + 1, continuous], dtype=object))
    try:
        return ChooseBestSolver(probe).name()
    except (RuntimeError, ValueError) as exc:
        raise MixedIntegerSolverUnavailable(
            "Exact GCS requires an enabled mixed-integer conic solver. " + str(exc)) from exc


def solve_target_sets(regions, edges, start, targets, weights, rounding=True,
                      candidate_region_routes=None, max_rounded_paths=10,
                      max_rounding_trials=100, rounding_seed=0,
                      relative_gap_tolerance=0.01, mip_time_limit=60.0,
                      mip_relative_gap=0.01, solver_backend=None):
    """Use a direct terminal vertex for a single target; charge physical motion."""
    from pydrake.geometry.optimization import Point
    from pydrake.solvers import L2NormCost
    from gcs.linear import BindingCost, LinearGCS

    if max_rounded_paths < 1 or max_rounding_trials < 1:
        raise ValueError("Rounding path and trial budgets must be positive")
    if (not np.isfinite(mip_time_limit) or mip_time_limit <= 0
            or not np.isfinite(mip_relative_gap) or mip_relative_gap < 0):
        raise ValueError("MIP time limit must be positive and MIP gap nonnegative")
    bound_quality(None, None, relative_gap_tolerance)  # Validate before solving.
    exact_solver = require_mixed_integer_solver() if not rounding else None
    dimension = len(start)

    planner = LinearGCS([r.to_drake() for r in regions], edges=edges,
                        vertex_cost_weights=weights)
    planner.options.max_rounded_paths = max_rounded_paths
    planner.options.max_rounding_trials = max_rounding_trials
    planner.options.rounding_seed = rounding_seed
    if solver_backend is not None:
        from pydrake.solvers import ClarabelSolver, MosekSolver
        if solver_backend not in ("mosek", "clarabel"):
            raise ValueError("Solver backend must be mosek or clarabel")
        conic_solver = MosekSolver() if solver_backend == "mosek" else ClarabelSolver()
        if not (conic_solver.available() and conic_solver.enabled()):
            raise RuntimeError(f"Requested solver {solver_backend} is not available/enabled")
        planner.options.solver = conic_solver
    if not rounding:
        from pydrake.solvers import GurobiSolver, MosekSolver, SolverOptions
        solver_options = SolverOptions()
        if exact_solver == MosekSolver.id().name():
            solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", mip_time_limit)
            solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", mip_relative_gap)
        elif exact_solver == GurobiSolver.id().name():
            solver_options.SetOption(GurobiSolver.id(), "TimeLimit", mip_time_limit)
            solver_options.SetOption(GurobiSolver.id(), "MIPGap", mip_relative_gap)
        else:
            raise RuntimeError(f"No bounded-runtime configuration for MIP solver {exact_solver}")
        planner.setSolverOptions(solver_options)
    vertices = list(planner.gcs.Vertices())
    planner.source = planner.gcs.AddVertex(Point(start), "fixed_start")
    multiple_targets = len(targets) > 1
    if not targets:
        return None
    if multiple_targets:
        planner.target = planner.gcs.AddVertex(Point(np.zeros(dimension)), "terminal_sink")
    for index, region in enumerate(regions):
        if region.contains(start):
            edge = planner.gcs.AddEdge(planner.source, vertices[index])
            for axis in range(dimension):
                edge.AddConstraint(edge.xu()[axis] == edge.xv()[axis])
    target_vertices = {}
    for index, target in enumerate(targets):
        vertex = planner.gcs.AddVertex(target["polytope"].to_drake(), f"local_target_{index}")
        target_vertices[vertex.name()] = index
        for region_index in target.get("regions", [target["region"]]):
            edge = planner.gcs.AddEdge(vertices[region_index], vertex)
            matrix = weights[region_index] * np.hstack((-np.eye(dimension), np.eye(dimension)))
            edge.AddCost(BindingCost(L2NormCost(matrix, np.zeros(dimension)),
                                     np.append(edge.xu(), edge.xv())))
        if multiple_targets:
            planner.gcs.AddEdge(vertex, planner.target)
        else:
            planner.target = vertex
    waypoints, results = planner.SolvePath(rounding=rounding, preprocessing=True)
    if waypoints is None and not candidate_region_routes:
        return None
    path = results.get("best_path") if waypoints is not None else None
    best_result = results.get("best_result") if path is not None else None

    def physical_cost(route, result):
        return float(sum(weights[planner.names.index(e.u().name())] * np.linalg.norm(
            result.GetSolution(e.xv()) - result.GetSolution(e.xu()))
            for e in route if e.u().name() in planner.names))

    actual_cost = physical_cost(path, best_result) if path is not None else np.inf
    native_cost = actual_cost if path is not None else None
    selection = "native_gcs"
    route_results = []
    if candidate_region_routes:
        if multiple_targets:
            raise ValueError("Additional region routes require one fixed target")
        # This adds candidate convex restrictions, not a constraint on the full
        # graph search. All native rounded routes remain eligible to win.
        from pydrake.geometry.optimization import GraphOfConvexSetsOptions
        edge_lookup = {(e.u().name(), e.v().name()): e for e in planner.gcs.Edges()}
        for route in candidate_region_routes:
            names = ([planner.source.name()] + [planner.names[i] for i in route]
                     + [planner.target.name()])
            pairs = list(zip(names[:-1], names[1:]))
            if len(set(route)) != len(route) or any(pair not in edge_lookup for pair in pairs):
                route_results.append({"region_route": list(route), "success": False,
                                      "error": "Region route is not a simple path in this graph"})
                continue
            active = [edge_lookup[pair] for pair in pairs]
            restricted = planner.gcs.SolveConvexRestriction(active, GraphOfConvexSetsOptions())
            item = {"region_route": list(route), "success": restricted.is_success()}
            if restricted.is_success():
                cost = physical_cost(active, restricted)
                item.update(cost=cost, solver_cost=float(restricted.get_optimal_cost()))
                if cost < actual_cost:
                    path, best_result, actual_cost = active, restricted, cost
                    selection = "stage1_route_convex_restriction"
            route_results.append(item)
    if path is None:
        return None
    selected = target_vertices[(path[-1].u() if multiple_targets else path[-1].v()).name()]
    waypoints = np.asarray([best_result.GetSolution(e.xv()) for e in path])
    if multiple_targets:
        waypoints = waypoints[:-1]  # Remove only the optional bookkeeping sink.
    region_route = [planner.names.index(e.u().name()) for e in path
                    if e.u().name() in planner.names]
    # MIP incumbents are upper bounds too. Do not manufacture a lower bound
    # from a successful MIP return without a separately reported solver bound.
    lower = results.get("relaxation_cost")
    quality = bound_quality(lower, actual_cost, relative_gap_tolerance,
                            scope="fixed_start_to_supplied_target_sets_graph")
    details = best_result.get_solver_details()
    mip_details = ({name: float(getattr(details, name))
                    for name in ("rescode", "solution_status", "optimizer_time")
                    if hasattr(details, name)} if not rounding else None)
    return {"waypoints": waypoints, "cost": float(actual_cost), "target": selected,
            "solver_cost": float(best_result.get_optimal_cost()),
            "region_route": region_route, "solution_source": selection,
            "native_cost": native_cost, "additional_route_results": route_results,
            "quality": quality, "solve_mode": "relaxation_rounding" if rounding else "mixed_integer",
            "exact_solver": exact_solver,
            "solver_name": best_result.get_solver_id().name(),
            "solver_termination": str(best_result.get_solution_result()),
            "mip_details": mip_details,
            "rounded_path_count": results.get("rounded_path_count", 0),
            "successful_rounded_path_count": results.get("successful_rounded_path_count", 0)}


def plan_two_stage(start, goal, obstacles, checkpoint_path="planar3R_iris_regions.npz",
                   geometric_epsilon=0.025, samples_per_branch=None,
                   reliability_source="goal", reliability_branch=0,
                   reliability_epsilon=0.01, min_reliability=0.0, seed=0,
                   rounding=True, compare_sampled=False, max_rounded_paths=200,
                   max_rounding_trials=300, rounding_seed=0, relative_gap_tolerance=0.01,
                   mip_time_limit=60.0, mip_relative_gap=0.01,
                   stage2_max_rounded_paths=200, stage2_max_rounding_trials=300,
                   geometry_only=False, ssm_arc_spacing=SSM_ARC_SPACING):
    started = time.perf_counter()
    supplied = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if (supplied.shape != (3,) or goal.shape != (2,) or not np.all(np.isfinite(supplied))
            or not np.all(np.isfinite(goal)) or not np.isfinite(reliability_epsilon)
            or reliability_epsilon <= 0 or not 0 <= min_reliability <= 1
            or not np.isfinite(geometric_epsilon) or geometric_epsilon <= 0
            or (samples_per_branch is not None and samples_per_branch < 2)
            or not np.isfinite(ssm_arc_spacing) or ssm_arc_spacing <= 0
            or max_rounded_paths < 1 or max_rounding_trials < 1
            or not np.isfinite(mip_time_limit) or mip_time_limit <= 0
            or not np.isfinite(mip_relative_gap) or mip_relative_gap < 0
            or stage2_max_rounded_paths < 1 or stage2_max_rounding_trials < 1):
        raise ValueError("Invalid endpoints, epsilon, sample count, or reliability threshold")
    bound_quality(None, None, relative_gap_tolerance)
    solve_settings = dict(max_rounded_paths=max_rounded_paths,
                          max_rounding_trials=max_rounding_trials, rounding_seed=rounding_seed,
                          relative_gap_tolerance=relative_gap_tolerance,
                          mip_time_limit=mip_time_limit, mip_relative_gap=mip_relative_gap)
    stage2_settings = dict(solve_settings, max_rounded_paths=stage2_max_rounded_paths,
                          max_rounding_trials=stage2_max_rounding_trials)
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
    report.update(solve_mode="relaxation_rounding" if rounding else "mixed_integer",
                  solver_settings=solve_settings, stage1_solver_settings=solve_settings,
                  stage2_solver_settings=stage2_settings)

    def finish(status, message, **values):
        report.update(status=status, message=message,
                      total_seconds=time.perf_counter() - started, **values)
        return report

    if np.linalg.norm(goal) > sum(L) + 1e-10:
        return finish("goal_unreachable", "Goal is outside radial reach; no GCS solve performed.")
    if not rounding and not geometry_only:
        try:
            report["exact_solver"] = require_mixed_integer_solver()
        except ImportError as exc:
            return finish("drake_unavailable", str(exc))
        except MixedIntegerSolverUnavailable as exc:
            return finish("mixed_integer_solver_unavailable", str(exc))
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
        raise ValueError("Only reliability_branch=0 is supported; all SSM branches share one combined map")
    partition = partition_sets[reliability_branch]
    report["reliability_branch"] = partition["label"]
    report["reliability_branch_count"] = partition.get("branch_count", 0)
    report["reliability_rule"] = partition.get("reliability_rule", "no_reliability_partition")
    report["reliability_partition_count"] = len(partition["regions"] or [])
    report["unmerged_reliability_partition_count"] = partition.get(
        "unmerged_partition_count", report["reliability_partition_count"])
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
    envelopes = build_envelopes(
        refined_branches, geometric_epsilon, samples_per_branch, arc_spacing=ssm_arc_spacing,
        refine_sample=(lambda q: refine_ik_configuration(q, *goal)) if samples_per_branch is None else None,
        sampling_report=report)
    targets = intersect_envelopes(envelopes, regions)
    report.update(envelope_count=len(envelopes), target_set_count=len(targets),
                  actual_sample_count=len({tuple(q) for e in envelopes for q in e.samples}),
                  region_count=len(regions), edge_count=len(edges),
                  geometry_seconds=time.perf_counter() - setup_started,
                  envelope_polyline_bounds=[e.polyline_distance_bound for e in envelopes])
    print(f"Geometry: {report['reliability_branch_count']} reliability branches; "
          f"{report['unmerged_reliability_partition_count']} -> "
          f"{report['reliability_partition_count']} boxes after equal-reliability merging; "
          f"{len(regions)} free regions, {len(edges)} directed edges; "
          f"{report['actual_sample_count']} SSM samples, {len(targets)} targets", flush=True)
    if not targets:
        return finish("no_path_in_iris_coverage", "No sampled target hull intersects connected coverage.")
    if geometry_only:
        return finish("geometry_only", "Geometry counts only; no GCS planning performed.",
                      success=True, original_region_count=len(original),
                      stage1_gcs_solve_calls=0, stage2_gcs_solve_calls=0)
    stage_started = time.perf_counter()
    report.update(stage1_mode="per_target_intersection", stage1_candidates=[],
                  stage1_gcs_solve_calls=0, stage2_gcs_solve_calls=0)
    first = None
    for target_index, candidate_target in enumerate(targets):
        candidate_started = time.perf_counter()
        print(f"Stage 1: solving target {target_index + 1}/{len(targets)}", flush=True)
        error = None
        try:
            report["gcs_solve_calls"] += 1
            report["stage1_gcs_solve_calls"] += 1
            candidate = solve_target_sets(
                regions, edges, start, [candidate_target], weights, rounding, **solve_settings)
        except ImportError as exc:
            report["gcs_solve_calls"] -= 1
            report["stage1_gcs_solve_calls"] -= 1
            return finish("drake_unavailable", str(exc))
        except RuntimeError as exc:
            candidate, error = None, str(exc)
        candidate_seconds = time.perf_counter() - candidate_started
        item = {"target": target_index, "envelope": candidate_target["envelope"],
                "region": candidate_target["region"], "seconds": candidate_seconds,
                "success": candidate is not None, "error": error}
        if candidate is not None:
            item.update(cost=candidate["cost"], solver_cost=candidate.get("solver_cost"),
                        endpoint=(candidate["waypoints"][-1] + offsets).tolist(),
                        quality=candidate.get("quality"),
                        rounded_path_count=candidate.get("rounded_path_count"))
            if first is None or candidate["cost"] < first["cost"]:
                first = dict(candidate, target=target_index)
        report["stage1_candidates"].append(item)
        result_text = f"cost={candidate['cost']:.10f}" if candidate is not None else "no solution"
        print(f"Stage 1: target {target_index + 1}/{len(targets)} finished in "
              f"{candidate_seconds:.2f}s; {result_text}", flush=True)
    report["stage1_seconds"] = time.perf_counter() - stage_started
    if first is None:
        return finish("solver_failed", "No individual target solve returned a feasible route.")
    candidate_lowers = [item.get("quality", {}).get("lower_bound")
                        if item.get("quality") else None for item in report["stage1_candidates"]]
    union_lower = min(candidate_lowers) if all(x is not None for x in candidate_lowers) else None
    report["stage1_quality"] = bound_quality(
        union_lower, first["cost"], relative_gap_tolerance,
        scope="union_of_sampled_target_intersections_graph_not_continuous_ssm")
    target = targets[first["target"]]
    envelope = envelopes[target["envelope"]]
    p = first["waypoints"][-1]
    report.update(stage1_cost=first["cost"], selected_target=first["target"],
                  selected_envelope=target["envelope"], selected_region=target["region"],
                  selected_ssm_branch=envelope.branch, selected_chart_piece=envelope.piece,
                  stage1_endpoint=(p + offsets).tolist())
    if "region_route" in first:
        report["stage1_region_route"] = first["region_route"]
    projection_started = time.perf_counter()
    estimated = project_to_local_ssm(p, envelope, regions[target["region"]],
                                     goal, obstacles, link_radius)
    report["projection_seconds"] = time.perf_counter() - projection_started
    if estimated is None:
        return finish("projection_failed", "No valid continuous IK projection in the selected local free portion.")
    report.update(estimated_goal=(estimated + offsets).tolist(),
                  projection_distance=float(np.linalg.norm(estimated - p)))
    comparison_started = time.perf_counter()
    selected_region = regions[target["region"]]
    connector_weight = float(weights[target["region"]])
    connector_cost = connector_weight * report["projection_distance"]
    connected_path = np.vstack((first["waypoints"], estimated)) + offsets
    connected_free, connected_checked = validate_joint_path(
        connected_path, obstacles, link_radius)
    connector_in_region = bool(selected_region.contains(p)
                               and selected_region.contains(estimated))
    connected_endpoint_error = float(np.linalg.norm(
        link_points(connected_path[-1])[-1] - goal))
    connected_start_error = float(np.linalg.norm(connected_path[0] - supplied))
    report["projection_connection_baseline"] = {
        "method": "stage1_path_plus_projection_segment",
        "success": bool(connected_free and connector_in_region
                        and max(connected_endpoint_error, connected_start_error) <= 1e-6),
        "stage1_cost": first["cost"],
        "connector_length": report["projection_distance"],
        "connector_region": target["region"],
        "connector_reliability": float(selected_region.reliability),
        "connector_weight": connector_weight,
        "connector_cost": float(connector_cost),
        "cost": float(first["cost"] + connector_cost),
        "connector_in_selected_region": connector_in_region,
        "collision_free": connected_free,
        "collision_samples": connected_checked,
        "endpoint_error": connected_endpoint_error,
        "start_error": connected_start_error,
        "waypoints": connected_path.tolist(),
        "seconds": time.perf_counter() - comparison_started,
        "gcs_solve_calls": 0,
    }
    # Fresh graph with ALL retained free regions and adjacencies. No first route
    # is frozen and no projection segment is appended to the returned motion.
    from planar3R_ssm_envelopes import sample_hull
    goal_regions = [i for i, region in enumerate(regions) if region.contains(estimated)]
    fixed_targets = [{"polytope": sample_hull([estimated]),
                      "region": goal_regions[0], "regions": goal_regions}]
    stage_started = time.perf_counter()
    report["gcs_solve_calls"] += 1
    report["stage2_gcs_solve_calls"] += 1
    print(f"Stage 2: replanning to projected goal from target {first['target'] + 1}; "
          f"stage 1 best cost={first['cost']:.10f}", flush=True)
    try:
        known_routes = [first["region_route"]] if "region_route" in first else None
        second = solve_target_sets(regions, edges, start, fixed_targets, weights, rounding,
                                   candidate_region_routes=known_routes, **stage2_settings)
    except RuntimeError as exc:
        return finish("solver_failed", str(exc))
    report["stage2_seconds"] = time.perf_counter() - stage_started
    if second is None:
        return finish("solver_failed", "Second-stage full GCS returned no feasible route.")
    report.update(stage2_solution_source=second.get("solution_source", "native_gcs"),
                  stage2_quality=second.get("quality"),
                  stage2_mip_details=second.get("mip_details"),
                  stage2_rounded_path_count=second.get("rounded_path_count"),
                  stage2_native_cost=second.get("native_cost"),
                  stage2_additional_route_results=second.get("additional_route_results", []),
                  stage2_additional_convex_restriction_calls=len(
                      second.get("additional_route_results", [])),
                  stage2_region_route=second.get("region_route"))
    if report["stage2_quality"] is not None:
        report["stage2_quality"] = dict(report["stage2_quality"], scope="fixed_endpoint_graph")
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
    if report["projection_connection_baseline"]["success"]:
        report["stage2_minus_projection_connection_cost"] = float(
            second["cost"] - report["projection_connection_baseline"]["cost"])
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
            baseline = solve_target_sets(regions, edges, start, baseline_targets, weights,
                                         rounding, **solve_settings) if baseline_targets else None
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
                                      "quality": baseline.get("quality") if baseline else None,
                                      "gcs_solve_calls": int(bool(baseline_targets)),
                                      "method": "one_graph_all_sampled_endpoints"}
    report["success"] = True
    quality_status = (report["stage2_quality"]["status"] if report["stage2_quality"]
                      else "bound_unavailable")
    return finish("success", "Validated second-stage path to continuous IK goal; "
                  f"graph-model quality: {quality_status}.")


def main():
    from planar3R_gcs_planner import GOAL_POINT, OBSTACLES

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--q", type=float, nargs=3, help="Fixed start configuration in radians")
    source.add_argument("--start-path", default=None, help="Read fixed start from saved path JSON")
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--gx", type=float, default=GOAL_POINT[0])
    parser.add_argument("--gy", type=float, default=GOAL_POINT[1])
    parser.add_argument("--geometric-epsilon", type=float, default=0.025)
    parser.add_argument("--ssm-arc-spacing", type=float, default=SSM_ARC_SPACING,
                        help="Joint-space arc sampling interval shared by all branches (radians)")
    parser.add_argument("--ssm-samples", type=int, default=None,
                        help="Optional legacy per-branch budget; overrides fixed arc spacing")
    parser.add_argument("--iris-regions", default="planar3R_iris_regions.npz")
    parser.add_argument("--reliability-source", choices=["start", "goal", "none"], default="goal")
    parser.add_argument("--reliability-branch", type=int, default=0,
                        help="Compatibility option; use 0 for the combined map of all found branches")
    parser.add_argument("--reliability-epsilon", type=float, default=0.01)
    parser.add_argument("--min-reliability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rounding", action="store_true")
    parser.add_argument("--max-rounded-paths", type=int, default=200)
    parser.add_argument("--max-rounding-trials", type=int, default=300)
    parser.add_argument("--stage2-max-rounded-paths", type=int, default=200)
    parser.add_argument("--stage2-max-rounding-trials", type=int, default=300)
    parser.add_argument("--rounding-seed", type=int, default=0)
    parser.add_argument("--relative-gap-tolerance", type=float, default=0.01,
                        help="Graph-model quality threshold; 0.01 means 1 percent")
    parser.add_argument("--mip-time-limit", type=float, default=60.0,
                        help="Per-call mixed-integer solver search time limit in seconds")
    parser.add_argument("--mip-relative-gap", type=float, default=0.01,
                        help="Mixed-integer solver stopping gap; independent of report threshold")
    parser.add_argument("--compare-sampled", action="store_true")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Report hull and intersection counts without running either GCS stage")
    parser.add_argument("--output", default="two_stage_path.json")
    args = parser.parse_args()
    try:
        start = args.q if args.q is not None else json.loads(
            Path(args.start_path or "iris_path.json").read_text(encoding="utf-8"))["waypoints"][args.waypoint_index]
        result = plan_two_stage(start, [args.gx, args.gy], OBSTACLES, args.iris_regions,
                                args.geometric_epsilon, args.ssm_samples, args.reliability_source,
                                args.reliability_branch, args.reliability_epsilon, args.min_reliability,
                                args.seed, not args.no_rounding, args.compare_sampled,
                                args.max_rounded_paths, args.max_rounding_trials,
                                args.rounding_seed, args.relative_gap_tolerance,
                                args.mip_time_limit, args.mip_relative_gap,
                                args.stage2_max_rounded_paths, args.stage2_max_rounding_trials,
                                geometry_only=args.geometry_only, ssm_arc_spacing=args.ssm_arc_spacing)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        result = {"success": False, "status": "invalid_input", "message": str(exc)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
