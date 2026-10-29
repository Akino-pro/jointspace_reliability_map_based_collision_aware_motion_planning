"""Independent union-hull and native nonlinear-terminal GCS comparisons."""
import hashlib
import json
import time

import numpy as np


BASELINE_KEYS = ("sampled_baseline", "union_hull_baseline", "nonlinear_goal_baseline")


def planar_position_jacobian(q, lengths):
    angles = np.cumsum(q)
    return np.array([[-np.sum(lengths[j:]*np.sin(angles[j:])),
                      np.sum(lengths[j:]*np.cos(angles[j:]))] for j in range(len(q))]).T


def baseline_seconds(report):
    return sum(report.get(key, {}).get("seconds", 0.) for key in BASELINE_KEYS)


def configure_nonlinear_restriction(options, backend):
    from pydrake.solvers import SnoptSolver, IpoptSolver, SolverOptions
    if backend not in ("snopt", "ipopt"):
        raise ValueError("Nonlinear restriction solver must be snopt or ipopt")
    solver = SnoptSolver() if backend == "snopt" else IpoptSolver()
    if not (solver.available() and solver.enabled()):
        raise RuntimeError(f"Nonlinear solver {backend} is unavailable")
    options.restriction_solver = solver
    settings = SolverOptions()
    if backend == "snopt":
        settings.SetOption(solver.id(), "Major iterations limit", 500)
        settings.SetOption(solver.id(), "Major feasibility tolerance", 1e-8)
        settings.SetOption(solver.id(), "Major optimality tolerance", 1e-7)
    else:
        settings.SetOption(solver.id(), "max_iter", 500)
        settings.SetOption(solver.id(), "tol", 1e-8)
        settings.SetOption(solver.id(), "max_cpu_time", 60.)
    options.restriction_solver_options = settings


def task_evaluator(task):
    """Supply exact task Jacobians to Drake's AutoDiff Python constraint."""
    def evaluate(x):
        from pydrake.autodiffutils import AutoDiffXd, ExtractValue, ExtractGradient
        x = np.asarray(x).reshape(-1)
        values = np.asarray(ExtractValue(x), dtype=float).reshape(-1)
        residual = np.asarray(task["residual"](values), dtype=float).reshape(-1)
        if len(x) and isinstance(x[0], AutoDiffXd):
            derivative = np.asarray(task["jacobian"](values)) @ ExtractGradient(x)
            return np.array([AutoDiffXd(value, row) for value, row in zip(residual, derivative)])
        return residual
    return evaluate


def add_terminal_task_constraint(vertex, task):
    from pydrake.geometry.optimization import GraphOfConvexSets
    from pydrake.solvers import MathematicalProgram
    program = MathematicalProgram()
    program.AddDecisionVariables(vertex.x())
    zero = np.zeros(task["dimension"])
    binding = program.AddConstraint(task_evaluator(task), zero, zero, vertex.x())
    vertex.AddConstraint(binding, {GraphOfConvexSets.Transcription.kRestriction})


def _fingerprint(regions, edges, start, weights, settings, task_id):
    contract = dict(start=np.asarray(start).tolist(), task=task_id, settings=settings,
                    regions=[dict(A=r.A.tolist(), b=r.b.tolist(), weight=float(w))
                             for r, w in zip(regions, weights)],
                    edges=[dict(pair=list(e), shift=np.asarray(getattr(e, "shift", [])).tolist()) for e in edges])
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def graph_task_fingerprint(regions, edges, start, weights, task_id):
    return _fingerprint(regions, edges, start, weights, {}, task_id)


def hull_targets_fingerprint(targets):
    contract = [dict(A=t["polytope"].A.tolist(), b=t["polytope"].b.tolist(),
                     regions=t.get("regions", [t["region"]]), envelope=t.get("envelope"))
                for t in targets]
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def validate_result(result, regions, start, targets, weights, task, validate_path):
    """Check graph/target membership, lifted weighted cost and true task."""
    from scripts.periodic_graph import path_in_regions
    path = np.asarray(result["waypoints"])
    route = result["region_route"]
    index = int(result["target"])
    if (not 0 <= index < len(targets) or path.shape != (len(route)+1, len(start))
            or not np.all(np.isfinite(path)) or not route
            or any(not 0 <= i < len(regions) for i in route)):
        return dict(success=False, status="path_validation_failed")
    graph_valid = path_in_regions(result, regions, start, targets[index]["polytope"])
    free, checked = validate_path(path)
    error = float(np.linalg.norm(task["residual"](path[-1])))
    length_cost = float(sum(weights[r]*np.linalg.norm(path[i+1]-path[i]) for i, r in enumerate(route)))
    cost_valid = bool(np.isfinite(result["cost"]) and abs(length_cost-result["cost"]) <= 1e-6)
    valid = bool(graph_valid and free and cost_valid and error <= 1e-6)
    return dict(success=valid, status="success" if valid else "path_validation_failed",
                collision_free=bool(free), collision_samples=checked, task_error=error,
                graph_membership_valid=bool(graph_valid), recomputed_cost=length_cost,
                cost_valid=cost_valid, cost=float(result["cost"]), quality=result["quality"],
                waypoints=path.tolist(), region_route=route, route_spec=result["route_spec"],
                endpoint=path[-1].tolist(), selected_target=index,
                rounded_path_count=result["rounded_path_count"], solver_name=result["solver_name"])


def compare_goal_sets(regions, edges, start, targets, envelopes, weights, settings,
                      stage2_settings, task, project, validate_path, offsets=None,
                      union_hulls=False, nonlinear_goal=False, rounding=True,
                      nonlinear_solver="snopt"):
    """Neither comparison seeds or replaces the primary planner.

    Union-hull targets exactly match the main stage one. The nonlinear method
    uses every connected free region as a terminal domain, with the real task
    enforced only in native restrictions. Its convex relaxation intentionally
    drops the task, a valid but possibly very weak surrogate. Sample hulls are
    NOT imposed on nonlinear endpoints, since they can exclude the true curve.
    """
    from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
    from scripts.planar3R_ssm_envelopes import sample_hull
    offsets = np.zeros(len(start)) if offsets is None else np.asarray(offsets)
    common_hash = _fingerprint(regions, edges, start, weights, settings, task["id"])
    graph_hash = graph_task_fingerprint(regions, edges, start, weights, task["id"])
    target_hash = hull_targets_fingerprint(targets)
    outputs = {}
    for key, enabled in (("union_hull_baseline", union_hulls), ("nonlinear_goal_baseline", nonlinear_goal)):
        if not enabled:
            continue
        began = time.perf_counter()
        nonlinear = key == "nonlinear_goal_baseline"
        report = dict(success=False, method=("native_gcs_nonlinear_terminal" if nonlinear else "one_graph_all_hull_targets_two_stage"),
                      gcs_solve_calls=0, solver_settings=dict(settings),
                      global_optimality_certified=False, input_sha256=common_hash,
                      graph_task_sha256=graph_hash, hull_targets_sha256=target_hash,
                      task=task["id"], path_coordinates="input_winding_restored",
                      angle_winding_offsets=offsets.tolist())
        outputs[key] = report
        if nonlinear:
            report.update(restriction_solver=nonlinear_solver, restriction_iteration_limit=500,
                          goal_surrogate="all_connected_free_regions_without_task_constraint",
                          surrogate_is_weak=True,
                          scope="full_true_task_within_same_connected_graph_chart")
        else:
            report.update(stage2_solver_settings=dict(stage2_settings), target_count=len(targets),
                          scope="same_hull_targets_projection_and_fixed_goal_replanning")
        try:
            if nonlinear and not rounding:
                report.update(status="unsupported_solve_mode", message="Nonlinear comparison requires rounding")
                continue
            goals = ([dict(polytope=r, region=i, regions=[i]) for i, r in enumerate(regions)]
                     if nonlinear else targets)
            report["target_count"] = len(goals)
            report["gcs_solve_calls"] = 1
            solve_began = time.perf_counter()
            first_diagnostics = {}
            report["stage1_rounding_diagnostics"] = first_diagnostics
            first = solve_target_sets(regions, edges, start, goals, weights, rounding=rounding,
                                      nonlinear_task=task if nonlinear else None,
                                      restriction_solver_backend=nonlinear_solver,
                                      diagnostics=first_diagnostics, **settings)
            report["stage1_seconds"] = time.perf_counter()-solve_began
            if first is None:
                report.update(status="solver_failed", message="No feasible rounded path returned")
                continue
            if nonlinear:
                report.update(validate_result(first, regions, start, goals, weights, task, validate_path))
            else:
                from scripts.periodic_graph import path_in_regions
                selected = int(first["target"])
                if not path_in_regions(first, regions, start, goals[selected]["polytope"]):
                    report.update(status="stage1_path_validation_failed")
                    continue
                target = goals[selected]
                report.update(stage1_region_route=first["region_route"], stage1_route_spec=first.get("route_spec"),
                              stage1_waypoints=(np.asarray(first["waypoints"])+offsets).tolist())
                chart_endpoint = first.get("chart_waypoints", first["waypoints"])[-1]
                projection_began = time.perf_counter()
                q_goal = project(chart_endpoint, envelopes[target["envelope"]], regions[target["region"]])
                report.update(stage1_cost=float(first["cost"]), stage1_quality=first["quality"],
                              selected_hull_target=selected, selected_ssm_branch=envelopes[target["envelope"]].branch,
                              projection_seconds=time.perf_counter()-projection_began)
                if q_goal is None:
                    report.update(status="projection_failed")
                    continue
                indices = [i for i, r in enumerate(regions) if r.contains(q_goal)]
                if not indices:
                    report.update(status="projection_failed")
                    continue
                fixed = [dict(polytope=sample_hull([q_goal]), region=indices[0], regions=indices)]
                report.update(estimated_goal=(q_goal+offsets).tolist(), gcs_solve_calls=2)
                solve_began = time.perf_counter()
                second_diagnostics = {}
                report["stage2_rounding_diagnostics"] = second_diagnostics
                second = solve_target_sets(regions, edges, start, fixed, weights, rounding=rounding,
                                           candidate_region_routes=[first["route_spec"]],
                                           diagnostics=second_diagnostics, **stage2_settings)
                report["stage2_seconds"] = time.perf_counter()-solve_began
                if second is None:
                    report.update(status="stage2_solver_failed")
                    continue
                report.update(validate_result(second, regions, start, fixed, weights, task, validate_path))
            if "waypoints" in report:
                report["waypoints"] = (np.asarray(report["waypoints"])+offsets).tolist()
                report["endpoint"] = (np.asarray(report["endpoint"])+offsets).tolist()
        except (RuntimeError, ValueError) as exc:
            report.update(status="solver_failed", message=str(exc))
        finally:
            report["seconds"] = time.perf_counter()-began
    return outputs
