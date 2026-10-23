"""One-shot spatial 4R replanning after one joint locks at its failure angle.

Use original IRIS slices and unweighted active-joint path length. Numerical IK
enumeration is not certified complete; missing roots are never called a proof
of unreachability. Singular goal IK is explicitly unsupported by this workflow.
"""

import argparse
import itertools
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.optimize import least_squares

from scripts import spatial4R_model as model
from scripts.planar3R_iris_regions import ReliablePolytope, interior_point, load_iris_checkpoint, polytope_adjacency
from scripts.planar3R_ssm_envelopes import sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets

MAX_ROUNDED_PATHS = 200
MAX_ROUNDING_TRIALS = 300
IK_MAX_STARTS = 256
IK_MAX_SOLUTIONS = 4
IK_MAX_EVALUATIONS = 160
IK_TOLERANCE = 1e-8
IK_DUPLICATE_TOLERANCE = 1e-5
IK_SINGULAR_TOLERANCE = 1e-7


def chart_configuration(q):
    supplied = np.asarray(q, dtype=float)
    outside = np.abs(supplied) > np.pi
    chart = np.where(outside, model.wrap(supplied), supplied)
    positive_boundary = outside & (supplied > 0) & np.isclose(chart, -np.pi, atol=1e-12, rtol=0)
    chart[positive_boundary] = np.pi
    return chart, supplied-chart


def reach_exclusion(goal, failed_joint, angle):
    """Conservative geometric exclusions, valid for the currently selected DH arm."""
    goal = np.asarray(goal)
    radius = float(np.hypot(model.A, model.D).sum())
    if np.linalg.norm(goal) > radius+IK_TOLERANCE:
        return "Goal exceeds the full arm's conservative radial reach bound."
    if failed_joint == 1:
        q = np.array([angle, 0., 0., 0.])
        # The first DH endpoint is fixed when joint 1 locks; the rest is bounded
        # by the sum of the remaining DH displacement magnitudes.
        center = model.kinematics(q)[3][1][1]
        remaining = float(np.hypot(model.A[1:], model.D[1:]).sum())
        if np.linalg.norm(goal-center) > remaining+IK_TOLERANCE:
            return "Goal exceeds the remaining arm's reach about the locked first DH endpoint."
    return None


def locked_joint_ik(goal, failed_joint, angle, initial=None, max_starts=IK_MAX_STARTS, seed=0):
    """Search the three active coordinates, deduplicating roots periodically.

    Search up to four distinct physical roots, including singular ones for
    diagnostics. Never infer a continuous solution set from rank alone.
    """
    goal = np.asarray(goal, dtype=float)
    if (failed_joint not in (1, 2, 3, 4) or goal.shape != (3,)
            or not np.all(np.isfinite(goal)) or not np.isfinite(angle)
            or not isinstance(max_starts, (int, np.integer)) or max_starts < 1):
        raise ValueError("Use joint 1..4, finite XYZ/angle and a positive integer IK budget")
    active = [i for i in range(4) if i != failed_joint-1]
    base = np.zeros(4)
    base[failed_joint-1] = angle
    def expand(u):
        q = base.copy()
        q[active] = model.wrap(u)
        return q
    report = dict(method="periodic multistart three-variable numerical IK",
                  max_starts=max_starts, random_seed=seed, starts_attempted=0,
                  physical_solution_limit=IK_MAX_SOLUTIONS,
                  completeness_certified=False, singular_solution_count=0,
                  position_tolerance=IK_TOLERANCE, duplicate_tolerance=IK_DUPLICATE_TOLERANCE,
                  singular_tolerance=IK_SINGULAR_TOLERANCE,
                  proven_unreachable=False)
    exclusion = reach_exclusion(goal, failed_joint, angle)
    if exclusion:
        report.update(proven_unreachable=True, reason=exclusion, termination_reason="geometric_reach_exclusion")
        return [], report
    seeds = []
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("Initial IK seed must contain four finite angles")
        seeds.append(model.wrap(initial[active]))
    # A coarse deterministic chart cover plus random starts, all within one
    # fixed budget. These are root-finding seeds, not collision regions.
    axis = -np.pi+(np.arange(5)+.5)*2*np.pi/5
    seeds.extend(np.array(q) for q in itertools.product(axis, repeat=3))
    rng = np.random.default_rng(seed)
    while len(seeds) < max_starts:
        seeds.append(rng.uniform(-np.pi, np.pi, 3))
    roots = []
    singular = []
    for u in seeds[:max_starts]:
        solved = least_squares(lambda u: model.position(expand(u))-goal, u,
                               jac=lambda u: model.jacobian(expand(u))[:, active],
                               method="lm", max_nfev=IK_MAX_EVALUATIONS,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        report["starts_attempted"] += 1
        q = expand(solved.x)
        if np.linalg.norm(model.position(q)-goal) > IK_TOLERANCE:
            continue
        if any(np.linalg.norm(model.wrap(q[active]-old[active])) < IK_DUPLICATE_TOLERANCE for old in roots):
            continue
        roots.append(q)
        singular.append(float(np.linalg.svd(model.jacobian(q)[:, active], compute_uv=False)[-1]))
        if len(roots) == IK_MAX_SOLUTIONS:
            break
    # Chart boundary representations must remain distinct GCS endpoints: no
    # opposite-boundary edges are added by this implementation.
    solutions = []
    for q in roots:
        choices = [([-np.pi, np.pi] if abs(abs(q[i])-np.pi) < 1e-9 else [q[i]]) for i in active]
        for values in itertools.product(*choices):
            candidate = q.copy()
            candidate[active] = values
            if np.linalg.norm(model.position(candidate)-goal) <= IK_TOLERANCE:
                solutions.append(candidate)
    report.update(physical_solution_count=len(roots), chart_solution_count=len(solutions),
                  termination_reason=("physical_solution_limit_reached" if len(roots) == IK_MAX_SOLUTIONS
                                      else "ik_start_budget_exhausted"),
                  minimum_active_jacobian_singular_values=singular,
                  singular_solution_count=sum(s < IK_SINGULAR_TOLERANCE for s in singular))
    return solutions, report


def slice_regions(regions, failed_joint, angle):
    active = [i for i in range(4) if i != failed_joint-1]
    slices = []
    for i, region in enumerate(regions):
        A = region.A[:, active]
        b = region.b-region.A[:, failed_joint-1]*angle
        if interior_point(A, b) is not None:
            slices.append(ReliablePolytope(A, b, parent_index=i))
    return slices


def plan_after_failure(q_failure, failed_joint, goal_point, checkpoint_path=None,
                       max_paths=MAX_ROUNDED_PATHS, max_trials=MAX_ROUNDING_TRIALS,
                       ik_max_starts=IK_MAX_STARTS, ik_seed=0, solver_backend="mosek"):
    began = time.perf_counter()
    supplied = np.asarray(q_failure, dtype=float)
    goal = np.asarray(goal_point, dtype=float)
    if (failed_joint not in (1, 2, 3, 4) or supplied.shape != (4,) or goal.shape != (3,)
            or not np.all(np.isfinite(supplied)) or not np.all(np.isfinite(goal))
            or max_paths < 1 or max_trials < 1
            or not isinstance(ik_max_starts, (int, np.integer)) or ik_max_starts < 1):
        raise ValueError("Use one failed joint 1..4, four finite angles, XYZ goal and positive search budgets")
    checkpoint_path = checkpoint_path or f"data/iris/spatial4R_{model.ROBOT_PARAMETER_SET}_iris_midpoint_regions.npz"
    q, offsets = chart_configuration(supplied)
    locked = failed_joint-1
    active = [i for i in range(4) if i != locked]
    report = dict(success=False, planner="spatial4R_post_failure_gcs", robot=model.PARAMETERS,
                  robot_parameter_set=model.ROBOT_PARAMETER_SET, failed_joint=failed_joint,
                  locked_angle=float(supplied[locked]), failure_configuration=supplied.tolist(),
                  start_configuration=supplied.tolist(), start_point=model.position(q).tolist(),
                  goal_point=goal.tolist(), active_joints=[i+1 for i in active],
                  iris_checkpoint=str(checkpoint_path), planning_chart_configuration=q.tolist(),
                  angle_winding_offsets=offsets.tolist(), physical_joint_limits=None,
                  cost_type="active_joint_space_euclidean_length", reliability_used=False,
                  gcs_solve_calls=0, ik_completeness_certified=False,
                  global_optimality_certified=False,
                  solver_settings=dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials))
    def finish(status, message, **values):
        report.update(status=status, message=message, total_seconds=time.perf_counter()-began, **values)
        return report
    exclusion = reach_exclusion(goal, failed_joint, q[locked])
    if exclusion:
        return finish("unreachable", exclusion+" Planning stopped before IK or GCS.")
    originals, metadata = load_iris_checkpoint(checkpoint_path, expected_dimension=4)
    saved = metadata["model"]
    if any(saved["dh"][name] != model.PARAMETERS[name] for name in ("a", "d", "alpha")):
        raise ValueError("Checkpoint DH geometry differs from the selected robot")
    obstacles, radius = saved["obstacles"], float(saved["link_radius"])
    report.update(obstacles=obstacles, link_radius=radius)
    if not model.collision_free(q, obstacles, radius):
        return finish("start_in_collision", "Failure configuration is in collision; planning stopped.")
    if np.linalg.norm(model.position(q)-goal) <= IK_TOLERANCE:
        return finish("success", "Already at the goal with the failed joint locked.", success=True,
                      waypoints=[supplied.tolist()], path_length=0., cost=0., endpoint_error=0.,
                      locked_joint_error=0., collision_free=True, collision_samples=1)
    ik_began = time.perf_counter()
    solutions, ik_report = locked_joint_ik(goal, failed_joint, q[locked], q, ik_max_starts, ik_seed)
    report.update(ik_seconds=time.perf_counter()-ik_began, ik_search=ik_report,
                  ik_solutions=[(p+offsets).tolist() for p in solutions])
    if ik_report["proven_unreachable"]:
        return finish("unreachable", ik_report["reason"]+" Planning stopped.")
    if ik_report["singular_solution_count"]:
        return finish("unsupported_singular_ik", "A rank-deficient goal IK was found. This discrete regular-IK workflow does not support it; planning stopped.")
    if not solutions:
        return finish("ik_search_failed", "No IK solution found within the numerical budget; this is not proof of unreachability. Planning stopped before GCS.")
    safe_goals = [p for p in solutions if model.collision_free(p, obstacles, radius)]
    report["collision_free_ik_solutions"] = [(p+offsets).tolist() for p in safe_goals]
    if not safe_goals:
        return finish("all_found_goals_in_collision", "Every found IK goal collides; IK enumeration is not certified complete. Planning stopped before GCS.")
    regions = slice_regions(originals, failed_joint, q[locked])
    edges = polytope_adjacency(regions)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    sources = [i for i, r in enumerate(regions) if r.contains(q[active])]
    reachable = set().union(*(nx.node_connected_component(graph, i) for i in sources)) if sources else set()
    connected = [(p, [i for i in sorted(reachable) if regions[i].contains(p[active])]) for p in safe_goals]
    connected = [(p, indices) for p, indices in connected if indices]
    report.update(slice_region_count=len(regions), connected_goal_count=len(connected))
    if not connected:
        return finish("no_path_in_iris_coverage", "No connection to any found safe IK goal in the current IRIS slices; planning stopped without generating regions or retrying.")
    keep = sorted(reachable)
    mapping = {old: new for new, old in enumerate(keep)}
    regions = [regions[i] for i in keep]
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in reachable and j in reachable]
    targets = [dict(polytope=sample_hull([p[active]]), region=mapping[indices[0]],
                    regions=[mapping[i] for i in indices]) for p, indices in connected]
    report.update(gcs_region_count=len(regions), gcs_edge_count=len(edges), gcs_solve_calls=1)
    print(f"Spatial post-failure: joint {failed_joint} locked, {len(regions)} 3D regions, "
          f"{len(connected)} found IK goals; one GCS solve", flush=True)
    solve_began = time.perf_counter()
    try:
        solved = solve_target_sets(regions, edges, q[active], targets, np.ones(len(regions)),
                                   max_rounded_paths=max_paths, max_rounding_trials=max_trials,
                                   solver_backend=solver_backend)
    except RuntimeError as exc:
        return finish("solver_failed", f"GCS failed; planning stopped without retrying: {exc}")
    report["solve_seconds"] = time.perf_counter()-solve_began
    if solved is None:
        return finish("solver_failed", "GCS returned no feasible rounded path; planning stopped without retrying.")
    u_path = np.asarray(solved["waypoints"])
    if u_path.ndim != 2 or u_path.shape[1] != 3 or not len(u_path) or not np.all(np.isfinite(u_path)):
        return finish("path_validation_failed", "GCS returned invalid path dimensions or values.")
    path = np.tile(supplied, (len(u_path), 1))
    path[:, active] = u_path+offsets[active]
    free, checked = model.validate_path(path, obstacles, radius)
    endpoint_error = float(np.linalg.norm(model.position(path[-1])-goal))
    start_error = float(np.linalg.norm(path[0]-supplied))
    lock_error = float(np.max(np.abs(path[:, locked]-supplied[locked])))
    length = float(np.linalg.norm(np.diff(u_path, axis=0), axis=1).sum())
    valid = free and max(endpoint_error, start_error, lock_error) <= 1e-6 and abs(length-solved["cost"]) <= 1e-5
    report.update(waypoints=path.tolist(), cost=solved["cost"], path_length=length,
                  collision_free=free, collision_samples=checked, endpoint_error=endpoint_error,
                  start_error=start_error, locked_joint_error=lock_error, solver_name=solved["solver_name"],
                  quality=dict(solved["quality"], scope="fixed_start_to_found_ik_goals_in_iris_slice_graph"),
                  selected_goal_index=int(solved["target"]), rounded_path_count=solved["rounded_path_count"])
    return finish("success" if valid else "path_validation_failed",
                  "Locked-joint path validated; IK search and graph coverage remain uncertified."
                  if valid else "Path failed collision, endpoint, locked-angle or length checks.", success=bool(valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-joint", type=int, choices=[1, 2, 3, 4], required=True)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--q", type=float, nargs=4)
    state.add_argument("--path", help="Saved pre-failure report; defaults to the current ftwmax midpoint result")
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--goal", type=float, nargs=3)
    parser.add_argument("--robot", choices=["ftwmax", "rcmax"])
    parser.add_argument("--iris-regions")
    parser.add_argument("--ik-starts", type=int, default=IK_MAX_STARTS)
    parser.add_argument("--ik-seed", type=int, default=0)
    parser.add_argument("--max-rounded-paths", type=int, default=MAX_ROUNDED_PATHS)
    parser.add_argument("--max-rounding-trials", type=int, default=MAX_ROUNDING_TRIALS)
    parser.add_argument("--solver", choices=["mosek", "clarabel"], default="mosek")
    parser.add_argument("--output", default='results/spatial4r/spatial4R_post_failure_path.json')
    args = parser.parse_args()
    try:
        saved = None
        if args.q is None:
            saved = json.loads(Path(args.path or 'results/spatial4r/spatial4R_ftwmax_midpoint_two_stage_path.json').read_text(encoding="utf-8"))
            if not saved.get("success") or not 0 <= args.waypoint_index < len(saved["waypoints"]):
                raise ValueError("Expected a successful pre-failure path and valid waypoint index")
        robot = args.robot or (saved.get("robot_parameter_set", "ftwmax") if saved else "ftwmax")
        model.select_robot(robot)
        if saved and any(saved["robot"][name] != model.PARAMETERS[name] for name in ("a", "d", "alpha")):
            raise ValueError("Input path robot differs from the selected robot")
        q = args.q if args.q is not None else saved["waypoints"][args.waypoint_index]
        goal = args.goal if args.goal is not None else (saved["goal_point"] if saved else model.position(model.GOAL_REFERENCE_CONFIGURATION))
        checkpoint = args.iris_regions or (saved["iris_checkpoint"] if saved else None)
        if saved:
            # Preserve the original physical scene and reject mismatches before
            # any IK or GCS solve. Obstacles do not move when a joint locks.
            _, checkpoint_metadata = load_iris_checkpoint(checkpoint, expected_dimension=4)
            scene = checkpoint_metadata["model"]
            if scene["obstacles"] != saved["obstacles"] or scene["link_radius"] != saved["link_radius"]:
                raise ValueError("Input path and checkpoint collision scenes differ")
        result = plan_after_failure(q, args.failed_joint, goal, checkpoint,
                                    args.max_rounded_paths, args.max_rounding_trials,
                                    args.ik_starts, args.ik_seed, args.solver)
    except (ValueError, KeyError, OSError, RuntimeError, IndexError) as exc:
        result = dict(success=False, status="invalid_input", message=str(exc), gcs_solve_calls=0)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
