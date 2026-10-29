"""Plan after one 7R joint locks, minimizing unweighted active-joint length.

Regular intersections of the full-pose goal SSMs with the locked-angle plane
are discrete six-variable IK roots. All found safe, connected roots compete in
one GCS graph. Numerical root discovery and IRIS coverage are not complete.
"""

import argparse
import itertools
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts import spatial7R_model as model
from scripts.planar3R_iris_regions import (
    ReliablePolytope, interior_point, load_iris_checkpoint, polytope_adjacency,
)
from scripts.planar3R_ssm_envelopes import sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
from scripts.spatial4R_post_failure_planner import chart_configuration

MAX_ROUNDED_PATHS = 200
MAX_ROUNDING_TRIALS = 300
IK_MAX_STARTS = 256
IK_TOLERANCE = 1e-8
IK_SINGULAR_TOLERANCE = 1e-7
DEFAULT_CHECKPOINT = "data/iris/spatial7R_60cm90deg_iris_regions.npz"
DEFAULT_PATH = "results/spatial7r/spatial7R_60cm90deg_two_stage_path.json"


def reach_exclusion(goal, failed_joint, angle):
    position = goal[:3, 3]
    if np.linalg.norm(position) > np.hypot(model.A, model.D).sum()+IK_TOLERANCE:
        return "Goal exceeds the arm's conservative radial reach bound."
    if failed_joint == 1:
        q = np.zeros(7)
        q[0] = angle
        center = model.kinematics(q)[3][1][1]
        if np.linalg.norm(position-center) > np.hypot(model.A[1:], model.D[1:]).sum()+IK_TOLERANCE:
            return "Goal exceeds the remaining arm's reach about the locked first DH endpoint."
    return None


def locked_joint_ik(goal, failed_joint, angle, initial=None,
                    max_starts=IK_MAX_STARTS, seed=0):
    """Enumerate found regular roots across SSM branches; use the full budget.

    There is no inherited four-root cap: a 6R full-pose problem can have more
    solutions. Root indices identify endpoints, not certified original SSM IDs.
    """
    goal = model.validate_pose(goal)
    if (failed_joint not in range(1, 8) or not np.isfinite(angle)
            or not isinstance(max_starts, (int, np.integer)) or max_starts < 1):
        raise ValueError("Use joint 1..7, a finite angle and a positive integer IK budget")
    active = [i for i in range(7) if i != failed_joint-1]
    base = np.zeros(7)
    base[failed_joint-1] = angle

    def expand(u):
        q = base.copy()
        q[active] = model.wrap(u)
        return q

    seeds = []
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("Initial IK seed must contain seven finite angles")
        seeds.append(model.wrap(initial[active]))
    # The reference is a seed only, never a substitute for the requested pose.
    seeds.append(model.wrap(model.GOAL_REFERENCE_CONFIGURATION[active]))
    rng = np.random.default_rng(seed)
    while len(seeds) < max_starts:
        seeds.append(rng.uniform(-np.pi, np.pi, 6))
    report = dict(method="periodic multistart six-variable full-pose numerical IK",
                  max_starts=max_starts, random_seed=seed, starts_attempted=0,
                  completeness_certified=False, proven_unreachable=False,
                  position_tolerance=IK_TOLERANCE, orientation_tolerance=IK_TOLERANCE,
                  singular_solution_count=0)
    exclusion = reach_exclusion(goal, failed_joint, angle)
    if exclusion:
        report.update(proven_unreachable=True, reason=exclusion)
        return [], report
    roots, singular_values = [], []
    for u in seeds[:max_starts]:
        solved = least_squares(lambda x: model.pose_residual(expand(x), goal), u,
                               jac=lambda x: model.residual_jacobian(expand(x), goal)[:, active],
                               method="lm", max_nfev=200,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        report["starts_attempted"] += 1
        q = expand(solved.x)
        if np.linalg.norm(model.pose_residual(q, goal)) > IK_TOLERANCE:
            continue
        if any(np.linalg.norm(model.wrap(q[active]-old[active])) < 1e-5 for old in roots):
            continue
        roots.append(q)
        singular_values.append(float(np.linalg.svd(model.jacobian(q)[:, active], compute_uv=False)[-1]))
    solutions = []
    for q, singular in zip(roots, singular_values):
        if singular < IK_SINGULAR_TOLERANCE:
            continue
        choices = [([-np.pi, np.pi] if abs(abs(q[i])-np.pi) < 1e-9 else [q[i]]) for i in active]
        for values in itertools.product(*choices):
            candidate = q.copy()
            candidate[active] = values
            if np.linalg.norm(model.pose_residual(candidate, goal)) <= IK_TOLERANCE:
                solutions.append(candidate)
    report.update(physical_solution_count=len(roots), chart_solution_count=len(solutions),
                  singular_solution_count=sum(s < IK_SINGULAR_TOLERANCE for s in singular_values),
                  minimum_active_jacobian_singular_values=singular_values,
                  termination_reason="ik_start_budget_exhausted")
    return solutions, report


def slice_regions(regions, failed_joint, angle):
    if failed_joint not in range(1, 8) or not np.isfinite(angle):
        raise ValueError("Use joint 1..7 and a finite angle")
    active = [i for i in range(7) if i != failed_joint-1]
    slices = []
    for index, region in enumerate(regions):
        A = region.A[:, active]
        b = region.b-region.A[:, failed_joint-1]*angle
        if interior_point(A, b) is not None:
            slices.append(ReliablePolytope(A, b, parent_index=index))
    return slices


def plan_after_failure(q_failure, failed_joint, goal_pose, checkpoint_path=DEFAULT_CHECKPOINT,
                       max_paths=MAX_ROUNDED_PATHS, max_trials=MAX_ROUNDING_TRIALS,
                       ik_max_starts=IK_MAX_STARTS, ik_seed=0, solver_backend="mosek"):
    began = time.perf_counter()
    supplied = np.asarray(q_failure, dtype=float)
    goal = model.validate_pose(goal_pose)
    if (failed_joint not in range(1, 8) or supplied.shape != (7,)
            or not np.all(np.isfinite(supplied))
            or any(not isinstance(n, (int, np.integer)) or n < 1
                   for n in (max_paths, max_trials, ik_max_starts))
            or solver_backend not in ("mosek", "clarabel")):
        raise ValueError("Use joint 1..7, seven finite angles, positive integer budgets and a supported solver")
    q, offsets = chart_configuration(supplied)
    locked = failed_joint-1
    active = [i for i in range(7) if i != locked]
    report = dict(success=False, planner="spatial7R_post_failure_gcs", robot=model.PARAMETERS,
                  task="full_pose_SE3", failed_joint=failed_joint,
                  locked_angle=float(supplied[locked]), failure_configuration=supplied.tolist(),
                  start_configuration=supplied.tolist(), goal_pose=goal.tolist(),
                  goal_point=goal[:3, 3].tolist(), active_joints=[i+1 for i in active],
                  iris_checkpoint=str(checkpoint_path), planning_chart_configuration=q.tolist(),
                  angle_winding_offsets=offsets.tolist(), physical_joint_limits=None,
                  cost_type="active_joint_space_euclidean_length", reliability_used=False,
                  gcs_solve_calls=0, ik_completeness_certified=False,
                  global_optimality_certified=False,
                  solver_settings=dict(max_rounded_paths=max_paths, max_rounding_trials=max_trials,
                                       solver_backend=solver_backend))

    def finish(status, message, **values):
        report.update(status=status, message=message, total_seconds=time.perf_counter()-began, **values)
        return report

    exclusion = reach_exclusion(goal, failed_joint, q[locked])
    if exclusion:
        return finish("unreachable", exclusion)
    originals, metadata = load_iris_checkpoint(checkpoint_path, expected_dimension=7)
    saved = metadata["model"]
    if any(saved["dh"][name] != model.PARAMETERS[name] for name in ("a", "d", "alpha")):
        raise ValueError("Checkpoint DH geometry differs from the 7R robot")
    obstacles, radius = saved["obstacles"], float(saved["link_radius"])
    report.update(obstacles=obstacles, link_radius=radius, self_collision=False)
    if not model.collision_free(q, obstacles, radius):
        return finish("start_in_collision", "Failure configuration is in collision.")
    if np.linalg.norm(model.pose_residual(q, goal)) <= IK_TOLERANCE:
        return finish("success", "Already at the full-pose goal.", success=True,
                      waypoints=[supplied.tolist()], cost=0., path_length=0., endpoint_error=0.,
                      orientation_error=0., locked_joint_error=0., start_error=0.,
                      collision_free=True, collision_samples=1)
    ik_began = time.perf_counter()
    solutions, ik_report = locked_joint_ik(goal, failed_joint, q[locked], q, ik_max_starts, ik_seed)
    report.update(ik_seconds=time.perf_counter()-ik_began, ik_search=ik_report,
                  ik_solutions=[(p+offsets).tolist() for p in solutions])
    if ik_report["singular_solution_count"]:
        return finish("unsupported_singular_ik", "Rank-deficient locked-joint IK found; discrete regular-goal planning cannot represent its solution set.")
    if not solutions:
        return finish("ik_search_failed", "No regular full-pose root found within the budget; unreachability is not established.")
    safe = [(i, p) for i, p in enumerate(solutions) if model.collision_free(p, obstacles, radius)]
    if not safe:
        return finish("all_found_goals_in_collision", "Every found regular goal collides; root discovery is incomplete.")
    regions = slice_regions(originals, failed_joint, q[locked])
    edges = polytope_adjacency(regions)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    sources = [i for i, r in enumerate(regions) if r.contains(q[active])]
    reachable = set().union(*(nx.node_connected_component(graph, i) for i in sources)) if sources else set()
    connected = [(root, p, [i for i in sorted(reachable) if regions[i].contains(p[active])])
                 for root, p in safe]
    connected = [item for item in connected if item[2]]
    report.update(slice_region_count=len(regions), collision_free_goal_count=len(safe),
                  connected_goal_count=len(connected),
                  connected_goals=[dict(ik_solution_index=root, configuration=(p+offsets).tolist())
                                   for root, p, _ in connected])
    if not connected:
        return finish("no_path_in_iris_coverage", "No found safe goal connects through the original IRIS slices; no new regions or retries were added.")
    keep = sorted(reachable)
    mapping = {old: new for new, old in enumerate(keep)}
    regions = [regions[i] for i in keep]
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in reachable and j in reachable]
    targets = [dict(polytope=sample_hull([p[active]]), region=mapping[indices[0]],
                    regions=[mapping[i] for i in indices]) for _, p, indices in connected]
    report.update(gcs_region_count=len(regions), gcs_edge_count=len(edges), gcs_solve_calls=1)
    print(f"7R post-failure: joint {failed_joint} locked, {len(regions)} 6D regions, "
          f"{len(targets)} goal roots; unweighted GCS", flush=True)
    solve_began = time.perf_counter()
    try:
        solved = solve_target_sets(regions, edges, q[active], targets, np.ones(len(regions)),
                                   max_rounded_paths=max_paths, max_rounding_trials=max_trials,
                                   solver_backend=solver_backend)
    except RuntimeError as exc:
        return finish("solver_failed", str(exc))
    report["solve_seconds"] = time.perf_counter()-solve_began
    if solved is None:
        return finish("solver_failed", "No feasible rounded GCS path returned.")
    u_path = np.asarray(solved["waypoints"])
    selected, route = int(solved["target"]), solved["region_route"]
    if (u_path.shape != (len(route)+1, 6) or not np.all(np.isfinite(u_path))
            or not 0 <= selected < len(targets) or not route
            or any(not 0 <= index < len(regions) for index in route)):
        return finish("path_validation_failed", "Invalid GCS path dimensions, target or route.")
    membership = (np.linalg.norm(u_path[0]-q[active]) <= 1e-7
                  and targets[selected]["polytope"].contains(u_path[-1], tol=1e-7)
                  and all(regions[index].contains(u_path[i], tol=1e-7)
                          and regions[index].contains(u_path[i+1], tol=1e-7)
                          for i, index in enumerate(route))
                  and all(pair in edges for pair in zip(route[:-1], route[1:])))
    path = np.tile(supplied, (len(u_path), 1))
    path[:, active] = u_path+offsets[active]
    free, checked = model.validate_path(path, obstacles, radius)
    residual = model.pose_residual(path[-1], goal)
    endpoint_error, orientation_error = map(float, (np.linalg.norm(residual[:3]), np.linalg.norm(residual[3:])))
    start_error = float(np.linalg.norm(path[0]-supplied))
    lock_error = float(np.max(np.abs(path[:, locked]-supplied[locked])))
    length = float(np.linalg.norm(np.diff(u_path, axis=0), axis=1).sum())
    valid = bool(membership and free and max(endpoint_error, orientation_error, start_error, lock_error) <= 1e-6
                 and np.isfinite(solved["cost"]) and abs(length-solved["cost"]) <= 1e-5)
    report.update(waypoints=path.tolist(), cost=float(solved["cost"]), path_length=length,
                  graph_membership_valid=bool(membership), region_route=route,
                  collision_free=free, collision_samples=checked, endpoint_error=endpoint_error,
                  orientation_error=orientation_error, start_error=start_error, locked_joint_error=lock_error,
                  selected_goal_index=selected, selected_ik_solution_index=connected[selected][0],
                  solver_name=solved["solver_name"], rounded_path_count=solved["rounded_path_count"],
                  quality=dict(solved["quality"], scope="fixed_start_to_found_full_pose_ik_goals_in_iris_slice_graph"))
    return finish("success" if valid else "path_validation_failed",
                  "Locked-joint full-pose path validated." if valid else "Path failed validation.", success=valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-joint", type=int, choices=range(1, 8), required=True)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--q", type=float, nargs=7)
    state.add_argument("--path", help="Successful pre-failure report")
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--goal-position", type=float, nargs=3)
    parser.add_argument("--goal-quaternion", type=float, nargs=4, help="Quaternion X Y Z W")
    parser.add_argument("--iris-regions")
    parser.add_argument("--ik-starts", type=int, default=IK_MAX_STARTS)
    parser.add_argument("--ik-seed", type=int, default=0)
    parser.add_argument("--max-rounded-paths", type=int, default=MAX_ROUNDED_PATHS)
    parser.add_argument("--max-rounding-trials", type=int, default=MAX_ROUNDING_TRIALS)
    parser.add_argument("--solver", choices=["mosek", "clarabel"], default="mosek")
    parser.add_argument("--output", default="results/spatial7r/spatial7R_post_failure_path.json")
    args = parser.parse_args()
    try:
        saved = None
        if args.q is None:
            saved = json.loads(Path(args.path or DEFAULT_PATH).read_text(encoding="utf-8"))
            if not saved.get("success") or not 0 <= args.waypoint_index < len(saved["waypoints"]):
                raise ValueError("Expected a successful pre-failure report and valid waypoint index")
            if any(saved["robot"][name] != model.PARAMETERS[name] for name in ("a", "d", "alpha")):
                raise ValueError("Input path robot differs from the 7R robot")
        if (args.goal_position is None) != (args.goal_quaternion is None):
            raise ValueError("Supply both goal position and quaternion")
        goal = np.asarray(saved["goal_pose"]) if saved else model.pose(model.GOAL_REFERENCE_CONFIGURATION)
        if args.goal_position is not None:
            goal = np.eye(4)
            goal[:3, :3] = Rotation.from_quat(args.goal_quaternion).as_matrix()
            goal[:3, 3] = args.goal_position
        q = args.q if args.q is not None else saved["waypoints"][args.waypoint_index]
        checkpoint = args.iris_regions or (saved["iris_checkpoint"] if saved else DEFAULT_CHECKPOINT)
        if saved:
            _, metadata = load_iris_checkpoint(checkpoint, expected_dimension=7)
            scene = metadata["model"]
            if scene["obstacles"] != saved["obstacles"] or scene["link_radius"] != saved["link_radius"]:
                raise ValueError("Input path and checkpoint collision scenes differ")
        result = plan_after_failure(q, args.failed_joint, goal, checkpoint,
                                    args.max_rounded_paths, args.max_rounding_trials,
                                    args.ik_starts, args.ik_seed, args.solver)
    except (ValueError, KeyError, OSError, RuntimeError, IndexError) as exc:
        result = dict(success=False, status="invalid_input", message=str(exc), gcs_solve_calls=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
