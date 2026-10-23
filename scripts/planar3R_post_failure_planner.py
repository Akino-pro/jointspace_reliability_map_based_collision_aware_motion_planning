"""One-shot replanning for one locked joint of the planar 3R robot."""

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from scripts.planar3R_collision import L, is_collision_free, link_points, wrap_to_pi
from scripts.planar3R_iris_regions import interior_point, load_iris_checkpoint, polytope_adjacency
from scripts.periodic_graph import remap_edges


MAX_ROUNDED_PATHS = 200
MAX_ROUNDING_TRIALS = 300


@dataclass
class LockedSlice:
    A: np.ndarray
    b: np.ndarray

    def contains(self, u, tol=1e-8):
        return bool(np.all(self.A @ np.asarray(u).reshape(2) <= self.b + tol))

    def to_drake(self):
        from pydrake.geometry.optimization import HPolyhedron
        return HPolyhedron(self.A, self.b)


def locked_joint_ik(goal, failed_joint, angle, lengths=L, tolerance=1e-9):
    """Return original 3R configurations, not straight-link collision proxies.

    Joint numbers are 1-based. Generic IK has at most two physical solutions.
    Equal-length folded and zero-effective-length cases can have continuous IK;
    report those explicitly instead of silently choosing arbitrary endpoints.
    """
    if failed_joint not in (1, 2, 3):
        raise ValueError("Exactly one failed joint numbered 1, 2, or 3 is required")
    l1, l2, l3 = np.asarray(lengths, dtype=float)
    target = np.asarray(goal, dtype=float).reshape(2)
    offset = 0.0
    if failed_joint == 1:
        c, s = np.cos(angle), np.sin(angle)
        target = np.array([[c, s], [-s, c]]) @ (target - l1 * np.array([c, s]))
        first, second = l2, l3
    elif failed_joint == 2:
        combined = l1 + l2 * np.exp(1j * angle)
        first, second = abs(combined), l3
        offset = np.angle(combined)
    else:
        combined = l2 + l3 * np.exp(1j * angle)
        first, second = l1, abs(combined)
        offset = np.angle(combined)
    radius = np.linalg.norm(target)
    if radius > first + second + tolerance or radius < abs(first - second) - tolerance:
        return [], False
    if first <= tolerance or second <= tolerance or radius <= tolerance:
        return [], True
    cosine = (radius**2 - first**2 - second**2) / (2 * first * second)
    elbow = np.arccos(np.clip(cosine, -1.0, 1.0))
    solutions = []
    for beta in (elbow, -elbow):
        shoulder = np.arctan2(target[1], target[0]) - np.arctan2(
            second * np.sin(beta), first + second * np.cos(beta)
        )
        if failed_joint == 1:
            q = np.array([angle, shoulder, beta])
        elif failed_joint == 2:
            q = np.array([shoulder - offset, angle, beta - angle + offset])
        else:
            q = np.array([shoulder, beta - offset, angle])
        active = [i for i in range(3) if i != failed_joint - 1]
        q[active] = wrap_to_pi(q[active])
        # Both +/-pi representations are valid endpoints of the current chart.
        # Connections across that boundary are an optional graph feature.
        choices = [([-np.pi, np.pi] if abs(abs(q[i]) - np.pi) < tolerance else [q[i]])
                   for i in active]
        for values in itertools.product(*choices):
            candidate = q.copy()
            candidate[active] = values
            if np.linalg.norm(link_points(candidate, lengths)[-1] - goal) > 1e-7:
                continue
            if not any(np.linalg.norm(candidate - old) < tolerance for old in solutions):
                solutions.append(candidate)
    return solutions, False


def slice_regions(regions, failed_joint, angle):
    locked = failed_joint - 1
    active = [i for i in range(3) if i != locked]
    slices = []
    for region in regions:
        A = region.A[:, active]
        b = region.b - region.A[:, locked] * angle
        if interior_point(A, b) is not None:
            slices.append(LockedSlice(A, b))
    return slices


def solve_to_any_goal(regions, edges, start, goals, source_indices, goal_indices):
    """One GCS solve with distinct goal points feeding a zero-cost dummy sink."""
    from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
    from scripts.planar3R_ssm_envelopes import sample_hull
    targets = [dict(polytope=sample_hull([q]), region=indices[0], regions=indices)
               for q, indices in zip(goals, goal_indices)]
    solved = solve_target_sets(regions, edges, start, targets, np.ones(len(regions)),
                               max_rounded_paths=MAX_ROUNDED_PATHS,
                               max_rounding_trials=MAX_ROUNDING_TRIALS)
    if solved is None:
        return None, None
    return solved["waypoints"], solved["cost"]


def plan_after_failure(q_failure, failed_joint, goal_point,
                       checkpoint_path='data/iris/planar3R_iris_regions.npz', periodic=False):
    started = time.perf_counter()
    q_failure = np.asarray(q_failure, dtype=float)
    goal = np.asarray(goal_point, dtype=float)
    if (failed_joint not in (1, 2, 3) or q_failure.shape != (3,) or goal.shape != (2,)
            or not np.all(np.isfinite(q_failure)) or not np.all(np.isfinite(goal))):
        raise ValueError("Use one failed joint (1..3), three finite angles in radians, and a 2D target")
    supplied_configuration = q_failure.copy()
    # Revolute joints have no physical angle stops. Use the cached coordinate
    # chart internally, then restore each supplied winding number in the output.
    outside_chart = np.abs(q_failure) > np.pi
    q_failure = np.where(outside_chart, wrap_to_pi(q_failure), q_failure)
    positive_boundary = outside_chart & (supplied_configuration > 0) & np.isclose(q_failure, -np.pi, atol=1e-12, rtol=0)
    q_failure[positive_boundary] = np.pi
    winding_offsets = supplied_configuration - q_failure
    locked = failed_joint - 1
    active = [i for i in range(3) if i != locked]
    angle = float(q_failure[locked])
    report = {"success": False, "failed_joint": failed_joint,
              "locked_angle": float(supplied_configuration[locked]),
              "failure_configuration": supplied_configuration.tolist(), "goal_point": goal.tolist(),
              "active_joints": [i + 1 for i in active], "iris_checkpoint": str(checkpoint_path),
              "cost_type": "active_joint_space_euclidean_length", "gcs_solve_calls": 0,
              "solver_settings": {"max_rounded_paths": MAX_ROUNDED_PATHS,
                                  "max_rounding_trials": MAX_ROUNDING_TRIALS},
              "planning_chart_configuration": q_failure.tolist(),
              "angle_winding_offsets": winding_offsets.tolist()}

    def finish(status, message, **values):
        report.update(status=status, message=message, total_seconds=time.perf_counter() - started, **values)
        return report

    originals, metadata = load_iris_checkpoint(checkpoint_path)
    model = metadata["model"]
    if model.get("link_lengths") != list(L) or len(model["link_lengths"]) != 3:
        raise ValueError("Checkpoint must match the current planar 3R robot")
    obstacles, link_radius = model["obstacles"], float(model["link_radius"])
    report.update(link_lengths=list(L), obstacles=obstacles, link_radius=link_radius)
    if not is_collision_free(q_failure, obstacles, link_radius=link_radius):
        return finish("start_in_collision", "The robot configuration at failure is in collision; planning stopped.")
    if np.linalg.norm(link_points(q_failure)[-1] - goal) <= 1e-8:
        return finish("success", "The failure configuration is already at the goal.", success=True,
                      waypoints=[supplied_configuration.tolist()], path_length=0.0, cost=0.0, endpoint_error=0.0)
    solutions, degenerate = locked_joint_ik(goal, failed_joint, angle)
    report["ik_solutions"] = [(q + winding_offsets).tolist() for q in solutions]
    if degenerate:
        return finish("unsupported_degenerate_ik", "Degenerate kinematics produce a continuous goal set; discrete-goal planning does not support it, so planning stopped.")
    if not solutions:
        return finish("unreachable", "The goal position is unreachable with the joint locked; planning stopped.")
    safe_goals = [q for q in solutions if is_collision_free(q, obstacles, link_radius=link_radius)]
    report["collision_free_ik_solutions"] = [(q + winding_offsets).tolist() for q in safe_goals]
    if not safe_goals:
        return finish("all_goals_in_collision", "All goal IK solutions are in collision; planning stopped.")
    regions = slice_regions(originals, failed_joint, angle)
    adjacency_statistics = {}
    edges = polytope_adjacency(regions, periodic=periodic, statistics=adjacency_statistics)
    report.update(periodic_connections=periodic, adjacency_statistics=adjacency_statistics)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    source_indices = [i for i, r in enumerate(regions) if r.contains(q_failure[active])]
    reachable = set()
    for index in source_indices:
        reachable.update(nx.node_connected_component(graph, index))
    connected_goals, goal_indices = [], []
    for q in safe_goals:
        indices = [i for i in sorted(reachable) if regions[i].contains(q[active])]
        if indices:
            connected_goals.append(q)
            goal_indices.append(indices)
    report.update(slice_region_count=len(regions), connected_goal_count=len(connected_goals))
    if not connected_goals:
        return finish("no_path_in_iris_coverage", "No collision-free path connects the endpoints within current IRIS coverage; planning stopped without generating additional regions.")
    keep = sorted(reachable)
    mapping = {old: new for new, old in enumerate(keep)}
    pair_regions = [regions[i] for i in keep]
    pair_edges = remap_edges(edges, mapping)
    report.update(gcs_region_count=len(keep), gcs_edge_count=len(pair_edges), gcs_solve_calls=1)
    print(f"Post-failure GCS: {len(keep)} 2D regions, {len(connected_goals)} IK goals", flush=True)
    try:
        u_path, cost = solve_to_any_goal(
            pair_regions, pair_edges, q_failure[active], [q[active] for q in connected_goals],
            [mapping[i] for i in source_indices], [[mapping[i] for i in indices] for indices in goal_indices],
        )
    except Exception as exc:
        return finish("solver_failed", f"GCS solve failed; planning stopped: {exc}")
    if u_path is None:
        return finish("solver_failed", "GCS returned no feasible path; planning stopped without retrying.")
    u_path = np.asarray(u_path, dtype=float)
    if (u_path.ndim != 2 or u_path.shape[1] != 2 or len(u_path) < 2
            or not np.all(np.isfinite(u_path))):
        return finish("path_validation_failed", "GCS returned a path with invalid dimensions or values; planning stopped without retrying.")
    q_path = np.tile(supplied_configuration, (len(u_path), 1))
    q_path[:, active] = u_path + winding_offsets[active]
    from scripts.planar3R_iris_gcs_planner import validate_joint_path
    collision_free, samples = validate_joint_path(q_path, obstacles, link_radius)
    endpoint_error = float(np.linalg.norm(link_points(q_path[-1])[-1] - goal))
    if not collision_free or endpoint_error > 1e-6 or np.linalg.norm(q_path[0] - supplied_configuration) > 1e-6:
        return finish("path_validation_failed", "The path failed collision or endpoint checks; planning stopped without retrying.")
    length = float(np.sum(np.linalg.norm(np.diff(u_path, axis=0), axis=1)))
    return finish("success", "Post-failure planning succeeded; the locked joint remains fixed throughout the path.", success=True,
                  waypoints=q_path.tolist(), path_length=length, cost=cost,
                  endpoint_error=endpoint_error, collision_samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-joint", type=int, choices=[1, 2, 3], required=True)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--q", nargs=3, type=float, help="Failure-time joint angles in radians")
    state.add_argument("--path", help="Use a waypoint of a pre-failure JSON path (default iris_path.json)")
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--gx", type=float)
    parser.add_argument("--gy", type=float)
    parser.add_argument("--iris-regions", default='data/iris/planar3R_iris_regions.npz')
    parser.add_argument("--periodic", action="store_true", help="Optional active-joint chart connections")
    parser.add_argument("--output", default='results/planar3r/post_failure_path.json')
    args = parser.parse_args()
    if (args.gx is None) != (args.gy is None):
        parser.error("--gx and --gy must be specified together")
    try:
        if args.q is not None:
            from scripts.planar3R_gcs_planner import GOAL_POINT
            q, target = args.q, GOAL_POINT
        else:
            source = json.loads(Path(args.path or 'results/planar3r/iris_path.json').read_text(encoding="utf-8"))
            if args.waypoint_index < 0 or args.waypoint_index >= len(source["waypoints"]):
                raise ValueError("waypoint-index is outside the saved path")
            q, target = source["waypoints"][args.waypoint_index], source["goal_point"]
        if args.gx is not None:
            target = [args.gx, args.gy]
        report = plan_after_failure(q, args.failed_joint, target, args.iris_regions, periodic=args.periodic)
    except (ValueError, KeyError, OSError) as exc:
        report = {"success": False, "status": "invalid_input", "message": str(exc), "gcs_solve_calls": 0}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
