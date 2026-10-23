"""Reliability-aware GCS using cached IRIS polytopes."""

import time

import networkx as nx
import numpy as np

from scripts.planar3R_collision import L, is_collision_free, link_points
from scripts.planar3R_iris_regions import (
    load_iris_checkpoint,
    polytope_adjacency,
    split_iris_regions,
)


class PlanningStopped(RuntimeError):
    """A preflight failure with a machine-readable result, without a GCS solve."""

    def __init__(self, report):
        super().__init__(report["message"])
        self.report = report


def select_connected_candidates(graph, start_pairs, goal_pairs, limit):
    """Spend the candidate budget on common connected components first.

    Adjacency is bidirectional. Selecting the first few SSM samples before this
    check can put all starts and goals in different components despite coverage.
    """
    components = list(nx.connected_components(graph.to_undirected()))
    membership = {node: i for i, nodes in enumerate(components) for node in nodes}
    start_groups, goal_groups = {}, {}
    for pairs, groups in ((start_pairs, start_groups), (goal_pairs, goal_groups)):
        for q, indices in pairs:
            for component in sorted({membership[i] for i in indices}):
                groups.setdefault(component, []).append((q, indices))
    common = [component for component in goal_groups if component in start_groups]
    selected_starts, selected_goals = [], []
    for offset in range(limit):
        for component in common:
            for groups, selected in ((start_groups, selected_starts), (goal_groups, selected_goals)):
                candidates = groups[component]
                if offset < len(candidates) and len(selected) < limit:
                    selected.append(candidates[offset])
    return selected_starts, selected_goals


def validate_joint_path(waypoints, obstacles, link_radius=0.0, step=0.01):
    """Check the GCS's straight segments in its Euclidean joint-angle chart.

    This is an independent sampled check, not a continuous collision proof.
    Do not wrap angles: wrapping would change the path optimized by GCS.
    """
    if step <= 0:
        raise ValueError("Validation step must be positive.")
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError("A path needs at least two planar 3R waypoints.")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("Path waypoints must be finite.")
    checked = 0
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        intervals = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
        for alpha in np.linspace(0.0, 1.0, intervals + 1):
            q = start + alpha * (end - start)
            checked += 1
            if not is_collision_free(q, obstacles, link_radius=link_radius):
                return False, checked
    return True, checked


def plan_with_iris_gcs(
    start_point, goal_point, obstacles, checkpoint_path,
    reliability_source="goal", min_reliability=0.0, seed=0,
    candidate_limit=3, reliability_epsilon=0.01, rounding=True,
):
    from scripts.planar3R_gcs_planner import (
        reliability_cost_weight,
        reliability_region_sets_from_source,
        ssm_candidate_points,
    )

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive.")
    started = time.perf_counter()
    # With unrestricted revolute joints, radial reachability follows the
    # polygon inequality. No artificial joint-angle box is imposed.
    reach = float(sum(L))
    inner_radius = max(0.0, 2.0 * max(L) - reach)
    for name, point in (("goal", goal_point), ("start", start_point)):
        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
            raise ValueError("Workspace endpoints must be finite 2D positions")
        radius = np.linalg.norm(point_array)
        if radius > reach + 1e-10 or radius < inner_radius - 1e-10:
            raise PlanningStopped({
                "success": False, "status": f"{name}_unreachable",
                "message": f"The {name} position is outside the robot's reachable workspace; planning stopped.",
                "start_point": list(start_point), "goal_point": list(goal_point), "gcs_solve_calls": 0,
                "total_seconds": time.perf_counter() - started,
            })
    original, metadata = load_iris_checkpoint(checkpoint_path)
    model = metadata.get("model", {})
    if model.get("link_lengths") != list(L) or model.get("obstacles") != [
        {"type": o.get("type", "circle"), "center": list(o["center"]), "radius": o["radius"]}
        for o in obstacles
    ]:
        raise ValueError("IRIS checkpoint geometry does not match the current planning scene.")
    link_radius = float(model["link_radius"])
    partition_sets = reliability_region_sets_from_source(
        reliability_source, start_point, goal_point, seed
    )
    starts = [q for q in ssm_candidate_points(*start_point, obstacles, seed=seed)
              if is_collision_free(q, obstacles, link_radius=link_radius)]
    goals = [q for q in ssm_candidate_points(*goal_point, obstacles, seed=seed + 1)
             if is_collision_free(q, obstacles, link_radius=link_radius)]
    best, failures, statistics = None, [], []
    solve_calls = 0

    for partition_set in partition_sets:
        split_started = time.perf_counter()
        regions = split_iris_regions(original, partition_set["regions"], min_reliability)
        split_seconds = time.perf_counter() - split_started
        edge_started = time.perf_counter()
        edges = polytope_adjacency(regions)
        edge_seconds = time.perf_counter() - edge_started
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(regions)))
        graph.add_edges_from(edges)
        start_pairs = [(q, [i for i, r in enumerate(regions) if r.contains(q)]) for q in starts]
        goal_pairs = [(q, [i for i, r in enumerate(regions) if r.contains(q)]) for q in goals]
        start_pairs = [(q, indices) for q, indices in start_pairs if indices]
        goal_pairs = [(q, indices) for q, indices in goal_pairs if indices]
        start_pairs, goal_pairs = select_connected_candidates(graph, start_pairs, goal_pairs, candidate_limit)
        source_indices = {i for _, indices in start_pairs for i in indices}
        target_indices = {i for _, indices in goal_pairs for i in indices}
        graph.add_node("source")
        graph.add_node("target")
        graph.add_edges_from(("source", i) for i in source_indices)
        graph.add_edges_from((i, "target") for i in target_indices)
        useful = nx.descendants(graph, "source") & nx.ancestors(graph, "target")
        keep = sorted(i for i in useful if isinstance(i, int))
        statistic = {
            "branch": partition_set["label"], "iris_regions": len(original),
            "split_regions": len(regions), "split_edges": len(edges),
            "gcs_regions": len(keep), "split_seconds": split_seconds,
            "adjacency_seconds": edge_seconds,
        }
        statistics.append(statistic)
        print(f"IRIS {statistic}", flush=True)
        if not keep:
            failures.append(f"{partition_set['label']}: no connected start-to-goal IRIS coverage")
            continue
        mapping = {old: new for new, old in enumerate(keep)}
        edges = [(mapping[i], mapping[j]) for i, j in edges if i in mapping and j in mapping]
        regions = [regions[i] for i in keep]
        weights = [reliability_cost_weight(r.reliability, reliability_epsilon) for r in regions]
        region_graph = nx.Graph()
        region_graph.add_nodes_from(range(len(regions)))
        region_graph.add_edges_from(edges)
        for start_q, start_indices in start_pairs:
            for goal_q, goal_indices in goal_pairs:
                source = [mapping[i] for i in start_indices if i in mapping]
                target = [mapping[i] for i in goal_indices if i in mapping]
                if not source or not target:
                    continue
                component = nx.node_connected_component(region_graph, source[0])
                if not any(i in component for i in target):
                    continue
                pair_keep = sorted(component)
                pair_mapping = {old: new for new, old in enumerate(pair_keep)}
                pair_regions = [regions[i] for i in pair_keep]
                pair_edges = [(pair_mapping[i], pair_mapping[j]) for i, j in edges
                              if i in pair_mapping and j in pair_mapping]
                source = [pair_mapping[i] for i in source if i in pair_mapping]
                target = [pair_mapping[i] for i in target if i in pair_mapping]
                pair_started = time.perf_counter()
                try:
                    from gcs.linear import LinearGCS
                    print(f"Solving IRIS GCS: {len(pair_regions)} regions, {len(pair_edges)} edges", flush=True)
                    planner = LinearGCS([r.to_drake() for r in pair_regions], edges=pair_edges,
                                        vertex_cost_weights=[weights[i] for i in pair_keep])
                    planner.addSourceTarget(start_q, goal_q, edges=[source, target])
                    solve_calls += 1
                    waypoints, results = planner.SolvePath(rounding=rounding, preprocessing=True)
                    if waypoints is None:
                        failures.append("GCS returned no path")
                        continue
                    waypoints = waypoints.T
                    collision_free, checked = validate_joint_path(waypoints, obstacles, link_radius)
                    endpoint_error = max(
                        np.linalg.norm(link_points(waypoints[0])[-1] - start_point),
                        np.linalg.norm(link_points(waypoints[-1])[-1] - goal_point),
                    )
                    if not collision_free or endpoint_error > 1e-4:
                        failures.append(f"Independent path check failed: collision_free={collision_free}, "
                                        f"endpoint_error={endpoint_error}")
                        continue
                    cost = results["rounded_cost" if rounding else "mip_cost"]
                    print(f"IRIS GCS candidate: cost={cost:.6f}, "
                          f"seconds={time.perf_counter() - pair_started:.3f}", flush=True)
                    if best is None or cost < best["cost"]:
                        best = {
                            "regions": pair_regions, "waypoints": waypoints, "cost": float(cost),
                            "start": start_q, "goal": goal_q,
                            "start_candidate_count": len(start_pairs),
                            "goal_candidate_count": len(goal_pairs),
                            "reliability_branch": partition_set["label"],
                            "path_reliabilities": [max((r.reliability for r in pair_regions if r.contains(q)),
                                                       default=0.0) for q in waypoints],
                            "solve_seconds": time.perf_counter() - pair_started,
                            "collision_samples": checked, "endpoint_error": float(endpoint_error),
                        }
                except Exception as exc:
                    failures.append(f"{partition_set['label']}: {exc}")
    if best is None:
        raise RuntimeError(f"IRIS GCS failed: {failures[:8]}. Generate more corridor regions if "
                           "the start/goal coverage is disconnected.")
    best["statistics"] = statistics
    best["success"] = True
    best["status"] = "success"
    best["gcs_solve_calls"] = solve_calls
    best["iris_checkpoint"] = str(checkpoint_path)
    best["iris_options"] = metadata.get("options")
    best["total_seconds"] = time.perf_counter() - started
    return best
