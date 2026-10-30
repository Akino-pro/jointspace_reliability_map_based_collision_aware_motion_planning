"""Heuristic path pruning before allocating any Drake GCS objects.

Uses feasible Chebyshev centers, not the paper's SLP-optimized intersection
representatives. Fixed center edge weights are estimates, not admissible bounds.
Every reachable target gets a shortest representative path before sampling.
"""
import time

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from scripts.planar3R_iris_regions import interior_point


def prune_representative_paths(regions, edges, start, targets, weights,
                               alpha=1.1, trials=100, seed=0):
    began = time.perf_counter()
    if not np.isfinite(alpha) or alpha < 1 or trials < 1:
        raise ValueError("Pruning requires finite alpha >= 1 and positive trials")
    if any(np.any(np.asarray(getattr(e, "shift", [])) != 0) for e in edges):
        raise ValueError("Representative pruning currently supports nonperiodic graphs only")
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(regions) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Pruning requires positive finite region weights")
    centers = []
    for region in regions:
        point = interior_point(region.A, region.b)
        if point is None:
            raise ValueError("Could not find a representative inside a region")
        centers.append(point)
    centers = np.asarray(centers)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(regions)))
    for i, j in edges:
        # Surrogate only: the GCS still charges exact segment cost in its region.
        graph.add_edge(i, j, weight=float(weights[i]*np.linalg.norm(centers[j]-centers[i])))
    source, sink = len(regions), len(regions)+1
    graph.add_node(source)
    for i, region in enumerate(regions):
        if region.contains(start):
            graph.add_edge(source, i, weight=float(weights[i]*np.linalg.norm(centers[i]-start)))
    rng = np.random.default_rng(seed)
    selected_edges, selected_nodes, target_stats = set(), set(), []

    def retain(path):
        selected_nodes.update(v for v in path if v < source)
        selected_edges.update((u, v) for u, v in zip(path[:-1], path[1:])
                              if u < source and v < source)

    for index, target in enumerate(targets):
        goal = interior_point(target["polytope"].A, target["polytope"].b, min_radius=-1e-10)
        representative_method = "chebyshev_center"
        if goal is None:
            # Thin/affine target hulls can have zero inscribed radius. Use the
            # same unmodified feasibility LP used to create their intersections.
            polytope = target["polytope"]
            feasible = linprog(np.zeros(polytope.A.shape[1]), A_ub=polytope.A, b_ub=polytope.b,
                               bounds=[(None, None)]*polytope.A.shape[1], method="highs")
            if not feasible.success:
                raise ValueError(f"Could not find a representative in target set {index}")
            goal = feasible.x
            representative_method = "feasibility_lp"
        graph.add_node(sink)
        for i in target.get("regions", [target["region"]]):
            graph.add_edge(i, sink, weight=float(weights[i]*np.linalg.norm(goal-centers[i])))
        distance, paths = nx.single_source_dijkstra(graph.reverse(copy=False), sink)
        if source not in distance:
            target_stats.append(dict(target=index, reachable=False, sampled_paths=0))
            graph.remove_node(sink)
            continue
        retain(list(reversed(paths[source])))
        minimum = distance[source]
        successes = 0
        for _ in range(trials):
            path, visited, spent = [source], {source}, 0.
            while path[-1] != sink:
                u = path[-1]
                options = [(v, data["weight"]) for v, data in graph[u].items()
                           if v not in visited and v in distance
                           and spent+data["weight"]+distance[v] <= alpha*minimum+1e-10]
                if not options:
                    break
                score = np.array([cost+distance[v] for v, cost in options])
                probability = np.exp(-(score-score.min()))
                v, cost = options[rng.choice(len(options), p=probability/probability.sum())]
                path.append(v)
                visited.add(v)
                spent += cost
            if path[-1] == sink:
                retain(path)
                successes += 1
        target_stats.append(dict(target=index, reachable=True, minimum_surrogate_cost=minimum,
                                 sampled_paths=successes, representative_method=representative_method))
        graph.remove_node(sink)
    # Keep terminal region identities, including unreachable targets, so target
    # indexing and the existing per-target failure reports remain unchanged.
    for target in targets:
        selected_nodes.update(target.get("regions", [target["region"]]))
        selected_nodes.add(target["region"])
    keep = sorted(selected_nodes)
    mapping = {old: new for new, old in enumerate(keep)}
    reduced_targets = [dict(t, region=mapping[t["region"]],
                            regions=[mapping[i] for i in t.get("regions", [t["region"]])]) for t in targets]
    reduced_edges = [(mapping[i], mapping[j]) for i, j in edges if (i, j) in selected_edges]
    stats = dict(method="feasible_center_surrogate_path_union", alpha=alpha, trials_per_target=trials,
                 seed=seed, original_regions=len(regions), original_edges=len(edges),
                 retained_regions=len(keep), retained_edges=len(reduced_edges),
                 retained_original_region_indices=keep, targets=target_stats,
                 preserves_full_graph_optimality=False, seconds=time.perf_counter()-began)
    return [regions[i] for i in keep], reduced_edges, reduced_targets, stats
