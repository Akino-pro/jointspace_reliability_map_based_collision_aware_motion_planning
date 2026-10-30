"""Intersection-vertex representative graph and trust-region SLP (paper Eq.29).

Our regional density is constant: w_r ||q_u-q_v||, a convex special case of
the paper's nonconvex edge cost. SLP linearizes a smoothed norm, uses exact
intersection halfspaces and accepts only decreases of the collective objective.
Final path selection is translated back to the existing region-vertex GCS;
this reproduces the representative/preprocessing idea, not the full GCSGC.
"""
import itertools
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag

from scripts.planar3R_iris_regions import interior_point
from scripts.goal_set_baselines import graph_task_fingerprint


def feasible_representative(A, b):
    point = interior_point(A, b, min_radius=-1e-10)
    if point is None:
        result = linprog(np.zeros(A.shape[1]), A_ub=A, b_ub=b,
                         bounds=[(None, None)]*A.shape[1], method="highs")
        if not result.success:
            raise ValueError("Intersection has no feasible representative")
        point = result.x
    return point


def collective_cost(points, links, weights, smoothing=0.):
    delta = points[links[:, 1]]-points[links[:, 0]]
    lengths = np.sqrt(np.sum(delta*delta, axis=1)+smoothing*smoothing)
    return float(weights[links[:, 2]] @ lengths)


def optimize_representatives(sets, links, weights, max_iterations=80,
                             initial_radius=.15, smoothing=1e-6, tolerance=1e-6):
    began = time.perf_counter()
    if max_iterations < 0 or initial_radius <= 0 or smoothing <= 0:
        raise ValueError("Expected nonnegative iteration cap and positive trust radius and smoothing")
    points = np.asarray([feasible_representative(A, b) for A, b in sets])
    initial = points.copy()
    if not len(links):
        return points, dict(initial_cost=0., final_cost=0., iterations=0,
                            status="no_edges", seconds=time.perf_counter()-began)
    matrix = block_diag([A for A, _ in sets], format="csr")
    bounds_rhs = np.concatenate([b for _, b in sets])
    edge_weights = weights[links[:, 2]]
    radius = initial_radius
    objective = collective_cost(points, links, weights, smoothing)
    history, accepted = [], 0
    status = "disabled" if max_iterations == 0 else "iteration_limit"
    for iteration in range(max_iterations):
        delta = points[links[:, 1]]-points[links[:, 0]]
        length = np.sqrt(np.sum(delta*delta, axis=1)+smoothing*smoothing)
        directions = edge_weights[:, None]*delta/length[:, None]
        gradient = np.zeros_like(points)
        np.add.at(gradient, links[:, 0], -directions)
        np.add.at(gradient, links[:, 1], directions)
        candidate = linprog(gradient.ravel(), A_ub=matrix, b_ub=bounds_rhs,
                            bounds=np.column_stack((points.ravel()-radius, points.ravel()+radius)),
                            method="highs")
        if not candidate.success:
            raise RuntimeError(f"SLP subproblem failed: {candidate.message}")
        proposal = candidate.x.reshape(points.shape)
        predicted = float(np.sum(gradient*(points-proposal)))
        if predicted <= tolerance*max(1., objective):
            status = "first_order_tolerance"
            break
        proposed_cost = collective_cost(proposal, links, weights, smoothing)
        decrease = objective-proposed_cost
        ratio = decrease/predicted
        take = decrease > 0 and ratio >= .1
        history.append(dict(iteration=iteration+1, objective=objective, candidate=proposed_cost,
                            trust_radius=radius, accepted=bool(take), ratio=ratio))
        if take:
            points, objective = proposal, proposed_cost
            accepted += 1
            if ratio > .75:
                radius = min(1., radius*1.5)
        else:
            radius *= .5
        if (iteration+1) % 10 == 0:
            print(f"Intersection SLP {iteration+1}: objective={objective:.8f}, radius={radius:.6g}", flush=True)
        if radius < 1e-7:
            status = "trust_region_minimum"
            break
    violation = max(float(np.max(A@point-b)) for (A, b), point in zip(sets, points))
    if violation > 2e-7:
        raise RuntimeError(f"SLP representative violates intersection halfspaces by {violation}")
    return points, dict(initial_cost=collective_cost(initial, links, weights),
                        final_cost=collective_cost(points, links, weights),
                        initial_smoothed_cost=collective_cost(initial, links, weights, smoothing),
                        final_smoothed_cost=objective, smoothing=smoothing, iterations=len(history),
                        accepted_iterations=accepted, status=status, history=history,
                        max_constraint_violation=max(0., violation), max_iterations=max_iterations,
                        seconds=time.perf_counter()-began)


def build_representative(regions, edges, weights, cache=None, max_iterations=80):
    began = time.perf_counter()
    dimension = regions[0].A.shape[1]
    if any(np.any(np.asarray(getattr(e, "shift", [])) != 0) for e in edges):
        raise ValueError("Intersection representatives currently support nonperiodic graphs")
    pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in edges if i != j})
    degree = np.bincount(np.asarray(pairs, dtype=int).ravel(), minlength=len(regions))
    projected_links = int(np.sum(degree*(degree-1)//2))
    if len(pairs) > 50000 or projected_links > 2000000:
        raise ValueError(f"Intersection graph requires {len(pairs)} vertices and {projected_links} "
                         "undirected edges; exceeds the bounded-memory implementation's capacity")
    key = graph_task_fingerprint(regions, edges, np.zeros(dimension), weights,
                                 dict(type="intersection_trust_slp_v1", max_iterations=max_iterations))
    cache_path = Path(cache) if cache is not None else None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["key"]) != key:
                raise ValueError("SLP cache does not match geometry, weights or optimizer settings")
            return saved["pairs"].copy(), saved["links"].copy(), saved["points"].copy(), dict(
                json.loads(str(saved["statistics"])), cache_hit=True, load_seconds=time.perf_counter()-began)
    incident = [[] for _ in regions]
    sets = []
    for index, (i, j) in enumerate(pairs):
        sets.append((np.vstack((regions[i].A, regions[j].A)), np.r_[regions[i].b, regions[j].b]))
        incident[i].append(index)
        incident[j].append(index)
    links = np.asarray([(u, v, r) for r, nodes in enumerate(incident)
                        for u, v in itertools.combinations(nodes, 2)], dtype=int).reshape(-1, 3)
    print(f"Intersection representative: {len(pairs)} vertices, {len(links)} undirected within-region edges", flush=True)
    if len(pairs):
        points, optimization = optimize_representatives(sets, links, np.asarray(weights), max_iterations)
    else:
        points = np.empty((0, dimension))
        optimization = dict(initial_cost=0., final_cost=0., status="no_intersections", seconds=0.)
    statistics = dict(intersection_vertices=len(pairs), representative_undirected_edges=len(links),
                      optimization=optimization, cache_hit=False, seconds=time.perf_counter()-began)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, key=np.asarray(key), pairs=pairs, links=links, points=points,
                            statistics=np.asarray(json.dumps(statistics)))
    return pairs, links, points, statistics


def erase_region_loops(route):
    simple = []
    for region in route:
        if simple and simple[-1] == region:
            continue
        if region in simple:
            simple = simple[:simple.index(region)+1]
        else:
            simple.append(region)
    return simple


def prune_intersection_slp(regions, edges, start, targets, weights, alpha=1.1,
                           trials=2000, seed=0, cache=None, max_iterations=80):
    began = time.perf_counter()
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(alpha) or alpha < 1 or trials < 1 or np.any(weights <= 0):
        raise ValueError("Expected alpha >= 1, positive trials and weights")
    pairs, links, points, representative = build_representative(regions, edges, weights, cache, max_iterations)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(pairs)))
    for u, v, region in links:
        graph.add_edge(int(u), int(v), weight=float(weights[region]*np.linalg.norm(points[v]-points[u])), region=int(region))
    source, sink = -1, len(pairs)
    graph.add_node(source)
    sources = {i for i, r in enumerate(regions) if r.contains(start)}

    def connect(u, v, region, cost):
        if not graph.has_edge(u, v) or cost < graph[u][v]["weight"]:
            graph.add_edge(u, v, region=int(region), weight=float(cost))

    for node, pair in enumerate(pairs):
        for region in sources.intersection(map(int, pair)):
            connect(source, node, region, weights[region]*np.linalg.norm(start-points[node]))
    rng = np.random.default_rng(seed)
    selected_nodes, selected_edges, target_stats = set(), set(), []
    def retain(path):
        route = erase_region_loops([graph[u][v]["region"] for u, v in zip(path[:-1], path[1:])])
        selected_nodes.update(route)
        selected_edges.update(zip(route[:-1], route[1:]))

    for index, target in enumerate(targets):
        goal = feasible_representative(target["polytope"].A, target["polytope"].b)
        goal_regions = set(target.get("regions", [target["region"]]))
        graph.add_node(sink)
        for node, pair in enumerate(pairs):
            for region in goal_regions.intersection(map(int, pair)):
                connect(node, sink, region, weights[region]*np.linalg.norm(points[node]-goal))
        for region in sources & goal_regions:
            connect(source, sink, region, weights[region]*np.linalg.norm(start-goal))
        distances, paths = nx.single_source_dijkstra(graph, sink)
        if source not in distances:
            target_stats.append(dict(target=index, reachable=False, sampled_paths=0))
            graph.remove_node(sink)
            continue
        retain(paths[source][::-1])
        successes = 0
        for _ in range(trials):
            route, visited, spent = [source], {source}, 0.
            while route[-1] != sink:
                u = route[-1]
                options = [(v, data["weight"]) for v, data in graph[u].items()
                           if v not in visited and v in distances
                           and spent+data["weight"]+distances[v] <= alpha*distances[source]+1e-10]
                if not options:
                    break
                score = np.asarray([cost+distances[v] for v, cost in options])
                probability = np.exp(-(score-score.min()))
                v, cost = options[rng.choice(len(options), p=probability/probability.sum())]
                route.append(v)
                visited.add(v)
                spent += cost
            if route[-1] == sink:
                retain(route)
                successes += 1
        target_stats.append(dict(target=index, reachable=True, minimum_surrogate_cost=distances[source], sampled_paths=successes))
        print(f"Intersection pruning target {index+1}/{len(targets)}: {successes}/{trials} successful samples", flush=True)
        graph.remove_node(sink)
    for target in targets:
        selected_nodes.update(target.get("regions", [target["region"]]))
        selected_nodes.add(target["region"])
    keep = sorted(selected_nodes)
    mapping = {old: new for new, old in enumerate(keep)}
    reduced_edges = [(mapping[i], mapping[j]) for i, j in edges if (i, j) in selected_edges]
    original_edges = set(edges)
    if any(pair not in original_edges for pair in selected_edges):
        raise RuntimeError("Translated intersection route is not in the original directed region graph")
    reduced_targets = [dict(t, region=mapping[t["region"]],
                            regions=[mapping[i] for i in t.get("regions", [t["region"]])]) for t in targets]
    statistics = dict(method="intersection_vertices_trust_region_slp", representative=representative,
                      alpha=alpha, trials_per_target=trials, seed=seed, targets=target_stats,
                      original_regions=len(regions), original_edges=len(edges), retained_regions=len(keep),
                      retained_edges=len(reduced_edges), retained_original_region_indices=keep,
                      preserves_full_graph_optimality=False, final_gcs_formulation="existing_region_vertices",
                      seconds=time.perf_counter()-began)
    return [regions[i] for i in keep], reduced_edges, reduced_targets, statistics
