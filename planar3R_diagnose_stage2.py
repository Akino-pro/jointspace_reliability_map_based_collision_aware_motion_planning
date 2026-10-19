"""Reproduce a saved final solve and test its known feasible region route."""

import json
import argparse
import time
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from gcs.linear import LinearGCS
from planar3R_gcs_planner import reliability_cost_weight, reliability_region_sets_from_source
from planar3R_iris_gcs_planner import validate_joint_path
from planar3R_iris_regions import load_iris_checkpoint, polytope_adjacency, split_iris_regions
from planar3R_ssm_envelopes import sample_hull
from planar3R_two_stage_gcs_planner import solve_target_sets
from planar3R_collision import link_points
from pydrake.geometry.optimization import GraphOfConvexSetsOptions


def diagnose(input_path, output_path, mip_only=False, stage2_only=False,
             max_rounded_paths=100, max_rounding_trials=100):
    saved = json.loads(Path(input_path).read_text(encoding="utf-8"))
    offsets = np.asarray(saved["angle_winding_offsets"])
    start = np.asarray(saved["start_configuration"]) - offsets
    goal = np.asarray(saved["estimated_goal"]) - offsets
    original, _ = load_iris_checkpoint(saved["iris_checkpoint"])
    partitions = reliability_region_sets_from_source(
        saved["reliability_source"], saved["start_point"], saved["goal_point"], 0)
    partition = next(p for p in partitions if p["label"] == saved["reliability_branch"])
    regions = split_iris_regions(original, partition["regions"], 0)
    edges = polytope_adjacency(regions)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    keep = sorted(set().union(*(nx.node_connected_component(graph, i)
                               for i, r in enumerate(regions) if r.contains(start))))
    mapping = {old: new for new, old in enumerate(keep)}
    edges = [(mapping[i], mapping[j]) for i, j in edges if i in mapping and j in mapping]
    regions = [regions[i] for i in keep]
    weights = [reliability_cost_weight(r.reliability, saved["reliability_epsilon"])
               for r in regions]
    assert len(regions) == saved["region_count"] and len(edges) == saved["edge_count"]
    goal_regions = [i for i, r in enumerate(regions) if r.contains(goal)]
    if stage2_only:
        reference_path = Path(input_path)
        if "stage1_region_route" in saved:
            previous = {**saved, "region_route": saved["stage1_region_route"]}
        else:
            reference_path = Path("stage2_quality_review.json")
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            if reference["input"] != str(input_path):
                raise ValueError("Known-route reference belongs to a different first-stage result")
            previous = reference["repaired_solve"]
        stage1_settings = saved.get("stage1_solver_settings", saved.get("solver_settings", {}))
        stage1_paths = stage1_settings.get("max_rounded_paths", 10)
        stage1_trials = stage1_settings.get("max_rounding_trials", 100)
        previous_points = np.asarray(previous["waypoints"])
        if (np.linalg.norm(previous_points[0] - (start + offsets)) > 1e-6
                or np.linalg.norm(previous_points[-1] - (goal + offsets)) > 1e-6):
            raise ValueError("Known-route reference has different fixed configurations")
        print(f"Reusing saved stage one ({stage1_paths} paths, {stage1_trials} trials); stage two "
              f"uses {max_rounded_paths} paths, {max_rounding_trials} trials", flush=True)
        started = time.perf_counter()
        solved = solve_target_sets(
            regions, edges, start, [{"polytope": sample_hull([goal]),
                                     "region": goal_regions[0], "regions": goal_regions}],
            weights, candidate_region_routes=[previous["region_route"]],
            max_rounded_paths=max_rounded_paths, max_rounding_trials=max_rounding_trials)
        report = {"input": str(input_path), "stage1_reused": True,
                  "stage1_max_paths": stage1_paths, "stage1_max_trials": stage1_trials,
                  "stage2_max_paths": max_rounded_paths,
                  "stage2_max_trials": max_rounding_trials,
                  "rounding_seed": 0, "stage1_cost": saved["stage1_cost"],
                  "reference": str(reference_path),
                  "previous_stage2_cost": previous["cost"],
                  "solve_seconds": time.perf_counter() - started,
                  "success": solved is not None}
        if solved is not None:
            path = solved["waypoints"] + offsets
            free, checked = validate_joint_path(path, saved["obstacles"], saved["link_radius"])
            endpoint_error = float(np.linalg.norm(link_points(path[-1])[-1] - saved["goal_point"]))
            start_error = float(np.linalg.norm(path[0] - (start + offsets)))
            fixed_goal_error = float(np.linalg.norm(path[-1] - (goal + offsets)))
            report.update(**{k: v for k, v in solved.items() if k != "waypoints"},
                          waypoints=path.tolist(), collision_free=free, collision_samples=checked,
                          endpoint_error=endpoint_error, start_error=start_error,
                          fixed_goal_error=fixed_goal_error,
                          success=bool(free and max(endpoint_error, start_error, fixed_goal_error) <= 1e-6))
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return
    if mip_only:
        print("Solving fixed-endpoint MIP with a 60s search limit and 1% stopping gap", flush=True)
        started = time.perf_counter()
        solved = solve_target_sets(
            regions, edges, start, [{"polytope": sample_hull([goal]),
                                     "region": goal_regions[0], "regions": goal_regions}],
            weights, rounding=False, mip_time_limit=60.0, mip_relative_gap=0.01)
        report = {"input": str(input_path), "mode": "mixed_integer_fixed_endpoint",
                  "mip_time_limit": 60.0, "mip_relative_gap": 0.01,
                  "solve_seconds": time.perf_counter() - started, "success": solved is not None}
        if solved is not None:
            path = solved["waypoints"] + offsets
            free, checked = validate_joint_path(path, saved["obstacles"], saved["link_radius"])
            report.update(**{k: v for k, v in solved.items() if k != "waypoints"},
                          waypoints=path.tolist(), collision_free=free,
                          collision_samples=checked,
                          fixed_goal_error=float(np.linalg.norm(path[-1] - (goal + offsets))))
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return
    captured = {}

    class CapturingGCS(LinearGCS):
        def SolvePath(self, *args, **kwargs):
            answer = super().SolvePath(*args, **kwargs)
            captured.update(planner=self, results=answer[1])
            return answer

    print("Reproducing native final-stage rounding", flush=True)
    with patch("gcs.linear.LinearGCS", CapturingGCS):
        native = solve_target_sets(regions, edges, start,
                                  [{"polytope": sample_hull([goal]),
                                    "region": goal_regions[0], "regions": goal_regions}],
                                  weights)
    planner = captured["planner"]
    native_route = [e.u().name() for e in captured["results"]["best_path"]]
    # Recover a valid route from saved stage-one waypoints by a layered shortest
    # path over all regions containing each entire segment. No GCS solve needed.
    points = np.asarray(saved["projection_connection_baseline"]["waypoints"])[:-1] - offsets
    edge_set = set(edges)
    states = {}
    for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
        current = {}
        for i, r in enumerate(regions):
            if not (r.contains(a) and r.contains(b)):
                continue
            segment_cost = weights[i] * np.linalg.norm(b - a)
            choices = [(cost + segment_cost, route + [i])
                       for j, (cost, route) in states.items() if j == i or (j, i) in edge_set]
            if k == 0:
                choices = [(segment_cost, [i])]
            if choices:
                current[i] = min(choices, key=lambda x: x[0])
        states = current
    _, route = states[saved["selected_region"]]
    simple_route = []
    for i in route:
        if i in simple_route:
            simple_route = simple_route[:simple_route.index(i) + 1]
        else:
            simple_route.append(i)
    names = [planner.source.name()] + [planner.names[i] for i in simple_route] + [planner.target.name()]
    edge_lookup = {(e.u().name(), e.v().name()): e for e in planner.gcs.Edges()}
    active = [edge_lookup[(a, b)] for a, b in zip(names[:-1], names[1:])]
    print(f"Solving convex restriction along known route {simple_route}", flush=True)
    restricted = planner.gcs.SolveConvexRestriction(active, GraphOfConvexSetsOptions())
    report = {"input": str(input_path), "native_cost": native["cost"],
              "native_solver_cost": native["solver_cost"], "native_route": native_route,
              "known_route": names, "restriction_success": restricted.is_success()}
    if restricted.is_success():
        path = np.asarray([restricted.GetSolution(active[0].xu())] +
                          [restricted.GetSolution(e.xv()) for e in active]) + offsets
        cost = sum(weights[planner.names.index(e.u().name())] * np.linalg.norm(
            restricted.GetSolution(e.xv()) - restricted.GetSolution(e.xu()))
            for e in active if e.u().name() in planner.names)
        free, checked = validate_joint_path(path, saved["obstacles"], saved["link_radius"])
        report.update(restriction_cost=float(cost),
                      restriction_solver_cost=restricted.get_optimal_cost(),
                      restriction_waypoints=path.tolist(), collision_free=free,
                      collision_samples=checked,
                      fixed_goal_error=float(np.linalg.norm(path[-1] - (goal + offsets))))
    # The ordinary solve now retains this bound; no duplicate relaxation solve.
    report.update(relaxation_cost=captured["results"].get("relaxation_cost"),
                  native_quality=native["quality"])
    print("Verifying repaired selection with the same fixed endpoints", flush=True)
    repaired = solve_target_sets(
        regions, edges, start, [{"polytope": sample_hull([goal]),
                                 "region": goal_regions[0], "regions": goal_regions}],
        weights, candidate_region_routes=[simple_route])
    repaired_path = repaired["waypoints"] + offsets
    repaired_free, repaired_checked = validate_joint_path(
        repaired_path, saved["obstacles"], saved["link_radius"])
    report["repaired_solve"] = {**repaired, "waypoints": repaired_path.tolist(),
                                "collision_free": repaired_free,
                                "collision_samples": repaired_checked}
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mip-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--input", default="two_stage_path_per_intersection.json")
    parser.add_argument("--max-rounded-paths", type=int, default=100)
    parser.add_argument("--max-rounding-trials", type=int, default=100)
    parser.add_argument("--output", default="stage2_quality_review.json")
    args = parser.parse_args()
    if args.mip_only and args.stage2_only:
        parser.error("Choose either --mip-only or --stage2-only")
    diagnose(args.input, args.output, args.mip_only,
             args.stage2_only, args.max_rounded_paths, args.max_rounding_trials)
