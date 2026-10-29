"""Reuse saved stage-one endpoints and replay only fixed-goal GCS budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from scripts.goal_set_baselines import graph_task_fingerprint, validate_result
from scripts.planar3R_iris_regions import load_iris_checkpoint, split_iris_regions, polytope_adjacency
from scripts.planar3R_ssm_envelopes import sample_hull
from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
from scripts.periodic_graph import remap_edges


def chart_configuration(supplied):
    supplied = np.asarray(supplied, dtype=float)
    outside = np.abs(supplied) > np.pi
    chart = np.where(outside, (supplied+np.pi) % (2*np.pi)-np.pi, supplied)
    boundary = outside & (supplied > 0) & np.isclose(chart, -np.pi, atol=1e-12, rtol=0)
    chart[boundary] = np.pi
    return chart, supplied-chart


def connected(regions, edges, start):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(regions)))
    graph.add_edges_from(edges)
    sources = [i for i, r in enumerate(regions) if r.contains(start)]
    keep = sorted(set().union(*(nx.node_connected_component(graph, i) for i in sources))) if sources else []
    if not keep:
        raise RuntimeError("Saved start is not covered by reconstructed regions")
    mapping = {old: new for new, old in enumerate(keep)}
    return [regions[i] for i in keep], remap_edges(edges, mapping)


def prepare_graph(source, spatial):
    start, offsets = chart_configuration(source["start_configuration"])
    goal = np.asarray(source["goal_point"])
    originals, metadata = load_iris_checkpoint(source["iris_checkpoint"], expected_dimension=len(start))
    scene = metadata["model"]
    obstacles, radius = scene["obstacles"], float(scene["link_radius"])
    epsilon = source.get("reliability_epsilon", .01)
    if spatial:
        from scripts import spatial4R_model as model
        from scripts.spatial4R_two_stage_gcs_planner import reliability_regions_for_branches
        model.select_robot(source.get("robot_parameter_set", "ftwmax"))
        if any(scene["dh"][key] != model.PARAMETERS[key] for key in ("a", "d", "alpha")):
            raise RuntimeError("Checkpoint robot changed")
        branches, _ = model.compute_ssms(goal, reference=model.GOAL_REFERENCE_CONFIGURATION,
                                         attempts=model.MAX_SSM_IK_ATTEMPTS, seed=1)
        _, merged = reliability_regions_for_branches(branches)
        regions = split_iris_regions(originals, merged)
        task = dict(id=dict(type="position_3D", goal=goal.tolist()),
                    residual=lambda q: model.position(q)-goal)
        validate = lambda p: model.validate_path(p, obstacles, radius)
    else:
        from scripts.planar3R_collision import L, link_points
        from scripts.planar3R_gcs_planner import reliability_region_sets_from_source
        from scripts.planar3R_iris_gcs_planner import validate_joint_path
        if scene["link_lengths"] != list(L):
            raise RuntimeError("Checkpoint link lengths changed")
        partition = reliability_region_sets_from_source(source.get("reliability_source", "goal"),
                                                        link_points(start)[-1], goal, seed=0)[0]
        regions = split_iris_regions(originals, partition["regions"])
        task = dict(id=dict(type="position_2D", goal=goal.tolist()),
                    residual=lambda q: link_points(q)[-1]-goal)
        validate = lambda p: validate_joint_path(p, obstacles, radius)
    if source.get("periodic_connections", False):
        raise ValueError("This replay runner supports the nonperiodic sweep reports only")
    edges = polytope_adjacency(regions)
    regions, edges = connected(regions, edges, start)
    weights = [epsilon+1-r.reliability for r in regions]
    fingerprint = graph_task_fingerprint(regions, edges, start, weights, task["id"])
    return dict(regions=regions, edges=edges, weights=weights, start=start,
                offsets=offsets, task=task, validate=validate, fingerprint=fingerprint)


def write_summary(folder, summary):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Matched-budget stage-two replay", "",
             "Only stage two is solved again. Saved projected endpoints and available original stage-one routes are reused.",
             "Every stage-two budget matches that case's stage-one per-call path/trial cap.",
             "Stitched total = previous method total minus previous stage-two solve time plus newly measured stage-two solve time.",
             "It is a timing estimate combining saved and replayed components, not a new end-to-end measurement.", "",
             "| Method | Budget in both stages | Final cost | New stage two (s) | Stitched total (s) | Retained stage-one route | Status |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for r in summary["runs"]:
        cost = "—" if r.get("cost") is None else f"{r['cost']:.10f}"
        lines.append(f"| {r['method']} | {r['max_paths']}/{r['max_trials']} | {cost} | "
                     f"{r['stage2_seconds']:.3f} | {r['stitched_total_seconds']:.3f} | "
                     f"{r['retained_stage1_route']} | {r['status']} |")
    lines += ["", "Source JSON hashes, graph identity checks and all replay validations are saved per case.",
              "No stage-one GCS, sampled-goal or nonlinear-goal experiment ran in this replay.",
              "Any unavailable original stage-one route is explicitly recorded; a previous stage-two route is never used as a substitute.",
              "The separate unfinished 4R first-stage sweep remains paused."]
    (folder / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["independent", "joint"], required=True)
    parser.add_argument("--joint-without-retained-route", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.group == "joint" and not args.joint_without_retained_route:
        parser.error("Old joint reports lack stage-one routes; explicitly choose the endpoint-only replay")
    cases = []
    if args.group == "independent":
        cases += [(p, False, False) for p in sorted(Path("results/planar3r/per_target_budget_sweep_20261029").glob("paths*.json"))]
        cases += [(p, True, False) for p in sorted(Path("results/spatial4r/per_target_budget_sweep_20261029").glob("paths*.json"))]
    else:
        cases += [(p, False, True) for p in sorted(Path("results/planar3r/union_hull_budget_sweep_20261029").glob("paths*.json"))]
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(status="running", group=args.group, stage1_gcs_solve_calls=0, runs=[])
    graphs = {}
    for path, spatial, joint in cases:
        source = json.loads(path.read_text(encoding="utf-8"))
        saved = source["union_hull_baseline"] if joint else source
        if not saved.get("success") or "estimated_goal" not in saved:
            continue
        settings = saved["solver_settings"] if joint else source["stage1_solver_settings"]
        paths, trials = settings["max_rounded_paths"], settings["max_rounding_trials"]
        method = "4R independent" if spatial else ("3R joint" if joint else "3R independent")
        print(f"BEGIN replay {method} {paths}/{trials}: saved goal, zero stage-one solves", flush=True)
        expected = saved["graph_task_sha256"] if joint else source["graph_task_sha256"]
        cache_key = (spatial, expected)
        if cache_key not in graphs:
            graphs[cache_key] = prepare_graph(source, spatial)
        graph = graphs[cache_key]
        if graph["fingerprint"] != expected:
            raise RuntimeError("Reconstructed graph/task differs from the saved first-stage graph")
        goal = np.asarray(saved["estimated_goal"])-graph["offsets"]
        indices = [i for i, r in enumerate(graph["regions"]) if r.contains(goal)]
        if not indices:
            raise RuntimeError("Saved projected endpoint is outside reconstructed coverage")
        targets = [dict(polytope=sample_hull([goal]), region=indices[0], regions=indices)]
        route = None if joint else source.get("stage1_route_spec", source.get("stage1_region_route"))
        if not joint and route is None:
            raise RuntimeError("Expected an original stage-one route in independent-method report")
        diagnostics = {}
        solve_began = time.perf_counter()
        solved = solve_target_sets(graph["regions"], graph["edges"], graph["start"], targets, graph["weights"],
                                   max_rounded_paths=paths, max_rounding_trials=trials, rounding_seed=0,
                                   solver_backend="mosek", candidate_region_routes=[route] if route is not None else None,
                                   diagnostics=diagnostics)
        seconds = time.perf_counter()-solve_began
        if joint:
            old_total = source["geometry_seconds"]+saved["seconds"]
        else:
            old_total = source["two_stage_total_seconds"]
        result = dict(method=method, max_paths=paths, max_trials=trials, source=str(path),
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), graph_task_sha256=expected,
                      source_stage1_cost=saved["stage1_cost"], saved_goal=saved["estimated_goal"],
                      retained_stage1_route=route is not None, missing_stage1_route=joint,
                      previous_cost=saved.get("cost"), previous_stage2_seconds=saved["stage2_seconds"],
                      stage2_seconds=seconds, stitched_total_seconds=old_total-saved["stage2_seconds"]+seconds,
                      rounding_diagnostics=diagnostics, stage1_gcs_solve_calls=0,
                      success=False, status="solver_failed")
        if solved is not None:
            result.update(validate_result(solved, graph["regions"], graph["start"], targets,
                                          graph["weights"], graph["task"], graph["validate"]))
            if "waypoints" in result:
                result["waypoints"] = (np.asarray(result["waypoints"])+graph["offsets"]).tolist()
                result["endpoint"] = (np.asarray(result["endpoint"])+graph["offsets"]).tolist()
        name = f"{'4r' if spatial else '3r'}_{'joint' if joint else 'independent'}_{paths:03d}_{trials:03d}.json"
        (folder / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
        row = {key: value for key, value in result.items() if key not in ("waypoints", "route_spec")}
        row["report"] = name
        summary["runs"].append(row)
        write_summary(folder, summary)
        print(f"END {method} {paths}/{trials}: {result['status']}, cost={result.get('cost')}, stage2={seconds:.3f}s", flush=True)
    summary["status"] = "complete"
    write_summary(folder, summary)


if __name__ == "__main__":
    main()
