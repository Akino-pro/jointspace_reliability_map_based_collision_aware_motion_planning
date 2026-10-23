"""Direct sampled-SSM terminals on the same weighted graph as two-stage GCS."""
import hashlib
import json
import time
import numpy as np


def compare_sampled_goals(regions, edges, start, envelopes, weights, settings,
                          collision_free, validate_path, task_error, rounding=True):
    """Keep every distinct covered safe sample; never prune by heuristic cost.

    Each point has one terminal attached to ALL containing regions. The dummy
    sink only selects a terminal and never represents physical motion.
    """
    from scripts.planar3R_ssm_envelopes import sample_hull
    from scripts.planar3R_two_stage_gcs_planner import solve_target_sets
    began = time.perf_counter()
    samples = sorted({tuple(q) for envelope in envelopes for q in envelope.samples})
    targets, covered_samples, terminal_edges = [], [], 0
    for values in samples:
        q = np.asarray(values)
        if not collision_free(q):
            continue
        indices = [i for i, region in enumerate(regions) if region.contains(q)]
        if indices:
            targets.append(dict(polytope=sample_hull([q]), region=indices[0], regions=indices))
            covered_samples.append(q.tolist())
            terminal_edges += len(indices)
    contract = dict(start=np.asarray(start).tolist(), samples=samples,
                    regions=[dict(A=r.A.tolist(), b=r.b.tolist(), weight=float(w))
                             for r, w in zip(regions, weights)],
                    edges=[dict(pair=list(e), shift=np.asarray(getattr(e, "shift", [])).tolist()) for e in edges],
                    solver_settings=settings, rounding=rounding)
    report = dict(method="one_graph_all_sampled_endpoints", target_count=len(targets),
                  unique_sample_count=len(samples), terminal_edge_count=terminal_edges,
                  input_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                  solver_settings=settings, gcs_solve_calls=0, success=False,
                  preparation_seconds=time.perf_counter()-began,
                  comparison_scope="same_samples_scene_regions_weights_edges_and_stage1_solver_settings",
                  global_optimality_certified=False)
    solve_began = time.perf_counter()
    result = None
    try:
        if targets:
            report["gcs_solve_calls"] = 1
            result = solve_target_sets(regions, edges, start, targets, weights,
                                       rounding=rounding, **settings)
        report["solve_seconds"] = time.perf_counter()-solve_began
        if result is None:
            report.update(status="solver_failed" if targets else "no_covered_sampled_goals",
                          cost=None, quality=None)
        else:
            path = result["waypoints"]
            free, checked = validate_path(path)
            error = float(task_error(path[-1]))
            start_error = float(np.linalg.norm(path[0]-start))
            valid = bool(free and max(error, start_error) <= 1e-6)
            report.update(success=valid, status="success" if valid else "path_validation_failed",
                          cost=result["cost"], quality=result["quality"],
                          selected_sample=covered_samples[result["target"]],
                          endpoint=path[-1].tolist(), waypoints=path.tolist(),
                          region_route=result["region_route"], route_spec=result["route_spec"],
                          collision_free=free, collision_samples=checked,
                          task_error=error, start_error=start_error,
                          rounded_path_count=result["rounded_path_count"], solver_name=result["solver_name"])
    except RuntimeError as exc:
        report.update(status="solver_failed", error=str(exc), cost=None, quality=None)
    report.update(seconds=time.perf_counter()-began)
    return report
