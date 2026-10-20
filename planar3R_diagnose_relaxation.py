"""Diagnose the reverse benchmark's first relaxation with bounded worker time."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def run_worker(args):
    from pydrake.geometry.optimization import Point
    from pydrake.solvers import ClarabelSolver, CommonSolverOption, L2NormCost, MosekSolver, SolverOptions
    from gcs.linear import BindingCost, LinearGCS
    import planar3R_two_stage_gcs_planner as workflow

    saved = json.loads(Path(args.input).read_text(encoding="utf-8"))
    captured = {}

    class InputsCaptured(BaseException):
        pass

    def capture(regions, edges, start, targets, weights, *unused, **kwargs):
        captured.update(regions=regions, edges=edges, start=start, target=targets[0], weights=weights)
        raise InputsCaptured()

    started = time.perf_counter()
    try:
        with patch.object(workflow, "solve_target_sets", side_effect=capture):
            workflow.plan_two_stage(
                saved["start_configuration"], saved["goal_point"], saved["obstacles"],
                saved["iris_checkpoint"], geometric_epsilon=saved["geometric_epsilon"],
                ssm_arc_spacing=saved["ssm_arc_spacing"],
                reliability_epsilon=saved["reliability_epsilon"])
    except InputsCaptured:
        pass
    regions, edges = captured["regions"], captured["edges"]
    planner = LinearGCS([r.to_drake() for r in regions], edges=edges,
                        vertex_cost_weights=captured["weights"])
    vertices = list(planner.gcs.Vertices())
    source = planner.gcs.AddVertex(Point(captured["start"]), "fixed_start")
    target_data = captured["target"]
    target = planner.gcs.AddVertex(target_data["polytope"].to_drake(), "local_target_0")
    for index, region in enumerate(regions):
        if region.contains(captured["start"]):
            edge = planner.gcs.AddEdge(source, vertices[index])
            for axis in range(3):
                edge.AddConstraint(edge.xu()[axis] == edge.xv()[axis])
    index = target_data["region"]
    edge = planner.gcs.AddEdge(vertices[index], target)
    matrix = captured["weights"][index] * np.hstack((-np.eye(3), np.eye(3)))
    edge.AddCost(BindingCost(L2NormCost(matrix, np.zeros(3)), np.append(edge.xu(), edge.xv())))
    solvers = {"clarabel": ClarabelSolver(), "mosek": MosekSolver()}
    options = planner.options
    options.convex_relaxation = True
    options.max_rounded_paths = 0
    options.preprocessing = args.preprocessing_solver != "none"
    options.solver = solvers[args.solver]
    options.solver_options = SolverOptions()
    options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    options.solver_options.SetOption(ClarabelSolver.id(), "time_limit", args.time_limit)
    options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_OPTIMIZER_MAX_TIME", args.time_limit)
    if args.threads is not None:
        options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_NUM_THREADS", args.threads)
    if options.preprocessing:
        options.preprocessing_solver = solvers[args.preprocessing_solver]
        prep = SolverOptions()
        prep.SetOption(ClarabelSolver.id(), "time_limit", args.time_limit)
        prep.SetOption(MosekSolver.id(), "MSK_DPAR_OPTIMIZER_MAX_TIME", args.time_limit)
        options.preprocessing_solver_options = prep
    report = dict(solver=args.solver, preprocessing_solver=args.preprocessing_solver,
                  mosek_threads=args.threads,
                  solver_time_limit=args.time_limit, graph_regions=len(regions),
                  graph_region_edges=len(edges), graph_total_edges=len(planner.gcs.Edges()),
                  setup_seconds=time.perf_counter() - started, status="before_solve")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("BEGIN SolveShortestPath", json.dumps(report), flush=True)
    began = time.perf_counter()
    result = planner.gcs.SolveShortestPath(source, target, options)
    report.update(wall_seconds=time.perf_counter() - began, success=result.is_success(),
                  status=str(result.get_solution_result()), solver_id=result.get_solver_id().name(),
                  cost=result.get_optimal_cost() if np.isfinite(result.get_optimal_cost()) else None)
    details = result.get_solver_details()
    report["solver_details"] = {key: getattr(details, key) for key in
                                ("iterations", "solve_time", "optimizer_time", "status",
                                 "solution_status", "rescode") if hasattr(details, key)}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("END SolveShortestPath", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="two_stage_reverse_multibranch_merged_mosek.json")
    parser.add_argument("--solver", choices=["clarabel", "mosek"], required=True)
    parser.add_argument("--preprocessing-solver", choices=["none", "clarabel", "mosek"], default="none")
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--watchdog-seconds", type=float, default=100)
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional MOSEK thread count for a controlled comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    log_path = Path(args.output).with_suffix(".log")
    began = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run([sys.executable, "-u", __file__, *sys.argv[1:], "--worker"],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.watchdog_seconds)
            code = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out, code = True, None
    report = json.loads(Path(args.output).read_text(encoding="utf-8")) if Path(args.output).exists() else {}
    report.update(watchdog_timed_out=timed_out, worker_exit_code=code,
                  total_seconds=time.perf_counter() - began, log=str(log_path))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
