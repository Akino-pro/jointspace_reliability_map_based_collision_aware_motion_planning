"""Inspect the unchanged 4R union-hull relaxation with controlled solver settings."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch


class DiagnosticComplete(BaseException):
    pass


def main():
    from gcs.linear import LinearGCS
    from scripts.spatial4R_two_stage_gcs_planner import plan_before_failure
    from pydrake.solvers import CommonSolverOption, MosekSolver, ClarabelSolver, SolverOptions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["default", "no-preprocessing", "primal", "dual", "single-thread", "clarabel"], default="default")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def diagnose(planner, **unused):
        options = planner.options
        options.convex_relaxation = True
        options.max_rounded_paths = 0
        options.preprocessing = args.mode != "no-preprocessing"
        options.solver_options = SolverOptions()
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        options.preprocessing_solver_options = SolverOptions()
        options.preprocessing_solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)
        options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_OPTIMIZER_MAX_TIME", 90.)
        if args.mode in ("primal", "dual"):
            options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1 if args.mode == "primal" else 2)
        if args.mode == "single-thread":
            options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_NUM_THREADS", 1)
        if args.mode == "clarabel":
            options.solver = ClarabelSolver()
            options.solver_options.SetOption(ClarabelSolver.id(), "max_iter", 300)
        began = time.perf_counter()
        result = planner.gcs.SolveShortestPath(planner.source, planner.target, options)
        details = result.get_solver_details()
        report = dict(mode=args.mode, success=result.is_success(),
                      termination=str(result.get_solution_result()), seconds=time.perf_counter()-began,
                      vertices=len(planner.gcs.Vertices()), edges=len(planner.gcs.Edges()),
                      solver=result.get_solver_id().name(),
                      details={key: float(getattr(details, key)) for key in
                               ("rescode", "solution_status", "optimizer_time", "iterations", "solve_time")
                               if hasattr(details, key)})
        if result.is_success():
            report["cost"] = result.get_optimal_cost()
        output.write_text(json.dumps(report, indent=2))
        print("RESULT "+json.dumps(report), flush=True)
        raise DiagnosticComplete()

    try:
        with patch.object(LinearGCS, "SolvePath", diagnose):
            plan_before_failure(max_paths=800, max_trials=1200, union_hulls_only=True,
                                stage2_max_paths=800, stage2_max_trials=1200)
    except DiagnosticComplete:
        pass


if __name__ == "__main__":
    main()
