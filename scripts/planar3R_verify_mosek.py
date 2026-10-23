"""Validate the configured MOSEK license with a tiny mixed-integer cone solve."""

import json
from pathlib import Path

import numpy as np
from pydrake.solvers import MathematicalProgram, MosekSolver


def main():
    solver = MosekSolver()
    report = {"available": solver.available(), "enabled": solver.enabled()}
    license_handle = solver.AcquireLicense()
    report["license_acquired"] = license_handle is not None
    program = MathematicalProgram()
    binary = program.NewBinaryVariables(1)[0]
    distance = program.NewContinuousVariables(1)[0]
    program.AddLorentzConeConstraint(np.array([distance, binary - 0.3], dtype=object))
    program.AddLinearCost(distance)
    result = solver.Solve(program)
    details = result.get_solver_details()
    report.update(success=result.is_success(), cost=result.get_optimal_cost(),
                  binary_value=float(result.GetSolution(binary)),
                  solver_termination=str(result.get_solution_result()),
                  details_fields=[name for name in dir(details) if not name.startswith("_")])
    Path('results/diagnostics/mosek_environment_check.json').write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not result.is_success() or abs(result.get_optimal_cost() - 0.3) > 1e-6:
        raise RuntimeError("MOSEK mixed-integer smoke test failed")


if __name__ == "__main__":
    main()
