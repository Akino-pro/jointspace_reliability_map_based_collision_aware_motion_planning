"""Numerical graph-model optimality bounds for nonnegative path costs."""

import math


def bound_quality(lower, upper, relative_tolerance=0.01, absolute_tolerance=1e-6,
                  scope="graph_model"):
    if (not math.isfinite(relative_tolerance) or relative_tolerance < 0
            or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0):
        raise ValueError("Gap tolerances must be finite and nonnegative")
    report = {"lower_bound": None, "upper_bound": upper,
              "absolute_gap": None, "relative_gap": None,
              "relative_tolerance": relative_tolerance,
              "absolute_tolerance": absolute_tolerance, "scope": scope,
              "within_tolerance": False, "status": "bound_unavailable"}
    if (lower is None or upper is None or not math.isfinite(lower)
            or not math.isfinite(upper)):
        return report
    # Zero is also a valid lower bound because all modeled path costs are >=0.
    lower = max(0.0, float(lower))
    upper = float(upper)
    report.update(lower_bound=lower, upper_bound=upper)
    numerical_tolerance = absolute_tolerance + 1e-8 * max(1.0, abs(lower), abs(upper))
    if lower > upper + numerical_tolerance or upper < -numerical_tolerance:
        report["status"] = "inconsistent_bounds"
        return report
    gap = max(0.0, upper - lower)
    relative_gap = gap / lower if lower > 0 else (0.0 if gap == 0 else None)
    within = gap <= absolute_tolerance + relative_tolerance * lower
    report.update(absolute_gap=gap, relative_gap=relative_gap,
                  within_tolerance=within,
                  status="within_tolerance" if within else "gap_exceeds_tolerance")
    return report
