"""Artificial limits for pre-failure planning; physical chart stays [-pi,pi]^3."""

import itertools

import numpy as np
from scipy.optimize import least_squares

from planar3R_collision import L
from planar3R_iris_regions import ReliablePolytope, interior_point


def validate_joint_limits(lower=None, upper=None):
    if (lower is None) != (upper is None):
        raise ValueError("Both artificial lower and upper bounds are required")
    lower = np.full(3, -np.pi) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(3, np.pi) if upper is None else np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper)) or np.any(lower >= upper)
            or np.any(lower < -np.pi) or np.any(upper > np.pi)):
        raise ValueError("Use three finite, positive-width joint intervals within [-pi,pi]")
    return lower, upper


def within_joint_limits(q, lower, upper, tolerance=1e-8):
    q = np.asarray(q)
    return bool(np.all(q >= lower - tolerance) and np.all(q <= upper + tolerance))


def clip_regions_to_joint_limits(regions, lower, upper):
    clipped = []
    for region in regions:
        A = np.vstack((region.A, np.eye(3), -np.eye(3)))
        b = np.concatenate((region.b, upper, -lower))
        if interior_point(A, b) is not None:
            clipped.append(ReliablePolytope(A, b, region.reliability,
                                            region.parent_index, region.partition_index))
    return clipped


def _trig_range(lower, upper, sine=False):
    if upper - lower >= 2 * np.pi:
        return -1.0, 1.0
    function = np.sin if sine else np.cos
    offset = np.pi / 2 if sine else 0.0
    first = int(np.ceil((lower - offset) / np.pi))
    last = int(np.floor((upper - offset) / np.pi))
    points = [lower, upper] + [offset + k * np.pi for k in range(first, last + 1)]
    values = function(points)
    return float(np.min(values)), float(np.max(values))


def workspace_bounds(lower, upper):
    """Conservative Cartesian enclosure of every configuration in a joint box."""
    angle_lower, angle_upper = np.cumsum(lower), np.cumsum(upper)
    bounds = np.zeros((2, 2))
    for lo, hi, length in zip(angle_lower, angle_upper, L):
        bounds[0] += length * np.asarray(_trig_range(lo, hi))
        bounds[1] += length * np.asarray(_trig_range(lo, hi, sine=True))
    return bounds[:, 0], bounds[:, 1]


def find_limited_ik(point, lower, upper, max_boxes=4096):
    """Return (configuration, status) before constructing any GCS graph.

    A bounded IK solution establishes reachability. Conservative interval
    enclosures can establish unreachability. A finite search that proves neither
    returns 'unresolved', never a false claim of unreachability.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("Workspace endpoints must be finite 2D positions")

    def outside(lo, hi):
        box_lower, box_upper = workspace_bounds(lo, hi)
        return bool(np.any(point < box_lower - 1e-10) or np.any(point > box_upper + 1e-10))

    if np.linalg.norm(point) > sum(L) + 1e-10 or outside(lower, upper):
        return None, "unreachable"

    def residual(q):
        angles = np.cumsum(q)
        return np.array([np.dot(L, np.cos(angles)), np.dot(L, np.sin(angles))]) - point

    def jacobian(q):
        angles = np.cumsum(q)
        return np.array([[-np.sum(np.asarray(L)[i:] * np.sin(angles[i:])) for i in range(3)],
                         [np.sum(np.asarray(L)[i:] * np.cos(angles[i:])) for i in range(3)]])

    def solve(seed, lo=lower, hi=upper):
        result = least_squares(residual, np.clip(seed, lo, hi), jac=jacobian,
                               bounds=(lo, hi), max_nfev=200,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if np.linalg.norm(residual(result.x)) <= 1e-8:
            return result.x
        return None

    center = 0.5 * (lower + upper)
    q = solve(center)
    if q is not None:
        return q, "reachable"
    for fractions in itertools.product((0.1, 0.5, 0.9), repeat=3):
        q = solve(lower + np.asarray(fractions) * (upper - lower))
        if q is not None:
            return q, "reachable"
    pending = [(lower.copy(), upper.copy())]
    unresolved = False
    for index in range(max_boxes):
        if not pending:
            return None, "unresolved" if unresolved else "unreachable"
        lo, hi = pending.pop()
        if outside(lo, hi):
            continue
        if index % 128 == 0:
            q = solve(0.5 * (lo + hi), lo, hi)
            if q is not None:
                return q, "reachable"
        axis = int(np.argmax((hi - lo) * np.array([3, 2, 1])))
        if hi[axis] - lo[axis] < 1e-6:
            unresolved = True
            continue
        mid = 0.5 * (lo[axis] + hi[axis])
        first_hi, second_lo = hi.copy(), lo.copy()
        first_hi[axis], second_lo[axis] = mid, mid
        pending.extend(((lo, first_hi), (second_lo, hi)))
    return None, "unresolved"
