"""Adaptive, chart-local sampled SSM hulls and continuous IK projection.

No Drake dependency. Sample hulls are not certified continuous outer envelopes.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull

from scripts.planar3R_collision import is_collision_free, link_points
from scripts.planar3R_iris_regions import ReliablePolytope


# Joint-space arc length in radians, shared by every target SSM branch.
# Approximately the density of the former 300-point (1.5, 1.5) example.
SSM_ARC_SPACING = 0.044


def resample_by_arclength(points, spacing):
    """Uniformly sample a chart-local polyline, preserving both endpoints.

    Spacing is an upper bound on nominal polyline arc intervals. Interpolated
    points require IK refinement before being treated as actual SSM samples.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("SSM arc spacing must be positive and finite")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.all(np.isfinite(points)):
        raise ValueError("SSM points must be finite ordered joint configurations")
    if len(points) < 2:
        return points.copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-12]]
    if len(points) < 2:
        return points.copy()
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = int(np.ceil(cumulative[-1] / spacing)) + 1
    positions = np.linspace(0, cumulative[-1], count)
    return np.column_stack([np.interp(positions, cumulative, points[:, i]) for i in range(points.shape[1])])


@dataclass
class LocalEnvelope:
    samples: np.ndarray
    polytope: ReliablePolytope
    branch: int
    piece: int
    polyline_distance_bound: float


def sample_hull(points):
    """Exact affine hull halfspaces, including points and lower-dimensional hulls."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    origin = points[0]
    _, singular, basis = np.linalg.svd(points - origin, full_matrices=True)
    rank = int(np.sum(singular > 1e-10))
    axes = basis[:rank]
    normal = basis[rank:]
    A = [*normal, *(-normal)]
    b = [*(normal @ origin), *(-normal @ origin)]
    if rank == 1:
        coordinates = (points - origin) @ axes.T
        A.extend([axes[0], -axes[0]])
        b.extend([coordinates.max() + axes[0] @ origin,
                  -coordinates.min() - axes[0] @ origin])
    elif rank >= 2:
        hull = ConvexHull((points - origin) @ axes.T)
        world = hull.equations[:, :-1] @ axes
        A.extend(world)
        b.extend(world @ origin - hull.equations[:, -1])
    return ReliablePolytope(np.asarray(A).reshape(-1, points.shape[1]), np.asarray(b))


def chord_bound(points):
    """Bound hull-to-sampled-polyline distance by twice the chord deviation.

    Every hull point is within d of the chord. Continuity of the polyline's
    axial projection supplies a polyline point within d of each chord point.
    This says nothing about the unknown continuous curve between samples.
    """
    delta = points[-1] - points[0]
    squared = delta @ delta
    t = np.zeros(len(points)) if squared < 1e-24 else np.clip(
        (points - points[0]) @ delta / squared, 0, 1)
    return float(2 * np.max(np.linalg.norm(points - points[0] - t[:, None] * delta, axis=1)))


def build_envelopes(branches, epsilon=0.025, samples_per_branch=None,
                    arc_spacing=SSM_ARC_SPACING, refine_sample=None, sampling_report=None):
    if (not np.isfinite(epsilon) or epsilon <= 0
            or (samples_per_branch is not None and samples_per_branch < 2)
            or not np.isfinite(arc_spacing) or arc_spacing <= 0):
        raise ValueError("Use positive finite epsilon/arc spacing and a valid optional sample budget")
    envelopes = []
    branch_lengths = []
    sample_counts = []
    refinement_failures = 0

    def divide(points, branch, piece):
        bound = chord_bound(points)
        if bound <= epsilon or len(points) <= 2:
            envelopes.append(LocalEnvelope(points, sample_hull(points), branch, piece, bound))
        else:
            middle = len(points) // 2
            divide(points[:middle + 1], branch, piece)
            divide(points[middle:], branch, piece)

    for branch_index, branch in enumerate(branches):
        branch = np.asarray(branch, dtype=float)
        if branch.ndim != 2 or branch.shape[1] < 2 or not np.all(np.isfinite(branch)):
            raise ValueError("SSM branches must contain finite ordered joint configurations")
        if not len(branch):
            branch_lengths.append(0.0)
            sample_counts.append(0)
            continue
        wrapped_steps = (np.diff(branch, axis=0) + np.pi) % (2 * np.pi) - np.pi
        branch_lengths.append(float(np.linalg.norm(wrapped_steps, axis=1).sum()))
        sampled_points = set()
        # Split BEFORE subsampling; a hull must never bridge a periodic seam.
        cuts = np.flatnonzero(np.any(np.abs(np.diff(branch, axis=0)) > np.pi, axis=1)) + 1
        for piece_index, piece in enumerate(np.split(branch, cuts)):
            if samples_per_branch is not None:
                count = max(2, int(np.ceil(samples_per_branch * len(piece) / len(branch))))
                indices = np.unique(np.linspace(0, len(piece) - 1, min(len(piece), count), dtype=int))
                samples = piece[indices]
            else:
                samples = resample_by_arclength(piece, arc_spacing)
            # Never bridge a failed IK correction or a newly introduced seam.
            group = []
            for q in samples:
                corrected = q if refine_sample is None else refine_sample(q)
                if corrected is None:
                    refinement_failures += 1
                    if group:
                        divide(np.asarray(group), branch_index, piece_index)
                        group = []
                    continue
                corrected = np.asarray(corrected, dtype=float)
                if group and np.any(np.abs(corrected - group[-1]) > np.pi):
                    divide(np.asarray(group), branch_index, piece_index)
                    group = []
                group.append(corrected)
                sampled_points.add(tuple(corrected))
            if group:
                divide(np.asarray(group), branch_index, piece_index)
        sample_counts.append(len(sampled_points))
    if sampling_report is not None:
        sampling_report.update(
            sampling_mode="legacy_branch_budget" if samples_per_branch is not None else "fixed_arc_spacing",
            ssm_arc_spacing=arc_spacing if samples_per_branch is None else None,
            ssm_branch_arc_lengths=branch_lengths, ssm_total_arc_length=sum(branch_lengths),
            ssm_branch_sample_counts=sample_counts, sample_refinement_failures=refinement_failures)
    return envelopes


def intersect_envelopes(envelopes, regions):
    """Retain feasible intersections, including zero-volume curve/point hulls."""
    targets = []
    for envelope_index, envelope in enumerate(envelopes):
        for region_index, region in enumerate(regions):
            A = np.vstack((envelope.polytope.A, region.A))
            b = np.concatenate((envelope.polytope.b, region.b))
            dimension = A.shape[1]
            feasible = linprog(np.zeros(dimension), A_ub=A, b_ub=b,
                               bounds=[(None, None)] * dimension, method="highs")
            if feasible.success:
                targets.append({"polytope": ReliablePolytope(A, b),
                                "envelope": envelope_index, "region": region_index})
    return targets


def project_to_local_ssm(point, envelope, region, goal, obstacles, link_radius,
                         position=None, collision_free=None):
    """Multistart constrained local projection onto continuous FK equality.

    The sample coordinate box restricts the selected local portion; it can omit
    unsampled extrema. This is local numerical projection, not a global proof.
    """
    point = np.asarray(point)
    position = position or (lambda q: link_points(q)[-1])
    collision_free = collision_free or (lambda q: is_collision_free(q, obstacles, link_radius=link_radius))
    samples = envelope.samples
    lower, upper = samples.min(axis=0), samples.max(axis=0)
    constraints = [
        {"type": "eq", "fun": lambda q: position(q) - goal},
        {"type": "ineq", "fun": lambda q: region.b - region.A @ q},
    ]
    candidates = []
    # Include exact sampled endpoints and use nearby samples as local seeds.
    order = np.argsort(np.linalg.norm(samples - point, axis=1))
    for q in samples:
        if (region.contains(q) and np.linalg.norm(position(q) - goal) <= 1e-7
                and collision_free(q)):
            candidates.append(q)
    for index in order[:8]:
        result = minimize(lambda q: np.sum((q - point)**2), samples[index],
                          method="SLSQP", bounds=list(zip(lower, upper)),
                          constraints=constraints, options={"ftol": 1e-12, "maxiter": 200})
        q = result.x
        if (result.success and np.all(np.isfinite(q)) and region.contains(q)
                and np.linalg.norm(position(q) - goal) <= 1e-7
                and collision_free(q)):
            candidates.append(q)
    if not candidates:
        return None
    return min(candidates, key=lambda q: np.linalg.norm(q - point)).copy()
