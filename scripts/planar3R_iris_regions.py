"""Portable IRIS checkpoints and exact intersections with reliability boxes.

Geometry helpers use NumPy/SciPy; Drake is imported only for conversion to GCS.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


@dataclass
class ReliablePolytope:
    A: np.ndarray
    b: np.ndarray
    reliability: float = 1.0
    parent_index: int = -1
    partition_index: int = -1

    def contains(self, point, tol=1e-8):
        return bool(np.all(self.A @ np.asarray(point).reshape(self.A.shape[1]) <= self.b + tol))

    def to_drake(self):
        from pydrake.geometry.optimization import HPolyhedron

        return HPolyhedron(self.A, self.b)


def interior_point(A, b, min_radius=1e-8):
    """Chebyshev LP; discard empty and numerically flat intersection cells."""
    dimension = A.shape[1]
    result = linprog(
        [0.0] * dimension + [-1.0],
        A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
        b_ub=b,
        bounds=[(None, None)] * dimension + [(0.0, None)],
        method="highs",
    )
    if result.success and result.x[dimension] > min_radius:
        return result.x[:dimension]
    return None


def split_iris_regions(regions, reliability_regions, min_reliability=0.0):
    """Intersect original IRIS halfspaces with each reliability box exactly."""
    children = []
    for i, region in enumerate(regions):
        if reliability_regions is None:
            if region.reliability >= min_reliability:
                children.append(region)
            continue
        for j, partition in enumerate(reliability_regions):
            reliability = float(partition["reliability"])
            if reliability < min_reliability:
                continue
            lower, upper = np.asarray(partition["lower"]), np.asarray(partition["upper"])
            if np.any(upper - lower <= 1e-10):
                continue
            dimension = region.A.shape[1]
            # Box support is an exact upper bound for each original halfspace.
            # Drop only strictly implied rows; retain rows near numerical ties.
            support = np.sum(np.where(region.A >= 0, region.A * upper, region.A * lower), axis=1)
            needed = support > region.b - 1e-10
            A = np.vstack((region.A[needed], np.eye(dimension), -np.eye(dimension)))
            b = np.concatenate((region.b[needed], upper, -lower))
            if interior_point(A, b) is not None:
                children.append(ReliablePolytope(A, b, reliability, i, j))
    return children


def polytope_adjacency(regions, periodic=False, broad_phase=True, statistics=None, progress=None):
    """Conservative AABB screening, followed by exact intersection LPs.

    Optional 2pi translations connect chart seams, including simultaneous
    multi-axis crossings. Bounding LP failures never silently discard a pair.
    This changes graph coverage, not the original collision polytopes.
    """
    from itertools import product
    from scripts.periodic_graph import RegionEdge
    tolerance = 1e-8
    edges = []
    boxes = []
    bounding_lps = 0
    if broad_phase or periodic:
        for region in regions:
            dimension = region.A.shape[1]
            bounds = []
            for axis in np.eye(dimension):
                coordinate = int(np.argmax(axis))
                rows = np.count_nonzero(region.A, axis=1) == 1
                coefficients = region.A[:, coordinate]
                positive = rows & (coefficients > 0)
                negative = rows & (coefficients < 0)
                if np.any(positive) and np.any(negative):
                    bounds.append([np.max(region.b[negative]/coefficients[negative])-tolerance,
                                   np.min(region.b[positive]/coefficients[positive])+tolerance])
                    continue
                bounding_lps += 2
                opts = [linprog(sign*axis, A_ub=region.A, b_ub=region.b,
                                bounds=[(None, None)]*dimension, method="highs")
                        for sign in (1, -1)]
                if any(not r.success for r in opts):
                    raise ValueError("Adjacency requires nonempty bounded regions")
                bounds.append([opts[0].fun-tolerance, -opts[1].fun+tolerance])
            boxes.append(np.asarray(bounds))
    tested = candidates = 0
    box_array = np.asarray(boxes)
    for i, first in enumerate(regions):
        if broad_phase and not periodic:
            # Screen all later boxes in NumPy, preserving the exact pair order
            # and the same conservative tolerances as the scalar test below.
            later = box_array[i+1:]
            overlap = np.all((box_array[i, :, 0] <= later[:, :, 1]) &
                             (later[:, :, 0] <= box_array[i, :, 1]), axis=1)
            indices = np.flatnonzero(overlap)+i+1
        else:
            indices = range(i+1, len(regions))
        for j in indices:
            j = int(j)
            second = regions[j]
            dimension = first.A.shape[1]
            if periodic:
                choices = [[k for k in (-1, 0, 1)
                            if boxes[i][a, 0] <= boxes[j][a, 1]+2*np.pi*k
                            and boxes[j][a, 0]+2*np.pi*k <= boxes[i][a, 1]]
                           for a in range(dimension)]
                shifts = [2*np.pi*np.asarray(k) for k in product(*choices)]
            elif broad_phase and np.any((boxes[i][:, 0] > boxes[j][:, 1]) |
                                         (boxes[j][:, 0] > boxes[i][:, 1])):
                shifts = []
            else:
                shifts = [np.zeros(dimension)]
            candidates += len(shifts)
            for shift in shifts:
                tested += 1
                result = linprog(np.zeros(dimension),
                    A_ub=np.vstack((first.A, second.A)),
                    b_ub=np.concatenate((first.b, second.b+second.A@shift)),
                    bounds=[(None, None)]*dimension, method="highs")
                if result.success:
                    edges.extend((RegionEdge(i, j, shift), RegionEdge(j, i, -shift))
                                 if periodic else ((i, j), (j, i)))
                elif result.status != 2:
                    raise RuntimeError(f"Adjacency LP failed: {result.message}")
        if progress is not None:
            progress(i+1, len(regions), tested, len(edges))
    if statistics is not None:
        statistics.update(input_region_count=len(regions), region_pairs=len(regions)*(len(regions)-1)//2,
                          bounding_lp_count=bounding_lps,
                          candidate_intersections=candidates, intersection_lp_count=tested,
                          directed_edges=len(edges), periodic=periodic,
                          periodic_directed_edges=sum(bool(np.any(getattr(e, "shift", 0))) for e in edges))
    return edges


def save_iris_checkpoint(path, regions, metadata):
    """Atomically replace a numeric, pickle-free NPZ after each completed seed."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    for i, region in enumerate(regions):
        arrays[f"A_{i}"] = region.A
        arrays[f"b_{i}"] = region.b
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_iris_checkpoint(path, allow_empty=False, expected_dimension=3):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        count = len([name for name in arrays.files if name.startswith("A_")])
        regions = [ReliablePolytope(arrays[f"A_{i}"].copy(), arrays[f"b_{i}"].copy())
                   for i in range(count)]
    if not regions and not allow_empty:
        raise ValueError("The IRIS checkpoint contains no completed regions.")
    for region in regions:
        if (region.A.ndim != 2 or region.A.shape[1] != expected_dimension
                or region.b.shape != (len(region.A),)):
            raise ValueError(f"Invalid {expected_dimension}D IRIS halfspace dimensions.")
        if not np.all(np.isfinite(region.A)) or not np.all(np.isfinite(region.b)):
            raise ValueError("Non-finite IRIS halfspaces.")
    return regions, metadata
