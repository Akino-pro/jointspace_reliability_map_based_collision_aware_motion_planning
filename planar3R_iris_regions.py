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
        return bool(np.all(self.A @ np.asarray(point).reshape(3) <= self.b + tol))

    def to_drake(self):
        from pydrake.geometry.optimization import HPolyhedron

        return HPolyhedron(self.A, self.b)


def interior_point(A, b, min_radius=1e-8):
    """Chebyshev LP; discard empty and numerically flat intersection cells."""
    result = linprog(
        [0.0, 0.0, 0.0, -1.0],
        A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
        b_ub=b,
        bounds=[(None, None)] * 3 + [(0.0, None)],
        method="highs",
    )
    if result.success and result.x[3] > min_radius:
        return result.x[:3]
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
            A = np.vstack((region.A, np.eye(3), -np.eye(3)))
            b = np.concatenate((region.b, upper, -lower))
            if interior_point(A, b) is not None:
                children.append(ReliablePolytope(A, b, reliability, i, j))
    return children


def polytope_adjacency(regions):
    """Feasibility LP retains overlaps and face/edge contacts."""
    edges = []
    for i, first in enumerate(regions):
        for j in range(i + 1, len(regions)):
            second = regions[j]
            result = linprog(
                np.zeros(3),
                A_ub=np.vstack((first.A, second.A)),
                b_ub=np.concatenate((first.b, second.b)),
                bounds=[(None, None)] * 3,
                method="highs",
            )
            if result.success:
                edges.extend(((i, j), (j, i)))
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


def load_iris_checkpoint(path, allow_empty=False):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        count = len([name for name in arrays.files if name.startswith("A_")])
        regions = [ReliablePolytope(arrays[f"A_{i}"].copy(), arrays[f"b_{i}"].copy())
                   for i in range(count)]
    if not regions and not allow_empty:
        raise ValueError("The IRIS checkpoint contains no completed regions.")
    for region in regions:
        if region.A.ndim != 2 or region.A.shape[1] != 3 or region.b.shape != (len(region.A),):
            raise ValueError("Invalid planar 3R IRIS halfspace dimensions.")
        if not np.all(np.isfinite(region.A)) or not np.all(np.isfinite(region.b)):
            raise ValueError("Non-finite IRIS halfspaces.")
    return regions, metadata
