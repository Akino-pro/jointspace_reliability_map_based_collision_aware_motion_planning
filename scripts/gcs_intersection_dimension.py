"""Filter existing GCS edges using the original authors' affine-dimension test."""
import time
import numpy as np


def axis_bounds(region):
    dimension = region.A.shape[1]
    lower, upper = np.full(dimension, -np.inf), np.full(dimension, np.inf)
    rows = np.count_nonzero(region.A, axis=1) == 1
    for axis in range(dimension):
        coefficients = region.A[:, axis]
        positive, negative = rows & (coefficients > 0), rows & (coefficients < 0)
        if np.any(positive):
            upper[axis] = np.min(region.b[positive]/coefficients[positive])
        if np.any(negative):
            lower[axis] = np.max(region.b[negative]/coefficients[negative])
    return lower, upper


def filter_intersection_edges(regions, edges, minimum_dimension, tolerance=1e-7, statistics=None):
    """Keep supplied directed edges whose intersection has affine dimension >= k.

    Coordinate bounds only reject provably lower-dimensional candidates within
    the stated numerical tolerance. All remaining unordered pairs are checked
    with gcs.base.polytopeDimension, using normalized halfspaces so its slack
    tolerance has a consistent geometric scale. Region geometry is unchanged.
    """
    from gcs.base import polytopeDimension
    dimension = regions[0].A.shape[1]
    if (not isinstance(minimum_dimension, int) or not 0 <= minimum_dimension <= dimension
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Use a valid intersection dimension and positive finite tolerance")
    if any(np.any(getattr(edge, "shift", 0)) for edge in edges):
        raise ValueError("Dimension filtering currently supports unshifted graph edges")
    began = time.perf_counter()
    bounds = [axis_bounds(region) for region in regions]
    lower, upper = (np.array([b[index] for b in bounds]) for index in (0, 1))
    pairs = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in edges})
    keep = set()
    screened = checked = 0
    histogram = {}
    for index, (i, j) in enumerate(pairs):
        widths = np.minimum(upper[i], upper[j])-np.maximum(lower[i], lower[j])
        upper_dimension = int(np.count_nonzero(widths > tolerance))
        if upper_dimension < minimum_dimension:
            screened += 1
        else:
            A, b = np.vstack((regions[i].A, regions[j].A)), np.r_[regions[i].b, regions[j].b]
            norms = np.linalg.norm(A, axis=1)
            if np.any(norms == 0):
                raise ValueError("Dimension testing requires nonzero halfspace normals")
            rank = polytopeDimension(A/norms[:, None], b/norms, tol=tolerance)
            checked += 1
            histogram[str(rank)] = histogram.get(str(rank), 0)+1
            if rank >= minimum_dimension:
                keep.add((i, j))
        if (index+1) % 50000 == 0 or index+1 == len(pairs):
            print(f"Intersection dimension: {index+1}/{len(pairs)} pairs; "
                  f"{checked} affine tests; {len(keep)} retained pairs", flush=True)
    retained = [(int(i), int(j)) for i, j in edges if (min(int(i), int(j)), max(int(i), int(j))) in keep]
    if statistics is not None:
        statistics.update(rule="intersection_affine_dimension_at_least_k", minimum_dimension=minimum_dimension,
                          tolerance=tolerance, halfspace_normals_normalized=True,
                          candidate_directed_edges=len(edges), directed_edges=len(retained),
                          tested_unordered_pairs=len(pairs), coordinate_screened_pairs=screened,
                          affine_dimension_test_pairs=checked, tested_dimension_histogram=histogram,
                          dimension_filter_seconds=time.perf_counter()-began)
    return retained
