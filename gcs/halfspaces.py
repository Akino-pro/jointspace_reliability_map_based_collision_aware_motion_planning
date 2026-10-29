"""Conservative removal of redundant linear inequalities; no region approximation."""
import numpy as np


def compact_halfspaces(A, b):
    """Keep the tightest inequality for each exactly identical row direction.

    Axis rows are normalized to +/- unit coordinates, enabling removal of
    duplicate IRIS and reliability-box bounds. Other normals stay unchanged.
    """
    rows = {}
    for row, bound in zip(np.asarray(A), np.asarray(b)):
        row = row.copy()
        nonzero = np.flatnonzero(row)
        if len(nonzero) == 1:
            scale = abs(row[nonzero[0]])
            row, bound = row/scale, bound/scale
        key = tuple(row)
        rows[key] = min(rows.get(key, np.inf), bound)
    return np.array(list(rows), dtype=float).reshape(-1, A.shape[1]), np.array(list(rows.values()))


def nonimplied_rows(A, b, target_A, target_b):
    """Return source rows not already implied by target rows or its axis box."""
    target = {}
    for row, bound in zip(target_A, target_b):
        key = tuple(row)
        target[key] = min(target.get(key, np.inf), bound)
    dimension = A.shape[1]
    lower, upper = np.full(dimension, -np.inf), np.full(dimension, np.inf)
    for row, bound in zip(target_A, target_b):
        nonzero = np.flatnonzero(row)
        if len(nonzero) == 1:
            axis = nonzero[0]
            value = bound/row[axis]
            if row[axis] > 0:
                upper[axis] = min(upper[axis], value)
            else:
                lower[axis] = max(lower[axis], value)
    bounded = np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    retained = []
    for index, (row, bound) in enumerate(zip(A, b)):
        if tuple(row) in target and target[tuple(row)] <= bound:
            continue
        if bounded:
            support = np.sum(np.where(row >= 0, row*upper, row*lower))
            margin = 1e-10*(1+abs(bound)+np.sum(np.abs(row)*np.maximum(abs(lower), abs(upper))))
            if support <= bound-margin:
                continue
        retained.append(index)
    return retained
