"""Periodic graph edges and continuous lifts of revolute-joint paths."""
import numpy as np


class RegionEdge(tuple):
    """A pair compatible with NetworkX, carrying a destination chart shift.

    The traversed segment ends at x_destination + shift in the source chart.
    Parallel edges may connect the same region pair through different seams.
    """
    def __new__(cls, source, destination, shift):
        obj = super().__new__(cls, (source, destination))
        obj.shift = np.asarray(shift, dtype=float)
        return obj


def remap_edges(edges, mapping):
    return [RegionEdge(mapping[e[0]], mapping[e[1]], e.shift)
            if hasattr(e, "shift") else (mapping[e[0]], mapping[e[1]])
            for e in edges if e[0] in mapping and e[1] in mapping]


def lift_path(path, result, edge_shifts, dimension):
    """Extract continuous angles without shortest-angle shortcuts inside regions."""
    offset = np.zeros(dimension)
    points = []
    for edge in path:
        offset = offset + edge_shifts.get(edge.name(), np.zeros(dimension))
        points.append(result.GetSolution(edge.xv()) + offset)
    return np.asarray(points), offset


def angle_error(first, second):
    delta = np.asarray(first)-np.asarray(second)
    return float(np.linalg.norm((delta+np.pi) % (2*np.pi)-np.pi))


def path_in_regions(solved, regions, start, target, tolerance=1e-7):
    """Check lifted segments in their original charts and the terminal set."""
    points = np.asarray(solved["waypoints"])
    route = solved["region_route"]
    if points.shape != (len(route)+1, len(start)) or not np.all(np.isfinite(points)):
        return False
    if np.linalg.norm(points[0]-start) > tolerance:
        return False
    shifts = solved["route_spec"]["shifts"]
    offset = np.zeros(len(start))
    for step, index in enumerate(route):
        if not all(regions[index].contains(p-offset, tol=tolerance)
                   for p in points[step:step+2]):
            return False
        offset += shifts[step]
    return target.contains(points[-1]-offset, tol=tolerance)
