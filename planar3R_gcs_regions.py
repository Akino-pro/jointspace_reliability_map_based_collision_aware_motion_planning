from dataclasses import dataclass

import numpy as np

from planar3R_collision import is_collision_free


@dataclass
class JointSpaceBox:
    lower: np.ndarray
    upper: np.ndarray
    index: tuple
    reliability: float = 1.0

    @property
    def center(self):
        return 0.5 * (self.lower + self.upper)


def build_joint_space_boxes(cells_per_axis):
    if isinstance(cells_per_axis, int):
        cells_per_axis = (cells_per_axis, cells_per_axis, cells_per_axis)
    cells_per_axis = tuple(int(v) for v in cells_per_axis)

    axes = [np.linspace(-np.pi, np.pi, n + 1) for n in cells_per_axis]
    boxes = []
    for i in range(cells_per_axis[0]):
        for j in range(cells_per_axis[1]):
            for k in range(cells_per_axis[2]):
                lower = np.array([axes[0][i], axes[1][j], axes[2][k]])
                upper = np.array([axes[0][i + 1], axes[1][j + 1], axes[2][k + 1]])
                boxes.append(JointSpaceBox(lower=lower, upper=upper, index=(i, j, k)))
    return boxes


def sample_points_in_box(box, samples_per_axis=2):
    if samples_per_axis <= 1:
        return [box.center]

    grids = [
        np.linspace(box.lower[axis], box.upper[axis], samples_per_axis)
        for axis in range(3)
    ]
    points = []
    for theta1 in grids[0]:
        for theta2 in grids[1]:
            for theta3 in grids[2]:
                points.append(np.array([theta1, theta2, theta3]))
    points.append(box.center)
    return points


def certify_box_by_sampling(box, obstacles, samples_per_axis=2, link_radius=0.0):
    for q in sample_points_in_box(box, samples_per_axis):
        if not is_collision_free(q, obstacles, link_radius=link_radius):
            return False
    return True


def point_in_region(point, region, tol=1e-10):
    return np.all(point >= region["lower"] - tol) and np.all(point <= region["upper"] + tol)


def score_point_from_reliability_regions(point, reliability_regions, default_reliability=0.0):
    for region in reliability_regions:
        if point_in_region(point, region):
            return float(region["reliability"])
    return float(default_reliability)


def assign_box_reliability(box, reliability_regions=None, default_reliability=1.0):
    if reliability_regions is None:
        return float(default_reliability)
    return score_point_from_reliability_regions(
        box.center,
        reliability_regions,
        default_reliability=0.0,
    )


def build_safe_reliable_regions(
    cells_per_axis,
    obstacles,
    reliability_regions=None,
    samples_per_axis=2,
    link_radius=0.0,
    min_reliability=0.0,
):
    safe_boxes = []
    for box in build_joint_space_boxes(cells_per_axis):
        if not certify_box_by_sampling(box, obstacles, samples_per_axis, link_radius):
            continue
        box.reliability = assign_box_reliability(box, reliability_regions)
        if box.reliability < min_reliability:
            continue
        safe_boxes.append(box)
    return safe_boxes


def boxes_touch_or_overlap(box_a, box_b, tol=1e-10):
    for axis in range(3):
        if box_a.upper[axis] < box_b.lower[axis] - tol:
            return False
        if box_b.upper[axis] < box_a.lower[axis] - tol:
            return False
    return True


def point_in_box(point, box, tol=1e-10):
    point = np.asarray(point, dtype=float).reshape(3)
    return np.all(point >= box.lower - tol) and np.all(point <= box.upper + tol)


def adjacency_edges(boxes, tol=1e-10):
    edges = []
    for i, box_i in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if boxes_touch_or_overlap(box_i, boxes[j], tol):
                edges.append((i, j))
                edges.append((j, i))
    return edges


def box_to_hpolyhedron(box):
    try:
        from pydrake.geometry.optimization import HPolyhedron
    except ImportError as exc:
        raise ImportError("pydrake is required to convert boxes to HPolyhedron.") from exc

    a_matrix = np.vstack((np.eye(3), -np.eye(3)))
    b_vector = np.concatenate((box.upper, -box.lower))
    return HPolyhedron(a_matrix, b_vector)
