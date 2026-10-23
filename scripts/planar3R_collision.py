import numpy as np


L = [1, 1, 1]


def wrap_to_pi(theta):
    return (np.asarray(theta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def link_points(q, link_lengths=None):
    if link_lengths is None:
        link_lengths = L

    theta = np.asarray(q, dtype=float).reshape(3)
    points = [np.array([0.0, 0.0])]
    angle = 0.0
    current = points[0].copy()
    for joint_angle, link_length in zip(theta, link_lengths):
        angle += joint_angle
        current = current + link_length * np.array([np.cos(angle), np.sin(angle)])
        points.append(current.copy())
    return points


def link_segments(q, link_lengths=None):
    points = link_points(q, link_lengths)
    return list(zip(points[:-1], points[1:]))


def point_segment_distance(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-15:
        return np.linalg.norm(point - a)
    t = np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(point - closest)


def segment_intersects_circle(a, b, center, radius):
    return point_segment_distance(center, a, b) <= radius


def segment_intersects_aabb(a, b, lower, upper):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    direction = b - a
    t_min = 0.0
    t_max = 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-15:
            if a[axis] < lower[axis] or a[axis] > upper[axis]:
                return False
        else:
            inv_d = 1.0 / direction[axis]
            t1 = (lower[axis] - a[axis]) * inv_d
            t2 = (upper[axis] - a[axis]) * inv_d
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            t_min = max(t_min, t_enter)
            t_max = min(t_max, t_exit)
            if t_min > t_max:
                return False
    return True


def segment_intersects_obstacle(a, b, obstacle, link_radius=0.0):
    obstacle_type = obstacle.get("type", "circle")
    if obstacle_type == "circle":
        center = np.asarray(obstacle["center"], dtype=float)
        radius = float(obstacle["radius"]) + link_radius
        return segment_intersects_circle(a, b, center, radius)

    if obstacle_type in ("box", "aabb"):
        lower = np.asarray(obstacle["lower"], dtype=float) - link_radius
        upper = np.asarray(obstacle["upper"], dtype=float) + link_radius
        return segment_intersects_aabb(a, b, lower, upper)

    raise ValueError(f"Unsupported obstacle type: {obstacle_type}")


def is_collision_free(q, obstacles, link_lengths=None, link_radius=0.0):
    for a, b in link_segments(q, link_lengths):
        for obstacle in obstacles:
            if segment_intersects_obstacle(a, b, obstacle, link_radius):
                return False
    return True
