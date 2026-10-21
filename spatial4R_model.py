"""Standard-DH spatial 4R model from the reference script, without its grid run."""

import ast
import itertools
import operator
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REFERENCE_FILE = Path(__file__).with_name("spatial_4r_reliability_maps_proposed_method.py")


def _constant_expression(node):
    """Evaluate only numeric literals, lists, arithmetic, and the constant np.pi."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_constant_expression(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _constant_expression(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow}
    if isinstance(node, ast.BinOp) and type(node.op) in operations:
        return operations[type(node.op)](_constant_expression(node.left), _constant_expression(node.right))
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "pi":
        return float(np.pi)
    raise ValueError("Reference parameters must use numeric constant expressions only")


def reference_parameters(path=REFERENCE_FILE, parameter_set="ftwmax"):
    """Read first (ftwmax) or last (rcmax) assignments without running the script."""
    if parameter_set not in ("ftwmax", "rcmax"):
        raise ValueError("Robot parameter set must be ftwmax or rcmax")
    names = {"alpha", "d", "l", "CA", "r1", "r2", "r3", "r4"}
    assignments = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    assignments.setdefault(target.id, []).append(node.value)
    index = 0 if parameter_set == "ftwmax" else -1
    values = {name: _constant_expression(assignments[name][index]) for name in names}
    return dict(alpha=values["alpha"], d=values["d"], a=values["l"],
                reference_CA=values["CA"], reliabilities=[values[f"r{i}"] for i in range(1, 5)])


def select_robot(parameter_set="ftwmax"):
    """Select a model once per planning process, before creating any Drake plant."""
    global PARAMETERS, ALPHA, D, A, JOINT_RELIABILITIES, ROBOT_PARAMETER_SET
    PARAMETERS = reference_parameters(parameter_set=parameter_set)
    ROBOT_PARAMETER_SET = parameter_set
    ALPHA = np.array(PARAMETERS["alpha"])
    D = np.array(PARAMETERS["d"])
    A = np.array(PARAMETERS["a"])
    JOINT_RELIABILITIES = np.array(PARAMETERS["reliabilities"])


select_robot("ftwmax")
LINK_RADIUS = 0.015
START_CONFIGURATION = np.array([0.2, -0.8, 0.6, -0.3])
GOAL_REFERENCE_CONFIGURATION = np.array([0.8, -0.4, 0.2, 0.4])
OBSTACLES = [{"type": "sphere", "center": [0.5, 0.0, 0.3], "radius": 0.10}]


def wrap(q):
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


def kinematics(q):
    """Return tool transform, joint origins/axes, and DH offset/link segments."""
    q = np.asarray(q, dtype=float).reshape(4)
    transform = np.eye(4)
    origins, axes, segments = [], [], []
    for theta, a, d, alpha in zip(q, A, D, ALPHA):
        origin = transform[:3, 3].copy()
        origins.append(origin)
        axes.append(transform[:3, 2].copy())
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        after_d = origin + d * transform[:3, 2]
        endpoint = after_d + a * (c * transform[:3, 0] + s * transform[:3, 1])
        segments.extend(((origin, after_d), (after_d, endpoint)))
        local = np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                          [0, sa, ca, d], [0, 0, 0, 1]])
        transform = transform @ local
    return transform, np.asarray(origins), np.asarray(axes), segments


def position(q):
    return kinematics(q)[0][:3, 3]


def jacobian(q):
    transform, origins, axes, _ = kinematics(q)
    return np.cross(axes, transform[:3, 3] - origins).T


def collision_free(q, obstacles=OBSTACLES, link_radius=LINK_RADIUS):
    """Capsule-sphere tests on BOTH offsets and link lengths; no self-collision."""
    if np.asarray(q).shape != (4,) or not np.all(np.isfinite(q)):
        return False
    for first, second in kinematics(q)[3]:
        delta = second - first
        squared = delta @ delta
        for obstacle in obstacles:
            if obstacle.get("type") != "sphere":
                raise ValueError("Spatial 4R currently supports spherical obstacles")
            center = np.asarray(obstacle["center"])
            fraction = 0.0 if squared < 1e-24 else np.clip((center-first) @ delta / squared, 0, 1)
            if np.linalg.norm(first + fraction*delta - center) <= obstacle["radius"] + link_radius:
                return False
    return True


def validate_path(points, obstacles=OBSTACLES, link_radius=LINK_RADIUS, spacing=0.01):
    points = np.asarray(points)
    checked = 0
    for first, second in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(second-first) / spacing)))
        for fraction in np.linspace(0, 1, count+1):
            checked += 1
            if not collision_free(first + fraction*(second-first), obstacles, link_radius):
                return False, checked
    if len(points) == 1:
        return collision_free(points[0], obstacles, link_radius), 1
    return True, checked


def refine_ik(q, goal, tolerance=1e-9, max_steps=30):
    q = np.asarray(q, dtype=float).copy()
    for _ in range(max_steps):
        error = np.asarray(goal) - position(q)
        if np.linalg.norm(error) < tolerance:
            return wrap(q)
        delta = np.linalg.lstsq(jacobian(q), error, rcond=1e-10)[0]
        q += delta * min(1.0, 0.15 / max(np.linalg.norm(delta), 1e-16))
    return None


def trace_ssm(initial, goal, step=0.025, max_steps=8000):
    q = np.asarray(initial).copy()
    points, previous, initial_tangent = [q.copy()], None, None
    for index in range(max_steps):
        _, singular, directions = np.linalg.svd(jacobian(q), full_matrices=True)
        if singular[-1] < 1e-8:
            return None
        tangent = directions[-1]
        if previous is not None and tangent @ previous < 0:
            tangent = -tangent
        if initial_tangent is None:
            initial_tangent = tangent.copy()
        corrected = refine_ik(q + step*tangent, goal)
        if corrected is None:
            return None
        points.append(corrected)
        if (index > 30 and np.linalg.norm(wrap(corrected-initial)) < 0.8*step
                and tangent @ initial_tangent > 0.5):
            points.append(np.asarray(initial).copy())
            return np.asarray(points)
        q, previous = corrected, tangent
    return None


def compute_ssms(goal, seed=1, reference=None, attempts=20, max_branches=4):
    rng = np.random.default_rng(seed)
    branches, failures, completed_attempts = [], 0, 0
    for trial in range(attempts):
        if len(branches) >= max_branches:
            break
        completed_attempts += 1
        initial = np.asarray(reference) if trial == 0 and reference is not None else rng.uniform(-np.pi, np.pi, 4)
        solved = least_squares(lambda q: position(q)-goal, initial, jac=jacobian,
                               bounds=(-np.pi, np.pi), max_nfev=180,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if np.linalg.norm(position(solved.x)-goal) > 1e-7:
            continue
        initial = refine_ik(solved.x, goal)
        if initial is None or any(np.min(np.linalg.norm(wrap(b-initial), axis=1)) < 0.06 for b in branches):
            continue
        traced = trace_ssm(initial, goal)
        if traced is None:
            failures += 1
        else:
            branches.append(traced)
    return branches, dict(ik_attempts=completed_attempts, failed_traces=failures,
                         all_branches_certified=False)


def reliability_lookup(reliabilities=None):
    """The reference conditional model, given at most one joint failure."""
    reliabilities = np.asarray(JOINT_RELIABILITIES if reliabilities is None else reliabilities)
    all_work = float(np.prod(reliabilities))
    except_joint = [float(np.prod(np.delete(reliabilities, i))) for i in range(4)]
    denominator = sum(except_joint) - 3*all_work
    lookup = {frozenset(): 0.0}
    for size in range(1, 5):
        for indices in itertools.combinations(range(4), size):
            lookup[frozenset(i+1 for i in indices)] = (
                sum(except_joint[i] for i in indices) - (size-1)*all_work) / denominator
    return lookup
