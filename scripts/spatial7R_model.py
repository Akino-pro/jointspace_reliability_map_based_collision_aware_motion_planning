"""Seven-revolute DH arm and full-pose SSMs from the reference's constants only."""

import ast
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts.spatial4R_model import _constant_expression, is_duplicate_ssm_start

REFERENCE_FILE = Path(__file__).with_name("spatial_7r_FTW_estimation_ssm.py")
MAX_SSM_BRANCHES = 8
MAX_SSM_IK_ATTEMPTS = 80
SSM_TRACE_STEP = .025
LINK_RADIUS = .015
START_CONFIGURATION = np.array([.20, -.62, .30, -.55, .44, -.36, .12])
# Reachable full-pose benchmark: 0.60 m translation and +90 degrees about base Z
# relative to START_CONFIGURATION; these values are checked by regression tests.
GOAL_REFERENCE_CONFIGURATION = np.array([
    .5952876562080038, -1.3217321088017597, .768067695761899,
    -.08239295782395306, .7566153581833555, -.1892518155672116, 1.1307149220565549,
])


def reference_parameters(path=REFERENCE_FILE):
    wanted = {"alpha", "l", "d", "CA", "joint_reliabilities"}
    values = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = _constant_expression(node.value)
    result = dict(a=values["l"], d=values["d"], alpha=values["alpha"],
                  reference_CA=values["CA"], reliabilities=values["joint_reliabilities"])
    if any(len(result[key]) != 7 for key in result):
        raise ValueError("Expected seven DH/reliability/CA entries in the reference")
    return result


PARAMETERS = reference_parameters()
A, D, ALPHA = (np.asarray(PARAMETERS[key]) for key in ("a", "d", "alpha"))
JOINT_RELIABILITIES = np.asarray(PARAMETERS["reliabilities"])


def wrap(q):
    return (np.asarray(q)+np.pi) % (2*np.pi)-np.pi


def kinematics(q):
    q = np.asarray(q, dtype=float).reshape(7)
    transform = np.eye(4)
    origins, axes, segments = [], [], []
    for theta, a, d, alpha in zip(q, A, D, ALPHA):
        origin = transform[:3, 3].copy()
        origins.append(origin)
        axes.append(transform[:3, 2].copy())
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        after_d = origin+d*transform[:3, 2]
        end = after_d+a*(c*transform[:3, 0]+s*transform[:3, 1])
        segments.extend(((origin, after_d), (after_d, end)))
        transform = transform @ np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                                           [0, sa, ca, d], [0, 0, 0, 1]])
    return transform, np.asarray(origins), np.asarray(axes), segments


def pose(q):
    return kinematics(q)[0]


def position(q):
    return pose(q)[:3, 3]


def validate_pose(goal):
    goal = np.asarray(goal, dtype=float)
    if (goal.shape != (4, 4) or not np.all(np.isfinite(goal))
            or not np.allclose(goal[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(goal[:3, :3])-1) > 1e-8):
        raise ValueError("Expected a finite homogeneous pose with a proper rotation matrix")
    return goal


def jacobian(q):
    transform, origins, axes, _ = kinematics(q)
    return np.vstack((np.cross(axes, transform[:3, 3]-origins).T, axes.T))


def pose_residual(q, goal):
    actual = pose(q)
    return np.r_[actual[:3, 3]-goal[:3, 3],
                 Rotation.from_matrix(actual[:3, :3] @ goal[:3, :3].T).as_rotvec()]


def residual_jacobian(q, goal):
    """Exact local derivative of position and world-frame SO(3) log residual."""
    error = pose_residual(q, goal)[3:]
    theta = np.linalg.norm(error)
    x, y, z = error
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    coefficient = 1/12+theta**2/720 if theta < 1e-5 else 1/theta**2-.5/(theta*np.tan(theta/2))
    left_inverse = np.eye(3)-.5*skew+coefficient*(skew @ skew)
    geometric = jacobian(q)
    return np.vstack((geometric[:3], left_inverse @ geometric[3:]))


def refine_ik(q, goal, tolerance=1e-9, max_steps=40):
    q = np.asarray(q, dtype=float).copy()
    for _ in range(max_steps):
        error = pose_residual(q, goal)
        if np.linalg.norm(error) <= tolerance:
            return wrap(q)
        delta = np.linalg.lstsq(residual_jacobian(q, goal), -error, rcond=1e-10)[0]
        q = wrap(q+delta*min(1., .15/max(np.linalg.norm(delta), 1e-16)))
    return None


def trace_ssm(initial, goal, step=SSM_TRACE_STEP, max_steps=8000):
    q = np.asarray(initial).copy()
    points, previous, initial_tangent = [q.copy()], None, None
    for index in range(max_steps):
        _, singular, directions = np.linalg.svd(jacobian(q), full_matrices=True)
        if singular[-1] < 1e-7:
            return None  # Higher-dimensional/singular SSMs are not curve traces.
        tangent = directions[-1]
        if previous is not None and tangent @ previous < 0:
            tangent = -tangent
        if initial_tangent is None:
            initial_tangent = tangent.copy()
        corrected = refine_ik(q+step*tangent, goal)
        if corrected is None:
            return None
        points.append(corrected)
        if index > 30 and np.linalg.norm(wrap(corrected-initial)) < .8*step and tangent @ initial_tangent > .5:
            points.append(np.asarray(initial).copy())
            return np.asarray(points)
        q, previous = corrected, tangent
    return None


def compute_ssms(goal, seed=1, reference=None, attempts=MAX_SSM_IK_ATTEMPTS,
                 max_branches=MAX_SSM_BRANCHES):
    goal = validate_pose(goal)
    if (not isinstance(attempts, (int, np.integer)) or attempts < 1
            or not isinstance(max_branches, (int, np.integer)) or not 1 <= max_branches <= 8):
        raise ValueError("Use a positive IK budget and branch limit from one to eight")
    if reference is not None and (np.asarray(reference).shape != (7,) or not np.all(np.isfinite(reference))):
        raise ValueError("Reference seed must contain seven finite angles")
    rng = np.random.default_rng(seed)
    branches, failures, duplicate, failed_ik, completed = [], 0, 0, 0, 0
    for index in range(attempts):
        if len(branches) == max_branches:
            break
        completed += 1
        initial = wrap(reference) if index == 0 and reference is not None else rng.uniform(-np.pi, np.pi, 7)
        solved = least_squares(lambda q: pose_residual(q, goal), initial,
                               jac=lambda q: residual_jacobian(q, goal), max_nfev=180,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        initial = refine_ik(solved.x, goal) if np.linalg.norm(solved.fun) < 1e-6 else None
        if initial is None:
            failed_ik += 1
            continue
        if is_duplicate_ssm_start(initial, branches):
            duplicate += 1
            continue
        branch = trace_ssm(initial, goal)
        if branch is None:
            failures += 1
        else:
            branches.append(branch)
    return branches, dict(branch_limit=max_branches, branch_count=len(branches), ik_attempt_budget=attempts,
                         ik_attempts=completed, failed_ik=failed_ik, failed_traces=failures,
                         duplicate_starts=duplicate, all_branches_certified=False,
                         termination_reason="branch_limit_reached" if len(branches) == max_branches else "ik_attempt_budget_exhausted")


def collision_free(q, obstacles, link_radius=LINK_RADIUS):
    if np.asarray(q).shape != (7,) or not np.all(np.isfinite(q)):
        return False
    for first, second in kinematics(q)[3]:
        delta = second-first
        squared = delta @ delta
        for obstacle in obstacles:
            if obstacle.get("type") != "sphere":
                raise ValueError("Spatial 7R currently supports spherical obstacles")
            center = np.asarray(obstacle["center"])
            t = 0. if squared < 1e-24 else np.clip((center-first) @ delta/squared, 0, 1)
            if np.linalg.norm(first+t*delta-center) <= obstacle["radius"]+link_radius:
                return False
    return True


def validate_path(points, obstacles, link_radius=LINK_RADIUS, spacing=.01):
    checked = 0
    pairs = list(zip(points[:-1], points[1:])) or [(points[0], points[0])]
    for first, second in pairs:
        for t in np.linspace(0, 1, max(1, int(np.ceil(np.linalg.norm(second-first)/spacing)))+1):
            checked += 1
            if not collision_free(first+t*(second-first), obstacles, link_radius):
                return False, checked
    return True, checked


def reliability_lookup():
    r = JOINT_RELIABILITIES
    all_work = float(np.prod(r))
    single = np.array([(1-r[i])*np.prod(np.delete(r, i)) for i in range(7)])
    denominator = all_work+single.sum()
    return {
        frozenset(i+1 for i in indices): (0. if not indices else float((all_work+single[list(indices)].sum())/denominator))
        for size in range(8) for indices in itertools.combinations(range(7), size)}


def default_obstacles(start=START_CONFIGURATION, goal=None):
    target = pose(GOAL_REFERENCE_CONFIGURATION) if goal is None else goal
    center = (position(start)+target[:3, 3])/2
    return [dict(type="sphere", center=center.tolist(), radius=.08)]


OBSTACLES = default_obstacles()
