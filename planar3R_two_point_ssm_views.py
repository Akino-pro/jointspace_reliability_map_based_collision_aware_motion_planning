import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

from helper_functions import forward_kinematics_3R, Jacobian_3R


L = [1, 1, 1]
STEP_SIZE = 0.01
TERMINATE_THRESHOLD = 9.0 / 5.0 * STEP_SIZE
MAX_SSM = 2
MAX_RANDOM_TRIALS = 80
MAX_TRACE_STEPS = 20000

POINT_1 = (0.5, 0.5)
POINT_2 = (1.5, 1.5)


def wrap_to_pi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def angular_distance(theta_a, theta_b):
    theta_a = np.asarray(theta_a).reshape(3)
    theta_b = np.asarray(theta_b).reshape(3)
    return np.linalg.norm(wrap_to_pi(theta_a - theta_b))


def solve_ik_dls(x_target, rng):
    q = rng.uniform(low=-np.pi, high=np.pi, size=(3, 1))
    error_threshold = 1e-6
    damping = 0.5

    for _ in range(10000):
        x_current = forward_kinematics_3R(q, L)
        delta_x = x_target - x_current
        error = np.linalg.norm(delta_x)
        if error <= error_threshold:
            return q

        j = Jacobian_3R(q, L)
        correction = j.T @ np.linalg.inv(j @ j.T + damping**2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9:
            return None

        q = wrap_to_pi(q + 0.1 * correction)

        next_x = forward_kinematics_3R(q, L)
        if np.linalg.norm(next_x - x_current) < 1e-8:
            return None

    return None


def stepwise_ssm(theta, n_j, x_target, previous_n_j):
    current_n = np.array(n_j).reshape(-1)
    previous_n = np.array(previous_n_j).reshape(-1)
    cos_theta = np.dot(current_n, previous_n) / (
        np.linalg.norm(current_n) * np.linalg.norm(previous_n)
    )
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if np.abs(angle_rad) > np.pi / 2.0:
        n_j = -n_j

    x = forward_kinematics_3R(theta, L)
    delta_x_step = x_target - x
    j_step = Jacobian_3R(theta, L)
    corrected_delta_theta = np.linalg.pinv(j_step) @ delta_x_step

    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = wrap_to_pi(theta + STEP_SIZE * dq)

    new_n_j = null_space(Jacobian_3R(theta_next, L))
    return theta_next, new_n_j, n_j


def is_duplicate_ssm_start(theta, existing_ssms):
    for ssm in existing_ssms:
        if np.any([angular_distance(theta, old_theta) <= TERMINATE_THRESHOLD for old_theta in ssm]):
            return True
    return False


def trace_ssm(theta_start, x_target):
    theta = theta_start.copy()
    theta_prime = theta_start.copy()
    n_j = null_space(Jacobian_3R(theta, L))
    if n_j.shape[1] == 0:
        return None

    old_n_j = n_j.copy()
    ssm_theta_list = [theta.copy()]
    closest_return = np.inf

    for step_index in range(MAX_TRACE_STEPS):
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j)
        n_j = new_n_j
        ssm_theta_list.append(theta.copy())

        return_distance = angular_distance(theta, theta_prime)
        closest_return = min(closest_return, return_distance)
        if return_distance <= TERMINATE_THRESHOLD and step_index > 4:
            return np.array(ssm_theta_list).reshape((-1, 3))

        if step_index == 15000 and closest_return > TERMINATE_THRESHOLD:
            theta_prime = theta.copy()
            ssm_theta_list = [theta.copy()]
            closest_return = np.inf

    return None


def compute_ssms_for_target(x, y, seed=0):
    target = np.array([x, y], dtype=float).reshape((2, 1))
    if np.linalg.norm(target) > np.sum(L) + 1e-9:
        raise ValueError(f"Target ({x}, {y}) is outside the maximum reach {np.sum(L)}.")

    rng = np.random.default_rng(seed)
    ssms = []

    for _ in range(MAX_RANDOM_TRIALS):
        if len(ssms) >= MAX_SSM:
            break

        ik_solution = solve_ik_dls(target, rng)
        if ik_solution is None or is_duplicate_ssm_start(ik_solution, ssms):
            continue

        ssm = trace_ssm(ik_solution, target)
        if ssm is None:
            continue

        if not is_duplicate_ssm_start(ssm[0].reshape((3, 1)), ssms):
            ssms.append(ssm)

    return ssms


def plot_target_ssms(ax, ssms, target_label):
    for ssm in ssms:
        ax.plot(ssm[:, 0], ssm[:, 1], ssm[:, 2], color="black", linewidth=1.5)

    ax.set_title(target_label)
    ax.set_xlabel("theta 1 [rad]")
    ax.set_ylabel("theta 2 [rad]")
    ax.set_zlabel("theta 3 [rad]")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot self-motion manifolds for two planar 3R workspace points."
    )
    parser.add_argument("--x1", type=float, default=POINT_1[0], help="x coordinate of workspace point 1")
    parser.add_argument("--y1", type=float, default=POINT_1[1], help="y coordinate of workspace point 1")
    parser.add_argument("--x2", type=float, default=POINT_2[0], help="x coordinate of workspace point 2")
    parser.add_argument("--y2", type=float, default=POINT_2[1], help="y coordinate of workspace point 2")
    parser.add_argument("--seed", type=int, default=0, help="random seed for IK initial guesses")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = [(args.x1, args.y1), (args.x2, args.y2)]

    all_ssms = []
    for target_index, (x, y) in enumerate(targets):
        ssms = compute_ssms_for_target(x, y, seed=args.seed + target_index)
        all_ssms.append(ssms)
        print(f"target {target_index + 1}: ({x:.6f}, {y:.6f}) -> {len(ssms)} SSM branch(es)")
        for ssm_index, ssm in enumerate(ssms):
            print(f"  SSM {ssm_index + 1}: {len(ssm)} sampled configurations")

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    plot_target_ssms(ax1, all_ssms[0], f"Target 1: ({args.x1:.3f}, {args.y1:.3f})")
    plot_target_ssms(ax2, all_ssms[1], f"Target 2: ({args.x2:.3f}, {args.y2:.3f})")

    fig.suptitle("Planar 3R Self-Motion Manifolds")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
