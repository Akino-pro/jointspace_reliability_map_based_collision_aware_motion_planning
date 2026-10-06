import time

import numpy as np
from matplotlib import patches
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import null_space
import matplotlib.pyplot as plt

from helper_functions import normalize_and_map_colors, sorted_indices, forward_kinematics_3R, Jacobian_3R, \
    reliability_computation

step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 20
max_ssm = 2
sample_num = 128


#joint reliabilities
r1 = 0.9
r2 = 0.9
r3 = 0.9 #0.5


r1=2.0/3.0
r2=2.0/3.0
r3=2.0/3.0


#3r global rcmax parameters
#L=[1.4730585535218415, 0.0012204516043387876, 1.5257209948738197]
#CA=[[-2.923316619542279, 2.923316619542279], [-2.259564909688944, 2.472685303382877], [-0.19861917260970885, 2.273666386265002]]

#3r global ftwmax parameters
#L=[0.5,1.25,1.25]
#CA=[[-np.pi, np.pi], [-53.1301*np.pi/180.0, 126.8698*np.pi/180.0], [106.0602*np.pi/180.0, 106.2602*np.pi/180.0]]

#3r local ftwmax parameters
#L=[1, 1, 1]
#CA=[[-18.21*np.pi/180.0,18.21*np.pi/180.0], [-111.34*np.pi/180.0, 111.34*np.pi/180.0], [-111.34*np.pi/180.0, 111.34*np.pi/180.0]]


#3r local rcmax parameters
#L=[1, 1, 1]
#CA=[[-3.10567309825358, 3.10567309825358], [-2.2823870365553507, 2.723116465162326], [-0.38499180251357207, 0.49457526971108656]]



L=[0.4454,0.3143,0.2553]
CA=[(-180 * np.pi / 180, 180 * np.pi / 180), (-147.8* np.pi / 180, 147.8 * np.pi / 180),(-120.3 * np.pi / 180, 120.3 * np.pi / 180)]


#CA=[[-0.28571601349497305, 0.28571601349497305], [-1.3124126752173293, 1.1709575531137366], [-1.9488681870803908, 1.9485564228143248]]  #ftwmax
#CA=[[-3.0956884245183267, 3.0956884245183267], [-0.3125298209691351, 1.3513169007407377], [-0.5259633111766422, 1.6359794608628453]]#rcmax






C_dot_A = CA.copy()
C_dot_A[0] = (-np.pi, np.pi)



cr_list = reliability_computation(r1, r2, r3)
#print(cr_list)
indices = sorted_indices(cr_list)
cr_list.append(r1*r2*r3/(r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3))
print(cr_list)
print(indices)



def stepwise_ssm(theta, n_j, x_target, previous_n_j):
    a = np.array(n_j).reshape(-1)  # Converts 3x1 to 1D array
    b = np.array(previous_n_j).reshape(-1)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    if np.abs(angle_rad) > np.pi / 2:
        n_j = -n_j

    x = forward_kinematics_3R(theta, L)
    delta_x_step = x_target - x
    J_step = Jacobian_3R(theta, L)
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + step_size * dq
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    new_J = Jacobian_3R(theta_next, L)

    new_n_j = null_space(new_J)
    return theta_next, new_n_j, n_j


def find_ranges(nums):
    # Sort the input list
    nums = sorted(set(nums))
    ranges = []
    start = nums[0]

    for i_fr in range(1, len(nums)):
        # If the current number is not consecutive
        if nums[i_fr] != nums[i_fr - 1] + 1:
            ranges.append([start, nums[i_fr - 1]])
            start = nums[i_fr]

    # Append the last range
    ranges.append([start, nums[-1]])
    return ranges


def extend_ranges(union_ranges):
    extended_ranges = []
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in union_ranges:
        if tr[0] == -np.pi and tr[1] == np.pi:
            return [[-np.pi,
                     np.pi]]
        elif tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        elif tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        else:
            extended_ranges.append(tr)

    if has_negative_pi_range is not None and has_positive_pi_range is not None:
        extended_ranges.append([has_positive_pi_range - 2 * np.pi, has_negative_pi_range])
        extended_ranges.append([has_positive_pi_range, has_negative_pi_range + 2 * np.pi])
    elif has_negative_pi_range is not None:
        extended_ranges.append([-np.pi, has_negative_pi_range])
    elif has_positive_pi_range is not None:
        extended_ranges.append([has_positive_pi_range, np.pi])
    return extended_ranges


def find_intersection_points(ssm_theta_list):
    tof = False
    counter = 0
    ip_ranges = []
    isin_list = []
    for theta_index in range(len(ssm_theta_list)):
        theta = ssm_theta_list[theta_index]
        theta_flatten = theta.flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, C_dot_A)):
            isin_list.append(theta_index)
            tof = True
            counter += 1
    ip_index_ranges = []
    if len(isin_list) != 0: ip_index_ranges = find_ranges(isin_list)

    for index_range in ip_index_ranges:
        local_min_theta1 = 100
        local_max_theta1 = -100
        for i in range(index_range[0], index_range[1] + 1):
            if ssm_theta_list[i][0][0] < local_min_theta1:
                local_min_theta1 = ssm_theta_list[i][0][0]
            if ssm_theta_list[i][0][0] > local_max_theta1:
                local_max_theta1 = ssm_theta_list[i][0][0]
        if np.abs(local_min_theta1 + np.pi) <= 1e-3: local_min_theta1 = -np.pi
        if np.abs(local_min_theta1 - np.pi) <= 1e-3: local_min_theta1 = np.pi
        if np.abs(local_max_theta1 + np.pi) <= 1e-3: local_max_theta1 = -np.pi
        if np.abs(local_max_theta1 - np.pi) <= 1e-3: local_max_theta1 = np.pi
        ip_ranges.append([local_min_theta1, local_max_theta1])
    return ip_ranges, tof


def find_critical_points(ssm_theta_list):
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        previous_theta = ssm_theta_list[0]
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(3):
                if np.abs(theta[i][0] - previous_theta[i][0]) > 3:
                    if theta[i][0] > previous_theta[i][0]:
                        thetas_cp_sum[i].append([-np.pi, thetas_cp_range[i][1]])
                        thetas_cp_range[i] = [theta[i][0], np.pi]
                    else:
                        thetas_cp_sum[i].append([thetas_cp_range[i][0], np.pi])
                        thetas_cp_range[i] = [-np.pi, theta[i][0]]
                elif theta[i][0] < thetas_cp_range[i][0]:
                    thetas_cp_range[i][0] = theta[i][0]
                elif theta[i][0] > thetas_cp_range[i][1]:
                    thetas_cp_range[i][1] = theta[i][0]
            previous_theta = theta
    for j in range(3):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        # thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
    return thetas_cp_sum
def find_single_intersection(ssm_theta_list):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof
def ik_2r_single(x, y, L1, L2, base=(0.0, 0.0)):
    """
    Inverse kinematics for a 2R planar robot with arbitrary base point.
    Returns a single solution in radians.

    Args:
        x, y       : target end-effector coordinates (global frame)
        L1, L2     : link lengths
        prev_angle : previous known joint angle (radians)
        joint_index: 1 for joint1, 2 for joint2 (which joint is known)
        base       : (x0, y0) base coordinates of the robot in global frame

    Returns:
        (theta1, theta2) in radians
    """
    # Shift target into base-centered frame
    x0, y0 = base
    X = x - x0
    Y = y - y0

    r = np.hypot(X, Y)
    if r > L1 + L2:
        print('too far')
        raise ValueError("Target out of reach")
    if r < abs(L1 - L2):
        print('too close')
        raise ValueError("Target out of reach")

    # cos(theta2), clamp for safety
    c2 = (X*X + Y*Y - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = np.clip(c2, -1.0, 1.0)

    # elbow up and down
    s2_pos = np.sqrt(1.0 - c2*c2)
    s2_neg = -s2_pos

    def solve(s2):
        th2 = np.arctan2(s2, c2)
        th1 = np.arctan2(Y, X) - np.arctan2(L2*s2, L1 + L2*c2)
        return th1, th2

    sol_up = solve(s2_pos)
    sol_dn = solve(s2_neg)

    return sol_up, sol_dn
def find_random_ssm(x_target, all_ssm_theta_list):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(3,))).T.reshape((3, 1))
    step_num = 0
    error_threshold = 1e-6
    error = 10
    ik = False
    while error > error_threshold:
        x_current = forward_kinematics_3R(q, L)
        delta_x = x_target - x_current
        error = np.linalg.norm(x_target - x_current)
        J = Jacobian_3R(q, L)
        correction = J.T @ np.linalg.inv(J @ J.T + 0.5 ** 2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9: return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
        q = q + 0.1 * correction

        for i in range(len(q)):
            q[i] %= 2 * np.pi
            if q[i] > np.pi: q[i] -= 2 * np.pi
            if q[i] < -np.pi: q[i] += 2 * np.pi

        step_num += 1

        previous_x = x_current
        x_current = forward_kinematics_3R(q, L)
        consecutive_delta_x = np.linalg.norm(previous_x - x_current)
        if consecutive_delta_x < 1e-8: return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    ik = True
    sol = q

    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold: return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
    theta = sol
    theta_prime = theta.copy()

    J = Jacobian_3R(theta, L)
    n_j = null_space(J)
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    highest_delta_theta = -1000.0
    while True:
        if threshold <= terminate_threshold and num > 4: break
        old_theta = theta.copy()
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j)
        threshold = np.linalg.norm(theta - theta_prime)
        d_theta = np.linalg.norm(theta - old_theta)
        if threshold < lowest: lowest = threshold
        if d_theta > highest_delta_theta: highest_delta_theta = d_theta
        if num == 100 and highest_delta_theta > 2 * step_size:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    print('ssm guided to wrong direction.')
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
        n_j = new_n_j
        if num == 15000 and lowest > terminate_threshold:
            print('jump to another ssm, reassign starting point')
            theta_prime = theta
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
            ssm_theta_list = [theta]
            num = -1
        num += 1
        ssm_theta_list.append(theta)
    all_ssm_theta_list.extend(ssm_theta_list)
    #print(f'found a new ssm with {num} points.')
    ssm_found = True

    ip_ranges, tof = find_intersection_points(ssm_theta_list)
    single_intersection_tf=find_single_intersection(ssm_theta_list)
    cp_ranges = find_critical_points(ssm_theta_list)
    return ik, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found, single_intersection_tf
def union_ranges(ranges):
    """
    Computes the union of a list of ranges, with special handling for ranges that wrap around
    [-π, π]. If such ranges are detected, they are replaced by modified ranges.

    Parameters:
        ranges (list of tuples): List of ranges, where each range is represented as (a, b).

    Returns:
        list of tuples: The merged union of ranges as a list of non-overlapping ranges.
    """
    if not ranges:
        return []

    # Optional: Validate input ranges
    for a, b in ranges:
        if a > b:
            raise ValueError(f"Invalid range: ({a}, {b}). Start must be <= end.")

    # Sort the ranges by their starting point (and by end if start points are the same)
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

    # Initialize the merged ranges with the first range
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1]:  # Overlap or adjacent
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged
def compute_beta_range(x, y,L,CA):
    F_list = [False] * (8)
    target_x = np.array([x, y]).T.reshape((2, 1))
    all_smm_beta_range = []
    reliable_beta_ranges = [[], [], [], [], [], [], [],[]] # the last element for prefailure workspace(task space projection of artificial joint limits)
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    find_count = 0
    ssm_found = 0
    single_intersection_tf_all=False
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf, single_intersection_tf = find_random_ssm(
            target_x, all_theta)
        single_intersection_tf_all=single_intersection_tf_all or single_intersection_tf
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)

        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            if beta0_um - beta0_lm >= 2 * np.pi:
                beta0_ranges.append([-np.pi, np.pi])
            elif beta0_lm < -np.pi:
                beta0_ranges.append([beta0_lm + 2 * np.pi, np.pi])
                beta0_ranges.append([-np.pi, beta0_um])
            elif beta0_um > np.pi:
                beta0_ranges.append([-np.pi, beta0_um - 2 * np.pi])
                beta0_ranges.append([beta0_lm, np.pi])
            else:
                beta0_ranges.append([beta0_lm, beta0_um])

        if ssm_found_tf:  ssm_found += 1; find_count = 0

        for cp_range in cp_ranges[0]:
            if cp_range[0] > -np.inf:
                theta1_ranges.append(cp_range)

        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf:
                theta2_ranges.append(cp_range)

        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf:
                theta3_ranges.append(cp_range)

    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)

    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            min_beta1 = CA[0][1] - tr[1]
            max_beta1 = CA[0][0] - tr[0]
            if (max_beta1 - min_beta1 >= 2 * np.pi) or (tr[0] == -np.pi and tr[1] == np.pi):
                max_beta1 = np.pi
                min_beta1 = -np.pi
    ion2 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta2_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[1][0] >= tr[0] and CA[1][1] <= tr[1]:
            ion2 = True
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
    min_beta2 = -np.pi
    max_beta2 = np.pi

    ion3 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta3_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[2][0] >= tr[0] and CA[2][1] <= tr[1]:
            ion3 = True
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
    min_beta3 = -np.pi
    max_beta3 = np.pi

    for index in range(len(beta0_ranges)):
        reliable_beta_ranges[-1].append(beta0_ranges[index])# prefailure workspace
        min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]

        if single_intersection_tf_all: F_list[-1] = True
        if ion1:
            if single_intersection_tf_all:F_list[0]=True
            min_beta_f_1 = max(min_beta0, min_beta1)
            max_beta_f_1 = min(max_beta0, max_beta1)
            if min_beta_f_1 <= max_beta_f_1:
                reliable_beta_ranges[0].append([min_beta_f_1, max_beta_f_1])
        if ion2:
            if single_intersection_tf_all:F_list[1] = True
            min_beta_f_2 = max(min_beta0, min_beta2)
            max_beta_f_2 = min(max_beta0, max_beta2)
            if min_beta_f_2 <= max_beta_f_2:

                reliable_beta_ranges[1].append([min_beta_f_2, max_beta_f_2])
        if ion3:
            if single_intersection_tf_all:F_list[2] = True
            min_beta_f_3 = max(min_beta0, min_beta3)
            max_beta_f_3 = min(max_beta0, max_beta3)
            if min_beta_f_3 <= max_beta_f_3:

                reliable_beta_ranges[2].append([min_beta_f_3, max_beta_f_3])
        if ion1 and ion2:
            if single_intersection_tf_all:F_list[3] = True
            min_beta_f_12 = max(min_beta0, min_beta1, min_beta2)
            max_beta_f_12 = min(max_beta0, max_beta1, max_beta2)
            if min_beta_f_12 <= max_beta_f_12:

                reliable_beta_ranges[3].append([min_beta_f_12, max_beta_f_12])
        if ion1 and ion3:
            if single_intersection_tf_all:F_list[4] = True
            min_beta_f_13 = max(min_beta0, min_beta1, min_beta3)
            max_beta_f_13 = min(max_beta0, max_beta1, max_beta3)
            if min_beta_f_13 <= max_beta_f_13:

                reliable_beta_ranges[4].append([min_beta_f_13, max_beta_f_13])
        if ion2 and ion3:
            if single_intersection_tf_all:F_list[5] = True
            min_beta_f_23 = max(min_beta0, min_beta2, min_beta3)
            max_beta_f_23 = min(max_beta0, max_beta2, max_beta3)
            if min_beta_f_23 <= max_beta_f_23:

                reliable_beta_ranges[5].append([min_beta_f_23, max_beta_f_23])
        if ion1 and ion2 and ion3:
            if single_intersection_tf_all:F_list[6] = True
            min_beta_f_ftw = max(min_beta0, min_beta1, min_beta2, min_beta3)
            max_beta_f_ftw = min(max_beta0, max_beta1, max_beta2, max_beta3)

            if min_beta_f_ftw <= max_beta_f_ftw:

                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
                reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])

    for rbr_index in range(len(reliable_beta_ranges)):
        reliable_beta_ranges[rbr_index] = union_ranges(reliable_beta_ranges[rbr_index])
    return union_ranges(all_smm_beta_range), reliable_beta_ranges,F_list, all_theta


def plot_all_theta_3d(all_theta, x, y):
    if not all_theta:
        print(f"No all_theta points to plot for x={x}, y={y}")
        return

    theta_points = np.asarray(all_theta, dtype=float).reshape(len(all_theta), 3)

    fig_theta = plt.figure()
    ax_theta = fig_theta.add_subplot(111, projection='3d')
    ax_theta.scatter(
        theta_points[:, 0],
        theta_points[:, 1],
        theta_points[:, 2],
        s=8,
        alpha=0.8,
    )
    ax_theta.set_title(f"all_theta at x={x:.4f}, y={y:.4f}")
    ax_theta.set_xlabel("theta 1")
    ax_theta.set_ylabel("theta 2")
    ax_theta.set_zlabel("theta 3")


def wedge_to_poly3d(wedge, z_value):
    theta1, theta2 = np.radians(wedge.theta1), np.radians(wedge.theta2)
    outer_radius = wedge.r
    inner_radius = wedge.r - wedge.width

    # Outer arc coordinates
    outer_arc_x = outer_radius * np.cos(np.linspace(theta1, theta2, 100))
    outer_arc_y = outer_radius * np.sin(np.linspace(theta1, theta2, 100))

    # Inner arc coordinates
    inner_arc_x = inner_radius * np.cos(np.linspace(theta2, theta1, 100))
    inner_arc_y = inner_radius * np.sin(np.linspace(theta2, theta1, 100))

    # Combine outer and inner arcs to form a closed polygon
    x = np.concatenate([outer_arc_x, inner_arc_x])
    y = np.concatenate([outer_arc_y, inner_arc_y])
    z = np.full_like(x, z_value)

    # Create a list of (x, y, z) tuples
    vertices = list(zip(x, y, z))
    return vertices

def main_function():

    cr_list.append(0)
    color_list, sm = normalize_and_map_colors(cr_list)
    final_wedges = []
    final_colors = []

    section_length = np.sum(L) / sample_num
    x_values = (np.arange(sample_num) + 0.5) * section_length
    y_values = np.zeros(sample_num)
    points = np.column_stack((x_values, y_values))

    ring_width = 2.0*x_values[0]
    
    
    fig3, ax2d3 = plt.subplots(figsize=(6, 6))
    ax2d3.set_xlim(-np.sum(L), np.sum(L))
    ax2d3.set_ylim(-np.sum(L),np.sum(L))
    ax2d3.set_aspect('equal')
    ax2d3.set_xlabel("x", fontsize=25)
    ax2d3.set_ylabel("y", fontsize=25)
    tick_positions = [-np.sum(L)/3.0*2.0,-np.sum(L)/3.0, 0, np.sum(L)/3.0,np.sum(L)/3.0*2.0]
    ax2d3.set_xticks(tick_positions)
    ax2d3.set_yticks(tick_positions)
    circle = patches.Circle((0, 0), np.sum(L), edgecolor=color_list[-1], facecolor=color_list[-1], linewidth=0, zorder=0)
    ax2d3.add_patch(circle)
    #ax2d3.axhline(0, color='black', linewidth=1, zorder=1000)  # y=0
    #ax2d3.axvline(0, color='black', linewidth=1, zorder=1000)  # x=0
    """
    x_ticks = np.linspace(0, 3, sample_num + 1)
    tick_length = 0.05
    for x in x_ticks:
        ax2d3.plot([x, x], [0, tick_length], color='black', linewidth=1, zorder=1000)
    """
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    start = time.perf_counter()
    for i in range(len(points)):
        point = points[i]
        x, y = point

        print(point)
        beta_ranges, reliable_beta_ranges,F_list, all_theta = compute_beta_range(x, y,L,CA)  # Get multiple beta ranges
        #if x >= 0.37 and x <= 0.4:
            #print(x, y)
            #plot_all_theta_3d(all_theta, x, y)


        level=0
        for b_r_index in indices:
            level+=1
            b_r = reliable_beta_ranges[b_r_index]
            z_level = b_r_index * 2 + 2
            color = color_list[b_r_index]

            for beta_range in b_r:
                #""" original approach
                # Compute angles in degrees (as required by Wedge)
                theta1 = np.degrees(beta_range[0])  # Start angle (-π)
                theta2 = np.degrees(beta_range[1])  # End angle (π)
    
                # Calculate the outer radius for the ring (x + ring_width / 2)
                outer_radius = x + ring_width / 2.0
                inner_radius = x-ring_width/2.0
    
                wedge = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1, theta2=theta2,
                    width=ring_width,
                    facecolor=color,  # Set face color to match 3D plot
                    edgecolor=color,
                    alpha=1.0,
                    zorder=level
                )
    
                wedge_2d = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1, theta2=theta2,
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0,
                    zorder=level
                )
                
                poly3d = wedge_to_poly3d(wedge, z_level)
                # Add the wedge to the plot
                # ax.add_patch(wedge)
                ax.add_collection3d(
                    Poly3DCollection(
                        [poly3d],
                        facecolor=color,
                        edgecolor=color,
                        alpha=1.0
                    )
                )
    
                ax2d3.add_patch(wedge_2d)  # Add wedge to 2D plot
    
                if b_r_index == len(reliable_beta_ranges) - 1:
                    final_wedges.append(wedge)
                    final_colors.append(color)
    

        prefailure_beta_range=reliable_beta_ranges[-1]
        color=color_list[-2]
        for beta_range in prefailure_beta_range:
            # Compute angles in degrees (as required by Wedge)
            theta1 = np.degrees(beta_range[0])  # Start angle (-π)
            theta2 = np.degrees(beta_range[1])  # End angle (π)

            # Calculate the outer radius for the ring (x + ring_width / 2)
            outer_radius = x + ring_width / 2.0
            inner_radius = x - ring_width / 2.0

            wedge = Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=theta1, theta2=theta2,
                width=ring_width,
                facecolor=color,  # Set face color to match 3D plot
                edgecolor=color,
                alpha=1.0,
                zorder=0
            )

            wedge_2d = Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=theta1, theta2=theta2,
                width=ring_width,
                facecolor=color,
                edgecolor=color,
                alpha=1.0,
                zorder=0
            )


            ax2d3.add_patch(wedge_2d)  # Add wedge to 2D plot

            if b_r_index == len(reliable_beta_ranges) - 1:
                final_wedges.append(wedge)
                final_colors.append(color)

            poly3d = wedge_to_poly3d(wedge, 0)
            ax.add_collection3d(
                Poly3DCollection(
                    [poly3d],
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0
                )
            )

    end = time.perf_counter()
    print(f"Loop took {end - start:.6f} seconds")

    ax2d3.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax2d3.tick_params(axis='y', labelsize=18)  # Increase font size for Y-axis ticks
    fig3.show()

    ax.set_xlim(-np.sum(L), np.sum(L))
    ax.set_ylim(-np.sum(L), np.sum(L))
    ax.set_zlim(0, 14)
    ax.set_xlabel("x", fontsize=25)
    ax.set_ylabel("y", fontsize=25)
    tick_positions = [-np.sum(L)/3.0*2.0, 0, np.sum(L)/3.0*2.0]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_aspect('equal')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    cbar = plt.colorbar(sm, ax=ax)  # Ensure colorbar is linked to the mappable
    tick_positions = np.arange(0.1, 1.1, 0.1)  # 1.1 ensures 1.0 is included
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in tick_positions])
    # Define the new labels and corresponding tick positionsq
    z_tick_labels = [r'$\mathit{F}=\emptyset$', r'$\mathit{F}=\{1\}$', r'$\mathit{F}=\{2\}$', r'$\mathit{F}=\{3\}$',
                     r'$\mathit{F}=\{1,2\}$', r'$\mathit{F}=\{1,3\}$', r'$\mathit{F}=\{2,3\}$',
                     r'$\mathit{F}=\{1,2,3\}$']
    ax.tick_params(axis='z', labelsize=18)
    z_tick_positions = [0, 2, 4, 6, 8, 10, 12, 14]  # These match your z_level values
    
    # Set the new labels on the Z-axis
    ax.set_zticks(z_tick_positions)
    ax.set_zticklabels(z_tick_labels)
    plt.show()

#main_function()



