import time

import numpy as np
from tqdm import tqdm
from scipy.linalg import null_space
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3

from scripts.helper_functions import normalize_and_map_colors, sorted_indices, union_ranges, \
    update_or_add_square_2d, update_beta_bar_multicolor, key3, set_sparse_xyz_labels, \
    set_axes_equal,generate_grid_centers, generate_square_grid, draw_rotated_grid, generate_2D_square_grid

proposed_sample_num=162
MIN_3D_PLOT_RADIUS = 3.1989682240512938

#ssm searching constant
step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 20
max_ssm = 4


#joint reliabilities
r1 = 0.9
r2 = 0.7
r3 = 0.5
r4 = 0.3

#dh parameters

#ftwmax
alpha = [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180]
d = [-0.29, 0, 0.05, 1]
l = [0.5, 0.48, 0.76, 0.95]
CA=[[-146*np.pi/180.0, 146*np.pi/180.0], [-234*np.pi/180.0, 10*np.pi/180.0], [-115*np.pi/180.0, 132*np.pi/180.0], [-101*np.pi/180.0,118*np.pi/180.0]]

#rcmax
CA=  [[-2.7873328856440396, 3.1363905327568893], [-2.4637381023154603, 2.136018018712809], [-1.8292591098878526, 2.0339314472150742], [-1.6142928309703712, 1.7054258649412979]]
alpha =[1.1599121046989387, -0.5154988273227061, 0.03775362985949342, 0.5365523262566805]
l= [0.04732540449946272, 0.8501471809443848, 0.0018628495298751034, 0.8495872435309159]
d =  [-0.0020589855582313892, -0.12359280272943952, 0.0025723721804752, 0.04538990560835715]



def reliability_computation(r1, r2, r3, r4):
    reliability_list = [r2 * r3 * r4, r1 * r3 * r4, r1 * r2 * r4, r1 * r2 * r3,
                        r1 * r3 * r4 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r1 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4]
    # reliability_list = [r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4]
    conditional_reliability_list = []
    for p in reliability_list:
        conditional_reliability_list.append(
            p /
            (r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4))
    return conditional_reliability_list


cr_list = reliability_computation(r1, r2, r3, r4)
print(cr_list)
print(sorted_indices(cr_list))

def convert_to_C_dot_A(CA):
    """
    Converts a list of ranges CA to C_dot_A by handling wrap-around cases around [-π, π].

    Parameters:
        CA (list of tuples): List of ranges, each as (lower, upper).

    Returns:
        list of lists of tuples: C_dot_A with each original range and any additional ranges to handle wrap-around.
    """
    C_dot_A = []

    for lower, upper in CA:
        # Initialize the sublist for the current range
        sublist = [(lower, upper)]

        # Check if the lower limit is below -π
        if lower < -np.pi:
            # Add [lower + 2π, upper + 2π] to handle wrap-around
            sublist.append((lower + 2 * np.pi, upper + 2 * np.pi))

        # Check if the upper limit is above π
        if upper > np.pi:
            # Add [lower - 2π, upper - 2π] to handle wrap-around
            sublist.append((lower - 2 * np.pi, upper - 2 * np.pi))

        # Append the sublist to C_dot_A
        C_dot_A.append(sublist)

    return C_dot_A
def convert_plot_to_voxel_matrix(ax, grid_size=100, plot_range=(-4, 4)):
    """
    Convert a Matplotlib 3D plot to a binary 3D matrix (voxel grid) by checking if each voxel
    overlaps with the plot data in ax.

    Parameters:
    ax : Matplotlib 3D axis object that contains the 3D plot data.
    grid_size : The size of the 3D grid (number of voxels along each axis), default is 100x100x100.
    plot_range : Tuple representing the range of the plot along each axis.

    Returns:
    voxel_matrix : A binary 3D matrix (voxel grid) representing the 3D plot in ax.
    """
    # Initialize an empty voxel matrix with the given size (0 for empty, 1 for filled)
    voxel_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Define the resolution for each axis based on plot range and grid size
    plot_min, plot_max = plot_range
    voxel_resolution = (plot_max - plot_min) / grid_size

    # Generate the center coordinates for each voxel in the grid
    x_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2
    y_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2
    z_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2

    # Check each voxel position to see if it intersects with plot elements in ax
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                # Check if the voxel at (x, y, z) overlaps with any of the plot elements
                # ax contains function can check if a point lies within plotted data (example only)
                # Note: for precise checks, you'd need to develop custom intersection logic
                point_in_plot = any(collection.contains_point([x, y, z]) for collection in ax.collections)

                # Update the voxel matrix
                if point_in_plot:
                    voxel_matrix[i, j, k] = 1

    return voxel_matrix
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


def square_center_xz(square):
    square_np = np.asarray(square)
    return float(np.mean(square_np[:, 0])), float(np.mean(square_np[:, 2]))


def subtract_angle_ranges(base_ranges, blocked_ranges):
    remaining = [list(r) for r in base_ranges]
    for blocked_lo, blocked_hi in union_ranges(blocked_ranges):
        next_remaining = []
        for lo, hi in remaining:
            if blocked_hi <= lo or blocked_lo >= hi:
                next_remaining.append([lo, hi])
                continue
            if lo < blocked_lo:
                next_remaining.append([lo, blocked_lo])
            if blocked_hi < hi:
                next_remaining.append([blocked_hi, hi])
        remaining = next_remaining
        if not remaining:
            break
    return remaining


def draw_zero_reliability_background(ax, grid_squares, reference_beta_ranges, radius, zero_color):
    full_beta_range = [[-np.pi, np.pi]]
    radius_sq = radius * radius
    for i, square in tqdm(enumerate(grid_squares), total=len(grid_squares), desc="Drawing zero reliability"):
        center_x, center_z = square_center_xz(square)
        if center_x * center_x + center_z * center_z > radius_sq:
            continue
        zero_beta_ranges = subtract_angle_ranges(full_beta_range, reference_beta_ranges[i])
        for beta_range in zero_beta_ranges:
            draw_rotated_grid(ax, square, beta_range, zero_color)


def draw_zero_reliability_radius_extension(ax, n_x, n_z, inner_radius, outer_radius, zero_color):
    if outer_radius <= inner_radius:
        return

    scale = outer_radius / inner_radius if inner_radius > 1e-12 else 1.0
    ext_n_x = max(n_x, int(np.ceil(n_x * scale)))
    ext_n_z = max(n_z, int(np.ceil(n_z * scale)))
    extension_squares = generate_square_grid(
        ext_n_x,
        ext_n_z,
        (0, outer_radius),
        (-outer_radius, outer_radius),
    )
    inner_radius_sq = inner_radius * inner_radius
    outer_radius_sq = outer_radius * outer_radius

    for square in tqdm(extension_squares, desc="Drawing zero reliability radius extension"):
        center_x, center_z = square_center_xz(square)
        center_radius_sq = center_x * center_x + center_z * center_z
        if inner_radius_sq < center_radius_sq <= outer_radius_sq:
            draw_rotated_grid(ax, square, [-np.pi, np.pi], zero_color)


def stepwise_ssm(theta, n_j, x_target, previous_n_j, robot):
    x = robot.fkine(theta.flatten())
    x = np.array(x)[:3, 3].T.reshape((3, 1))
    delta_x_step = x_target - x
    J_step = robot.jacob0(theta.flatten())[:3, :]
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + dq * step_size
    # x_next = robot.fkine(theta_next.flatten())
    # print(x_next - x_target)
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    new_J = robot.jacob0(theta_next.flatten())[:3, :]
    new_n_j = null_space(new_J)
    m, n = np.shape(new_n_j)
    if n == 2:
        print('special case')
        n_j_1 = new_n_j[:, 0]
        n_j_2 = new_n_j[:, 1]
        if np.dot(n_j_1, previous_n_j) > np.dot(n_j_2, previous_n_j):
            new_n_j = n_j_1
        else:
            new_n_j = n_j_2
    a = np.array(new_n_j).reshape(-1)  # Converts 4x1 to 1D array
    a_prime = np.array(-new_n_j).reshape(-1)
    b = np.array(previous_n_j).reshape(-1)
    dot_product = np.dot(a, b)
    dot_product_prime = np.dot(a_prime, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_a_prime = np.linalg.norm(a_prime)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta_prime = dot_product_prime / (norm_a_prime * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    cos_theta_prime = np.clip(cos_theta_prime, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_rad_prime = np.arccos(cos_theta_prime)
    if np.abs(angle_rad) > np.abs(angle_rad_prime):
        new_n_j = -new_n_j

    return theta_next, new_n_j, n_j
def find_intersection_points(ssm_theta_list, C_dot_A):
    tof = False
    counter = 0
    ip_ranges = []
    isin_list = []
    for theta_index in range(len(ssm_theta_list)):
        theta = ssm_theta_list[theta_index]
        theta_flatten = theta.flatten()
        if all(any(r[0] <= v <= r[1] for r in ranges) for v, ranges in zip(theta_flatten, C_dot_A)):
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
        if np.abs(local_min_theta1 + np.pi) <= 1e-2: local_min_theta1 = -np.pi
        if np.abs(local_min_theta1 - np.pi) <= 1e-2: local_min_theta1 = np.pi
        if np.abs(local_max_theta1 + np.pi) <= 1e-2: local_max_theta1 = -np.pi
        if np.abs(local_max_theta1 - np.pi) <= 1e-2: local_max_theta1 = np.pi
        ip_ranges.append([local_min_theta1, local_max_theta1])
        # print([local_min_theta1, local_max_theta1])
    return union_ranges(ip_ranges), tof
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
def extend_ranges(unioned_ranges):
    extended_ranges = []
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in unioned_ranges:
        if tr[0] == -np.pi and tr[1] == np.pi:
            return [[- np.pi, np.pi]]
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
def find_critical_points(ssm_theta_list):
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
                , [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        thetas_cp_range[3] = [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
        previous_theta = ssm_theta_list[0]
        ssm_theta_list.append(previous_theta)
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(4):
                #if i==3: print(theta[i][0])
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
    for j in range(4):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
        #if j == 0: print(thetas_cp_sum[j])
    # print(thetas_cp_sum)
    return thetas_cp_sum
def find_single_intersection(ssm_theta_list):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof
def find_random_ssm(x_target, all_ssm_theta_list, robot, C_dot_A):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(4,))).T.reshape((4, 1))
    Tep = SE3(x_target)
    result = robot.ikine_LM(Tep, mask=[1, 1, 1, 0, 0, 0], q0=q)
    if not result.success: return result.success, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    sol = result.q.T.reshape((4, 1))
    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    theta = sol
    theta_prime = theta.copy()

    J = robot.jacob0(theta.flatten())[:3, :]
    n_j = null_space(J)[:, 0].T.reshape((4, 1))
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    while True:
        num += 1
        if threshold <= terminate_threshold and num >= 4: break
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j, robot)
        threshold = np.linalg.norm(theta - theta_prime)
        # print(threshold)
        if threshold < lowest: lowest = threshold

        if num == 100:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
        n_j = new_n_j
        if num > 5000:
            print('stuck at a too small smm')
            theta_prime = theta
            num = 0
        ssm_theta_list.append(theta)

    """
    points = np.array(ssm_theta_list)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')

        # Set labels
    plt.xlabel('theta1')
    plt.ylabel('theta2')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 1], points[:, 2], c='b', marker='o')

        # Set labels
    plt.xlabel('theta2')
    plt.ylabel('theta3')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 2], points[:, 3], c='b', marker='o')

        # Set labels
    plt.xlabel('theta3')
    plt.ylabel('theta4')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 3], points[:, 0], c='b', marker='o')

        # Set labels
    plt.xlabel('theta4')
    plt.ylabel('theta1')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()
    """


    all_ssm_theta_list.extend(ssm_theta_list)
    # print(f'found a new ssm with {num} points.')
    ssm_found = True

    ip_ranges, tof = find_intersection_points(ssm_theta_list, C_dot_A)
    cp_ranges = find_critical_points(ssm_theta_list)
    single_intersection_tf = find_single_intersection(ssm_theta_list)
    return True, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found,single_intersection_tf
def compute_beta_range(x, y, z, robot, C_dot_A, CA):
    target_x = np.array([x, y, z]).T.reshape((3, 1))
    all_smm_beta_range = []
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    find_count = 0
    ssm_found = 0
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf = find_random_ssm(
            target_x, all_theta, robot, C_dot_A)
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            # print(beta0_lm, beta0_um)
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
            if cp_range[0] > -np.inf: theta1_ranges.append(cp_range)
        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf: theta2_ranges.append(cp_range)
        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf: theta3_ranges.append(cp_range)
        for cp_range in cp_ranges[3]:
            if cp_range[0] > -np.inf: theta4_ranges.append(cp_range)

    # print(beta0_ranges)
    if len(theta1_ranges) == 0: return []
    if len(theta2_ranges) == 0: return []
    if len(theta3_ranges) == 0: return []
    if len(theta4_ranges) == 0: return []
    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)

    # print(theta1_ranges_union)
    # print(theta2_ranges_union)
    # print(theta3_ranges_union)
    # print(theta4_ranges_union)
    # print('hello')
    # print(extend_ranges(theta1_ranges_union))
    # print(extend_ranges(theta2_ranges_union))
    # print(extend_ranges(theta3_ranges_union))
    # print(extend_ranges(theta4_ranges_union))

    if len(all_theta) != 0:
        points = np.array(all_theta)

        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')

        # Set labels
        plt.xlabel('theta1')
        plt.ylabel('theta2')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 1], points[:, 2], c='b', marker='o')

        # Set labels
        plt.xlabel('theta2')
        plt.ylabel('theta3')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 2], points[:, 3], c='b', marker='o')

        # Set labels
        plt.xlabel('theta3')
        plt.ylabel('theta4')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 3], points[:, 0], c='b', marker='o')

        # Set labels
        plt.xlabel('theta4')
        plt.ylabel('theta1')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            # print('joint 1 succeed')
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
            # print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')

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
            # print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')

    ion4 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta4_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[3][0] >= tr[0] and CA[3][1] <= tr[1]:
            ion4 = True
            # print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')

    # for index in range(len(beta0_ranges)):
    #    min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
    #    min_beta_f_ftw = -np.pi
    #   max_beta_f_ftw = np.pi
    #    all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
    if ion1 and ion2 and ion3 and ion4:
        for index in range(len(beta0_ranges)):
            min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
            # print(min_beta0, max_beta0)
            # print(min_beta1, max_beta1)
            # print(min_beta2, max_beta2)
            # print(min_beta3, max_beta3)
            # print(min_beta3, max_beta4)

            min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
            max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
            if min_beta_f_ftw <= max_beta_f_ftw:
                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
            # reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])

    # for rbr_index in range(len(reliable_beta_ranges)):
    #    reliable_beta_ranges[rbr_index] = union_ranges(reliable_beta_ranges[rbr_index])
    # print(all_smm_beta_range)
    return union_ranges(all_smm_beta_range)
def compute_reliable_beta_range(x, y, z, robot, C_dot_A, CA, all_reliable_beta_ranges):
    # 15 cases
    reliable_beta_ranges = [[] for _ in range(16)]
    target_x = np.array([x, y, z]).T.reshape((3, 1))
    F_list=[False] * 16
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    find_count = 0
    ssm_found = 0
    single_intersection_tf_all = False
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf,single_intersection_tf = find_random_ssm(
            target_x, all_theta, robot, C_dot_A)
        single_intersection_tf_all = single_intersection_tf_all or single_intersection_tf
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            # print(beta0_lm, beta0_um)
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
            if cp_range[0] > -np.inf: theta1_ranges.append(cp_range)
        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf: theta2_ranges.append(cp_range)
        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf: theta3_ranges.append(cp_range)
        for cp_range in cp_ranges[3]:
            if cp_range[0] > -np.inf: theta4_ranges.append(cp_range)

    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)
    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)

    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            # print('joint 1 succeed')
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
            # print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')

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
            # print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')

    ion4 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta4_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[3][0] >= tr[0] and CA[3][1] <= tr[1]:
            ion4 = True
            # print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if single_intersection_tf_all:F_list[-1] = True
    if ion1:
        if single_intersection_tf_all: F_list[0] = True
    if ion1 and ion2:
        if single_intersection_tf_all: F_list[4] = True
    if ion1 and ion3:
        if single_intersection_tf_all: F_list[5] = True
    if ion1 and ion4:
        if single_intersection_tf_all: F_list[6] = True
    if ion1 and ion2 and ion3:
        if single_intersection_tf_all: F_list[10] = True
    if ion1 and ion2 and ion4:
        if single_intersection_tf_all: F_list[11] = True
    if ion1 and ion3 and ion4:
        if single_intersection_tf_all: F_list[12] = True
    if ion1 and ion2 and ion3 and ion4:
        if single_intersection_tf_all: F_list[14] = True
    if ion2:
        if single_intersection_tf_all: F_list[1] = True
    if ion3:
        if single_intersection_tf_all: F_list[2] = True
    if ion4:
        if single_intersection_tf_all: F_list[3] = True
    if ion2 and ion3:
        if single_intersection_tf_all: F_list[7] = True
    if ion2 and ion4:
        if single_intersection_tf_all: F_list[8] = True
    if ion3 and ion4:
        if single_intersection_tf_all: F_list[9] = True
    if ion2 and ion3 and ion4:
        if single_intersection_tf_all: F_list[13] = True


    for index in range(len(beta0_ranges)):
        reliable_beta_ranges[-1].append(beta0_ranges[index])
        min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
        min_beta_f_ftw_v1 = max(min_beta0, -np.pi)
        max_beta_f_ftw_v1 = min(max_beta0, np.pi)
        v1_valid = False
        valid = False
        if min_beta_f_ftw_v1 <= max_beta_f_ftw_v1: v1_valid = True
        min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
        max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
        if min_beta_f_ftw <= max_beta_f_ftw: valid = True
        #print(ion1,ion2,ion3,ion4)
        if valid:
            if ion1:

                reliable_beta_ranges[0].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2:

                reliable_beta_ranges[4].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion3:

                reliable_beta_ranges[5].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion4:

                reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion3:

                reliable_beta_ranges[10].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[0].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion4:

                reliable_beta_ranges[11].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[1].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion3 and ion4:

                reliable_beta_ranges[12].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[2].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion3 and ion4:

                reliable_beta_ranges[14].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[4].append([min_beta_f_ftw, max_beta_f_ftw])

        if v1_valid:
            if ion2:

                reliable_beta_ranges[1].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion3:

                reliable_beta_ranges[2].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion4:

                reliable_beta_ranges[3].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion3:

                reliable_beta_ranges[7].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion4:

                reliable_beta_ranges[8].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion3 and ion4:

                reliable_beta_ranges[9].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion3 and ion4:

                reliable_beta_ranges[13].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])

    for i in range(16):
        reliable_beta_ranges[i] = union_ranges(reliable_beta_ranges[i])
        all_reliable_beta_ranges[i].append(reliable_beta_ranges[i])

    return all_reliable_beta_ranges,F_list


"""
@measure_time
def ssm_estimation(grid_sample_num, d, alpha, l, CA):
    C_dot_A = CA.copy()
    C_dot_A[0] = (-np.pi, np.pi)
    C_dot_A = convert_to_C_dot_A(C_dot_A)
    robot = DHRobot(
        [
            RevoluteDH(d=d[0], alpha=alpha[0], a=l[0], qlim=CA[0]),
            RevoluteDH(d=d[1], alpha=alpha[1], a=l[1], qlim=CA[1]),
            RevoluteDH(d=d[2], alpha=alpha[2], a=l[2], qlim=CA[2]),
            RevoluteDH(d=d[3], alpha=alpha[3], a=l[3], qlim=CA[3]),
        ], name="spatial 4R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(4):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_size = (64, 64, 64)
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
    angle_ranges = []
    reachable_points = 0

    for center in tqdm(grid_centers , desc="Processing Items"):
        print(center)
        beta_ranges = compute_beta_range(center[0], center[1], center[2], robot, C_dot_A, CA)
        # print(beta_ranges)
        if len(beta_ranges) != 0: reachable_points += 1
        angle_ranges.append(beta_ranges)
    
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Generate grid of squares
    arc_color = 'blue'
    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Draw arcs for each square grid by rotating the entire grid square、
    for i, square in enumerate(grid_squares):
        for beta_range in angle_ranges[i]:
            draw_rotated_grid(ax, square, beta_range, arc_color)

    # Set plot labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    #plot sampled grids
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    # Draw squares only if the angle_ranges[i] is non-empty
    for i, square in enumerate(grid_squares):
        color = 'k'
        alpha_level = 0.3
        if angle_ranges[i]:  # Check if the list is non-empty
            alpha_level = 1.0
            # Plot the square grid directly
        square_poly = Poly3DCollection([square], color=color, alpha=alpha_level)
        ax.add_collection3d(square_poly)

    frame_points = [
        (x_range[0], 0, z_range[0]), (x_range[1], 0, z_range[0]),
        (x_range[1], 0, z_range[1]), (x_range[0], 0, z_range[1]),
        (x_range[0], 0, z_range[0])  # Closing the loop
    ]

    frame = Line3DCollection([frame_points], colors='k', linewidths=2)
    ax.add_collection3d(frame)

    # Set plot labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
        n_x, n_z, x_range, z_range, grid_size, angle_ranges
    )

    # grayscale_matrix = convert_plot_to_voxel_matrix(ax)
    shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_matrix,
                                                                                     kernel_size, Lambda)
    # print(shape_area)
    # print(connected_connectivity)
    print(general_connectivity)
    return general_connectivity
"""


# reliability
# @measure_time
def ssm_estimation(grid_sample_num, d, alpha, l, CA):
    C_dot_A = CA.copy()
    C_dot_A[0] = (-np.pi, np.pi)
    C_dot_A = convert_to_C_dot_A(C_dot_A)
    robot = DHRobot(
        [
            RevoluteDH(d=d[0], alpha=alpha[0], a=l[0], qlim=CA[0]),
            RevoluteDH(d=d[1], alpha=alpha[1], a=l[1], qlim=CA[1]),
            RevoluteDH(d=d[2], alpha=alpha[2], a=l[2], qlim=CA[2]),
            RevoluteDH(d=d[3], alpha=alpha[3], a=l[3], qlim=CA[3]),
        ], name="spatial 4R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(4):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    print(max_length)
    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_size = (64, 64, 64)
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)


    all_reliable_beta_ranges = [[] for _ in range(16)]
    indices = sorted_indices(cr_list)
    cr_list.append(r1*r2*r3*r4/(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4))
    color_list, sm = normalize_and_map_colors(cr_list)

    fig, ax2 = plt.subplots()
    ax2.set_xlim([0, max_length])
    ax2.set_ylim([-max_length, max_length])
    ax2.set_aspect(1)
    cbar = plt.colorbar(sm, ax=ax2, label='Reliability Spectrum')

    plt.ion()
    index_dict = {}

    start = time.perf_counter()
    twod_squares = generate_2D_square_grid(n_x, n_z, x_range, z_range)
    for i, center in tqdm(enumerate(grid_centers), total=len(grid_centers), desc="Processing grid centers"):
        # print(center)
        square = twod_squares[i]
        all_reliable_beta_ranges, F_list = compute_reliable_beta_range(
            center[0], center[1], center[2], robot, C_dot_A, CA, all_reliable_beta_ranges
        )

        last_true = None

        if F_list[-1]:
            last_true=-1
        for you in indices:  # indices is ascending
            if F_list[you]:
                last_true = you

        if last_true is not None:
            color = color_list[last_true]
            update_or_add_square_2d(ax2, square, color, 1.0, i, index_dict=index_dict)

    end = time.perf_counter()
    print(f"Loop took {end - start:.6f} seconds")


    with open("my_list.txt", "w") as file:
        file.write(str(all_reliable_beta_ranges))



    #plot beta ranges
    fig, ax = plt.subplots(figsize=(12, 4))

    sub_indices = [
        i for i in range(len(grid_centers))
        if any(len(all_reliable_beta_ranges[you][i]) != 0 for you in indices)
    ]
    sub_grid_centers = [grid_centers[i] for i in sub_indices]


    for i in range(len(grid_centers)):
        ax.bar(i, 0, width=0.6, color="white", edgecolor="purple")

    tick_positions = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=18)

    ax.set_ylabel("Failure Tolerant Rotation Angles of β", fontsize=27)
    ax.set_xlabel("x, z Task Space Locations", fontsize=27)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    NDIGITS = 8
    x_index_map = {
        key3(grid_centers[i], ndigits=NDIGITS): j
        for j, i in enumerate(sub_indices)
    }

    #3D plot
    fig = plt.figure()
    ax4 = fig.add_subplot(111, projection='3d')

    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 3])
    ax4.set_zlim([-3, 3])
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)

    plot_radius = max(max_length, MIN_3D_PLOT_RADIUS)
    zero_color = sm.cmap(sm.norm(0.0))
    draw_zero_reliability_background(
        ax4,
        grid_squares,
        all_reliable_beta_ranges[-1],
        plot_radius,
        zero_color
    )
    draw_zero_reliability_radius_extension(
        ax4,
        n_x,
        n_z,
        max_length,
        plot_radius,
        zero_color,
    )

    sorted_3d_indices = sorted([-1] + list(range(15)), key=lambda idx: cr_list[idx])
    for you in sorted_3d_indices:

        #3D plot
        arc_color = color_list[you]

        for i, square in tqdm(enumerate(grid_squares), desc=f"Drawing reliability {cr_list[you]:.6f}"):
            for beta_range in all_reliable_beta_ranges[you][i]:
                # draw_wedge(ax, square, beta_range, arc_color)
                draw_rotated_grid(ax4, square, beta_range, arc_color)

    for you in indices:
        for j, i in enumerate(sub_indices):
            pt = grid_centers[i]
            ranges = all_reliable_beta_ranges[you][i]
            if not ranges:
                continue
            update_beta_bar_multicolor(
                ax, pt, ranges,
                zorder=you + 1,
                x_index_map=x_index_map,  # built from the same pts
                ndigits=NDIGITS
            )

    ax4.view_init(elev=35, azim=146, roll=-111)
    R = plot_radius
    ax4.set_xlim(-R, R)
    ax4.set_ylim(-R, R)
    ax4.set_zlim(-R, R)
    set_axes_equal(ax4)
    ax4.set_box_aspect([1, 1, 1])  # forces equal aspect
    axis_length = plot_radius * 1.2  # extend beyond the shape for visibility
    ax4.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # X-axis
    ax4.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Y-axis
    ax4.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Z-axis

    # Add axis labels near arrow tips
    ax4.text(axis_length, 0, 0, r"$x$", fontsize=14, color='k')
    ax4.text(0, axis_length, 0, r"$y$", fontsize=14, color='k')
    ax4.text(0, 0, axis_length, r"$z$", fontsize=14, color='k')
    ax4.set_xlabel("x", fontsize=25)
    ax4.set_ylabel("y", fontsize=25)
    ax4.set_zlabel("z", fontsize=25)
    tick_positions = [-3, -2, -1, 0, 1, 2, 3]
    ax4.set_xticks(tick_positions)
    ax4.set_yticks(tick_positions)
    ax4.set_zticks(tick_positions)
    ax4.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax4.tick_params(axis='y', labelsize=18)
    ax4.tick_params(axis='z', labelsize=18)
    plt.show()


    set_sparse_xyz_labels(ax, sub_grid_centers, max_labels=20, ndigits=2)
    ax.set_xlim(-0.5, len(sub_grid_centers) - 0.5)
    ax.set_facecolor("white")
    ax2.set_xlabel('X', fontsize=24)
    ax2.set_ylabel('Z', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=18)


    frame_points = [
        (x_range[0], z_range[0]), (x_range[1], z_range[0]),
        (x_range[1], z_range[1]), (x_range[0], z_range[1]),
        (x_range[0], z_range[0])  # Closing the loop
    ]

    # Draw frame
    frame_x, frame_z = zip(*frame_points)
    ax2.plot(frame_x, frame_z, color='k', linewidth=2)
    plt.ioff()
    plt.show()

    return "succeed"

#main function
ap = ssm_estimation(proposed_sample_num, d, alpha, l, CA)
print(ap)


