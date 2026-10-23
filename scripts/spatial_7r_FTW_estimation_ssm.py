import time
from tqdm import tqdm

import numpy as np
from scipy.linalg import null_space
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from helper_functions import fibonacci_sphere_angles, union_ranges, compute_reliability, compute_reliability_by_f_list, generate_grid_centers



step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 10
max_ssm = 16
positional_samples = 8
orientation_samples = 3
theta_phi_list = fibonacci_sphere_angles(orientation_samples)

joint_reliabilities=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]

alpha = [-62 * np.pi / 180, -79 * np.pi / 180, 90 * np.pi / 180, 29 * np.pi / 180, 81 * np.pi / 180, -80 * np.pi / 180,0]#was -90*np.pi/180
l = [0.4, 0.8, 0.2, 1, 0.6, 0.4, 0]#last one was 0.2
d = [-0.4, -0.6, 0.2, 0.6, -0.8, 0.2, 0]# was 0.8
CA = [(-107 * np.pi / 180, 107 * np.pi / 180), (-164 * np.pi / 180, 141 * np.pi / 180),
      (-132 * np.pi / 180, 132 * np.pi / 180), (-151 * np.pi / 180, 102 * np.pi / 180),
      (-115 * np.pi / 180, 149 * np.pi / 180), (-75 * np.pi / 180, 129 * np.pi / 180),
      (16 * np.pi / 180, 193 * np.pi / 180)]

def zyz_to_R(psi, theta, phi):
    rot_fwd = R.from_euler('zyz', [psi, theta, phi], degrees=False)
    R_mat = rot_fwd.as_matrix()
    return R_mat
def R_to_zyz(R_mat):
    rot_back = R.from_matrix(R_mat)
    recovered_angles = rot_back.as_euler('zyz', degrees=False)
    return recovered_angles
def angle_between_2_orientation(R1, R2):
    # print(np.arccos((np.trace(np.dot(R1.T, R2)) - 1) / 2))
    return np.arccos((np.trace(np.dot(R1.T, R2)) - 1) / 2)
def range_sum(range_list):
    result = 0.0
    for single_range in range_list:
        result = +(single_range[1] - single_range[0])
    return result
def sample_points_in_ranges(ranges, min_distance):
    """
    Samples as many points as possible in the given ranges such that the
    distance between any two points is at least `min_distance`.

    Parameters:
    - ranges: list of tuples [(start1, end1), (start2, end2), ...]
              Each tuple represents a range [start, end].
    - min_distance: float
                    Minimum distance between any two points.

    Returns:
    - List of sampled points.
    """
    sampled_points = []

    for range_start, range_end in ranges:
        # Generate evenly spaced points in the current range
        num_points = int((range_end - range_start) // min_distance) + 1
        points = np.linspace(range_start, range_end, num_points)
        sampled_points.extend(points)

    return sampled_points
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
def stepwise_ssm(theta, n_j, Tep, previous_n_j, robot):
    x_pos_ori = robot.fkine(theta.flatten())
    x_pos = np.array(x_pos_ori)[:3, 3].T.reshape((3, 1))
    x_target = np.array(Tep)[:3, 3].T.reshape((3, 1))
    cn = np.array(x_pos_ori)[:3, 0].T.reshape((3, 1)).flatten()
    co = np.array(x_pos_ori)[:3, 1].T.reshape((3, 1)).flatten()
    ca = np.array(x_pos_ori)[:3, 2].T.reshape((3, 1)).flatten()
    tn = np.array(Tep)[:3, 0].T.reshape((3, 1)).flatten()
    to = np.array(Tep)[:3, 1].T.reshape((3, 1)).flatten()
    ta = np.array(Tep)[:3, 2].T.reshape((3, 1)).flatten()
    delta_pos = x_target - x_pos
    # delta_ori_n = np.cross(tn, cn)
    # delta_ori_o = np.cross(to, co)
    # delta_ori_a = np.cross(ta, ca)
    delta_ori_n = np.cross(cn, tn)
    delta_ori_o = np.cross(co, to)
    delta_ori_a = np.cross(ca, ta)
    delta_ori = 0.5 * (delta_ori_n + delta_ori_o + delta_ori_a).reshape((3, 1))
    delta_x_step = np.vstack((delta_pos, delta_ori))
    # print(delta_x_step)
    # J_step = robot.jacob0(theta.flatten())[:3, :]
    J_step = robot.jacob0(theta.flatten())
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    # dq = n_j + corrected_delta_theta * np.linalg.norm(corrected_delta_theta)
    # corrected_delta_theta/= np.linalg.norm(corrected_delta_theta)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + dq * step_size

    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    # new_J = robot.jacob0(theta_next.flatten())[:3, :]
    new_J = robot.jacob0(theta_next.flatten())
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
    if np.dot(new_n_j.reshape(-1), previous_n_j.reshape(-1)) < 0:
        new_n_j = -new_n_j
    # print(f'n_j {new_n_j[0]}')
    # print(f'delta {delta_x_step[0]}')
    """ 
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
    """

    return theta_next, new_n_j, n_j
def find_intersection_points(ssm_theta_list, C_dot_A):
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
    return union_ranges(ip_ranges)
def find_ip_7th_joint(ssm_theta_list, C_dot_A_7th):
    counter = 0
    ip_ranges = []
    isin_list = []
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(any(r[0] <= v <= r[1] for r in ranges) for v, ranges in zip(theta_flatten, C_dot_A_7th)):
            isin_list.append(theta_index)
            tof = True
            counter += 1
    ip_index_ranges = []
    if len(isin_list) != 0: ip_index_ranges = find_ranges(isin_list)

    for index_range in ip_index_ranges:
        local_min_theta7 = 100
        local_max_theta7 = -100
        for i in range(index_range[0], index_range[1] + 1):
            if ssm_theta_list[i][6][0] < local_min_theta7:
                local_min_theta7 = ssm_theta_list[i][6][0]
            if ssm_theta_list[i][6][0] > local_max_theta7:
                local_max_theta7 = ssm_theta_list[i][6][0]
        if np.abs(local_min_theta7 + np.pi) <= 1e-2: local_min_theta7 = -np.pi
        if np.abs(local_min_theta7 - np.pi) <= 1e-2: local_min_theta7 = np.pi
        if np.abs(local_max_theta7 + np.pi) <= 1e-2: local_max_theta7 = -np.pi
        if np.abs(local_max_theta7 - np.pi) <= 1e-2: local_max_theta7 = np.pi
        ip_ranges.append([local_min_theta7, local_max_theta7])
        # print([local_min_theta1, local_max_theta1])
    return union_ranges(ip_ranges)
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
            return [[- 3 * np.pi, np.pi], [-np.pi, 3 * np.pi]]
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
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf],
                       [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], [], [], [], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
                , [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
                , [ssm_theta_list[0][4][0], ssm_theta_list[0][4][0]]
                , [ssm_theta_list[0][5][0], ssm_theta_list[0][5][0]]
                , [ssm_theta_list[0][6][0], ssm_theta_list[0][6][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        thetas_cp_range[3] = [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
        thetas_cp_range[4] = [ssm_theta_list[0][4][0], ssm_theta_list[0][4][0]]
        thetas_cp_range[5] = [ssm_theta_list[0][5][0], ssm_theta_list[0][5][0]]
        thetas_cp_range[6] = [ssm_theta_list[0][6][0], ssm_theta_list[0][6][0]]
        previous_theta = ssm_theta_list[0]
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(7):
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
    for j in range(7):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
    # print(thetas_cp_sum)
    return thetas_cp_sum
def find_single_intersection(ssm_theta_list):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof
def find_random_ssm(r, x_target, all_ssm_theta_list, robot, C_dot_A, C_dot_A_7):
    ssm_found = False

    # Initialize random joint configuration
    q = np.random.uniform(low=-np.pi, high=np.pi, size=(7,))
    q = q.reshape((7, 1))

    # Generate a random rotation matrix
    # Method: create a random matrix and use QR decomposition
    # R is now a proper rotation matrix

    # Construct the SE3 with random orientation and given position x_target
    # Assuming x_target is a 3D vector representing desired position
    Tep = SE3.Rt(r, x_target)

    # Now solve IK with full orientation constraints
    result = robot.ikine_LM(Tep, mask=[1, 1, 1, 1, 1, 1], q0=q)

    if not result.success:
        return result.success, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found, [],False

    sol = result.q.reshape((7, 1))
    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found, [],False
    theta = sol
    theta_prime = theta.copy()

    # J = robot.jacob0(theta.flatten())[:3, :]
    J = robot.jacob0(theta.flatten())
    n_j = null_space(J)[:, 0].T.reshape((7, 1))
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    all_dis = []
    tf_reset = False
    while True:
        num += 1
        # print(num)

        if threshold <= terminate_threshold and num >= 4: break

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

            # Plot the list of values. By default, the x-axis will be the index of each value (0, 1, 2, ...)
            plt.plot(all_dis, marker='o')

            # Add labels and title
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Plot of Values vs. Index')

            # Display the plot
            plt.show()
            
            break
            """
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, Tep, old_n_j, robot)
        threshold = np.linalg.norm(theta - theta_prime)
        all_dis.append(threshold)
        if threshold < lowest: lowest = threshold

        if num == 2000:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found, [],False

        if num == 10000 and not tf_reset:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found, [],False

            theta_prime = theta
            num = 0
            ssm_theta_list = [theta]
            all_dis = []
            tf_reset = True

        # m, n = np.shape(new_n_j)
        n_j = new_n_j

        if num > 15000:
            print('stuck at a too small smm')
            print('should never show up now')
            # for configuration in all_ssm_theta_list:
            #    if np.linalg.norm(configuration - theta) <= terminate_threshold:
            #        # print('ssm guided to wrong direction.')
            #        return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found
            # print('still not found')
            return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found, [],False
        ssm_theta_list.append(theta)


    all_ssm_theta_list.extend(ssm_theta_list)
    print(f'found a new ssm with {num} points.')
    ssm_found = True

    ip_ranges = find_intersection_points(ssm_theta_list, C_dot_A)
    ip_ranges_alpha = find_ip_7th_joint(ssm_theta_list, C_dot_A_7)
    cp_ranges = find_critical_points(ssm_theta_list)
    single_intersection_tf = find_single_intersection(ssm_theta_list)
    return True, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found, ip_ranges_alpha,single_intersection_tf
def compute_beta_range(r, target_x, robot, C_dot_A, CA):
    all_smm_beta_range = []
    all_alpha_ranges = []
    all_theta = []
    alpha0_ranges = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    theta5_ranges = []
    theta6_ranges = []
    theta7_ranges = []
    F_list=[]
    find_count = 0
    ssm_found = 0
    C_dot_A_7 = CA.copy()
    C_dot_A_7[6] = (-np.pi, np.pi)
    C_dot_A_7 = convert_to_C_dot_A(C_dot_A_7)
    single_intersection_tf_all = False
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf, iprs7,single_intersection_tf = find_random_ssm(
            r, target_x, all_theta, robot, C_dot_A, C_dot_A_7)
        single_intersection_tf_all=single_intersection_tf_all or single_intersection_tf
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        iprs7 = extend_ranges(iprs7)
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

        for intersection_range in iprs7:
            alpha0_lm = CA[6][0] - intersection_range[1]
            alpha0_um = CA[6][1] - intersection_range[0]
            if alpha0_um - alpha0_lm >= 2 * np.pi:
                alpha0_ranges.append([-np.pi, np.pi])
            elif alpha0_lm < -np.pi:
                alpha0_ranges.append([alpha0_lm + 2 * np.pi, np.pi])
                alpha0_ranges.append([-np.pi, alpha0_um])
            elif alpha0_um > np.pi:
                alpha0_ranges.append([-np.pi, alpha0_um - 2 * np.pi])
                alpha0_ranges.append([alpha0_lm, np.pi])
            else:
                alpha0_ranges.append([alpha0_lm, alpha0_um])
        if ssm_found_tf:  ssm_found += 1; find_count = 0

        for cp_range in cp_ranges[0]:
            if cp_range[0] > -np.inf: theta1_ranges.append(cp_range)
        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf: theta2_ranges.append(cp_range)
        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf: theta3_ranges.append(cp_range)
        for cp_range in cp_ranges[3]:
            if cp_range[0] > -np.inf: theta4_ranges.append(cp_range)
        for cp_range in cp_ranges[4]:
            if cp_range[0] > -np.inf: theta5_ranges.append(cp_range)
        for cp_range in cp_ranges[5]:
            if cp_range[0] > -np.inf: theta6_ranges.append(cp_range)
        for cp_range in cp_ranges[6]:
            if cp_range[0] > -np.inf: theta7_ranges.append(cp_range)

    #if len(theta1_ranges) == 0: return [], [],0
    #if len(theta2_ranges) == 0: return [], [],0
    #if len(theta3_ranges) == 0: return [], [],0
    #if len(theta4_ranges) == 0: return [], [],0
    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)
    theta5_ranges_union = union_ranges(theta5_ranges)
    theta6_ranges_union = union_ranges(theta6_ranges)
    theta7_ranges_union = union_ranges(theta7_ranges)

    """
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
    """
    ion1_alpha = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    if single_intersection_tf_all: F_list.append(0)
    for tr in theta1_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1_alpha = True
            if single_intersection_tf_all: F_list.append(1)

    if CA[0][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[0][1] <= has_negative_pi_range and CA[0][0] + 2 * np.pi >= has_positive_pi_range:
                ion1_alpha = True
                if single_intersection_tf_all: F_list.append(1)

    if CA[0][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[0][1] - 2 * np.pi <= has_negative_pi_range and CA[0][0] >= has_positive_pi_range:
                ion1_alpha = True
                if single_intersection_tf_all: F_list.append(1)

    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            if single_intersection_tf_all: F_list.append(1)
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
            if single_intersection_tf_all: F_list.append(2)
            # print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                if single_intersection_tf_all: F_list.append(2)
                # print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                if single_intersection_tf_all: F_list.append(2)
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
            if single_intersection_tf_all: F_list.append(3)
            # print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                if single_intersection_tf_all: F_list.append(3)
                # print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                if single_intersection_tf_all: F_list.append(3)
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
            if single_intersection_tf_all: F_list.append(4)
            # print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                if single_intersection_tf_all: F_list.append(4)
                # print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                if single_intersection_tf_all: F_list.append(4)
                # print('joint 4 succeed')

    ion5 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta5_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[4][0] >= tr[0] and CA[4][1] <= tr[1]:
            ion5 = True
            if single_intersection_tf_all: F_list.append(5)
            # print('joint 5 succeed')
    if CA[4][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[4][1] <= has_negative_pi_range and CA[4][0] + 2 * np.pi >= has_positive_pi_range:
                ion5 = True
                if single_intersection_tf_all: F_list.append(5)
                # print('joint 5 succeed')
    if CA[4][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[4][1] - 2 * np.pi <= has_negative_pi_range and CA[4][0] >= has_positive_pi_range:
                ion5 = True
                if single_intersection_tf_all: F_list.append(5)
                # print('joint 5 succeed')

    ion6 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta6_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[5][0] >= tr[0] and CA[5][1] <= tr[1]:
            ion6 = True
            if single_intersection_tf_all: F_list.append(6)
            # print('joint 6 succeed')
    if CA[5][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[5][1] <= has_negative_pi_range and CA[5][0] + 2 * np.pi >= has_positive_pi_range:
                ion6 = True
                if single_intersection_tf_all: F_list.append(6)
                # print('joint 6 succeed')
    if CA[5][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[5][1] - 2 * np.pi <= has_negative_pi_range and CA[5][0] >= has_positive_pi_range:
                ion6 = True
                if single_intersection_tf_all: F_list.append(6)
                # print('joint 6 succeed')

    ion7 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    # print(theta7_ranges_union)
    for tr in theta7_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[6][0] >= tr[0] and CA[6][1] <= tr[1]:
            ion7 = True
            if single_intersection_tf_all: F_list.append(7)
            # print('joint 7 succeed')
    if CA[6][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[6][1] <= has_negative_pi_range and CA[6][0] + 2 * np.pi >= has_positive_pi_range:
                ion7 = True
                if single_intersection_tf_all: F_list.append(7)
                # print('joint 7 succeed')
    if CA[6][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[6][1] - 2 * np.pi <= has_negative_pi_range and CA[6][0] >= has_positive_pi_range:
                ion7 = True
                if single_intersection_tf_all: F_list.append(7)

    ion7_alpha = False
    min_alpha1 = 0
    max_alpha1 = 0
    theta7_ranges_union = extend_ranges(theta7_ranges_union)
    # print(theta7_ranges_union)
    for tr in theta7_ranges_union:
        if CA[6][0] >= tr[0] and CA[6][1] <= tr[1]:
            ion7_alpha = True
            if single_intersection_tf_all: F_list.append(7)
            min_alpha1 = CA[6][1] - tr[1]
            max_alpha1 = CA[6][0] - tr[0]
            if (max_alpha1 - min_alpha1 >= 2 * np.pi) or (tr[0] == -np.pi and tr[1] == np.pi):
                max_alpha1 = np.pi
                min_alpha1 = -np.pi

    #if ion1 and ion2 and ion3 and ion4 and ion5 and ion6 and ion7:
    if ion1 and (ion2 or ion3 or ion4 or ion5 or ion6 or ion7):
        # print(ion1_alpha, ion2, ion3, ion4, ion5, ion6, ion7_alpha)
        for index in range(len(beta0_ranges)):
            min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
            min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
            max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
            if min_beta_f_ftw <= max_beta_f_ftw:
                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
    #if ion1_alpha and ion2 and ion3 and ion4 and ion5 and ion6 and  ion7_alpha:
    if (ion1_alpha or ion2 or ion3 or ion4 or ion5 or ion6) and ion7_alpha:
        for index in range(len(alpha0_ranges)):
            min_alpha0, max_alpha0 = alpha0_ranges[index][0], alpha0_ranges[index][1]
            min_alpha_f_ftw = max(min_alpha0, min_alpha1, -np.pi)
            max_alpha_f_ftw = min(max_alpha0, max_alpha1, np.pi)
            if min_alpha_f_ftw <= max_alpha_f_ftw:
                all_alpha_ranges.append([min_alpha_f_ftw, max_alpha_f_ftw])
    return union_ranges(all_smm_beta_range), union_ranges(all_alpha_ranges),compute_reliability([ion2,ion3,ion4,ion5,ion6],joint_reliabilities[1:-1]),F_list

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
            RevoluteDH(d=d[4], alpha=alpha[4], a=l[4], qlim=CA[4]),
            RevoluteDH(d=d[5], alpha=alpha[5], a=l[5], qlim=CA[5]),
            RevoluteDH(d=d[6], alpha=alpha[6], a=l[6], qlim=CA[6]),
        ], name="spatial 7R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(7):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    print(max_length)



    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
    angle_ranges = []
    reachable_points = 0
    all_data = []
    index = 0


    for center in tqdm(grid_centers, desc="Processing Items"):
        # Compute beta ranges for each center
        #print(center)
        all_alpha_ranges = []
        all_beta_ranges = []
        all_realiability=[]
        all_true_reliability=[]
        positional_beta_ranges = []
        target_x = np.array([center[0], center[1], center[2]]).T.reshape((3, 1))
        for sample_tuple in tqdm(theta_phi_list, desc="Processing Items"):
            sampled_orientation = zyz_to_R(sample_tuple[0], sample_tuple[1], 0)
            beta_ranges, alpha_ranges,reliability,F_list = compute_beta_range(sampled_orientation, target_x, robot, C_dot_A, CA)
            all_alpha_ranges.append(alpha_ranges)
            all_realiability.append(reliability)
            all_true_reliability.append(compute_reliability_by_f_list(F_list,joint_reliabilities))
            positional_beta_ranges.extend(beta_ranges)
            positional_beta_ranges = union_ranges(positional_beta_ranges)

            if len(beta_ranges) != 0:
                for beta_range in beta_ranges:
                    limit1 = beta_range[0] + sample_tuple[0] + np.pi
                    limit2 = beta_range[1] + sample_tuple[0] + np.pi
                    if limit1 < 0:
                        limit1 += 2 * np.pi
                    if limit1 > 2 * np.pi:
                        limit1 -= 2 * np.pi
                    if limit2 < 0:
                        limit2 += 2 * np.pi
                    if limit2 > 2 * np.pi:
                        limit2 -= 2 * np.pi
            all_beta_ranges.append(beta_ranges)


        all_data.append((0, index, all_beta_ranges, all_alpha_ranges,all_realiability,all_true_reliability))
        index += 1

        if len(positional_beta_ranges) != 0: reachable_points += 1
        angle_ranges.append(positional_beta_ranges)

    with open("my_7r_list.txt", "w") as file:
        file.write(str(all_data))




ap = ssm_estimation(positional_samples, d, alpha, l, CA)

