import numpy as np

import cvxpy as cp

import numbotics.robot as rob
import numbotics.spatial as spt
import numbotics.topology as top

# from poly_helper import *

import time

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator


arm_name = '3R'

arm = rob.Robot('../arms/' + arm_name + '.rob')

blue = [0.0,0.0,0.8]
gray = [0.4,0.4,0.4]

# cm_arm = matplotlib.colormaps['viridis']

OBSTACLES = [
    np.array([[1.25,0.05],[1.25,-0.45],[1.75,-0.45],[1.75,0.05]]),
    np.array([[-0.25,1.25],[-0.25,1.75],[0.25,1.75],[0.25,1.25]]),
    np.array([[-1.15,0.05],[-1.15,0.55],[-1.65,0.55],[-1.65,0.05]])
]

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amstext} \usepackage{bm} \renewcommand{\Im}{\mathfrak{I}}'
})
# matplotlib.rcParams['lines.color'] = 'crimson'

cmap = cm.get_cmap('viridis')


fig = plt.figure(figsize=(10,10))
axs = fig.add_subplot(1,1,1)
fig2 = plt.figure(figsize=(10,10))
axs2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure(figsize=(10,10))
axs3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure(figsize=(10,10))
axs4 = fig4.add_subplot(1,1,1)

off1 = np.array([[1.0,0.0]]).T
off2 = np.array([[1.0,0.0]]).T
off3 = np.array([[1.0,0.0]]).T


def interp_traj(q, samples):
    q_out = np.zeros((samples+1,q.shape[1]))
    samples -= 1

    length_i = np.linalg.norm(top.tr_inv(top.tr(q[:-1])/top.tr(q[1:])),axis=1)
    tot_length = np.sum(length_i)
    length_per = tot_length/float(samples)

    i = 0
    acc = 0.0
    prev_change_q = 0.0
    change_q = length_i[i]

    q_out[0] = q[0]
    q_ind = 1

    while acc - tot_length < 1e-6:

        while acc - change_q > 1e-6:
            i += 1
            prev_change_q = change_q
            change_q += length_i[i]

        t = (acc-prev_change_q)/(change_q-prev_change_q)
        n = (1.0-t)*q[i] + t*q[i+1]

        q_out[q_ind] = n
        q_ind += 1

        acc += length_per

    return q_out


def line_integral(cost_function, points, step_size=0.1):

    integral = 0.0
    num_points = len(points)

    for i in range(num_points - 1):

        start_point = points[i]
        end_point = points[i + 1]

        vector = end_point - start_point
        segment_length = np.linalg.norm(vector)
        unit_vector = vector / segment_length

        position = np.copy(start_point)

        while np.linalg.norm(position - start_point) < segment_length:

            next_position = position + unit_vector * step_size
            if np.linalg.norm(next_position - start_point) > segment_length:
                next_position = end_point

            avg_cost = 0.5 * (cost_function(position) + cost_function(next_position))

            distance_moved = np.linalg.norm(next_position - position)
            integral += avg_cost * distance_moved

            position = next_position

    return integral

def T(tht):
    return np.array([[np.cos(tht), -np.sin(tht)],[np.sin(tht),np.cos(tht)]])

# def draw_fig1(regions):
#     for r in regions:
#         plot_vpolytope(axs, r.vertices(), fc='pink')
#     axs.set_xlabel(r'$\theta_{1}$',fontsize=42,labelpad=14)
#     axs.set_ylabel(r'$\theta_{2}$',fontsize=42,rotation=0,labelpad=22)
#     axs.set_xlim([-np.pi,np.pi])
#     axs.set_ylim([-np.pi,np.pi])
#     axs.tick_params(axis='both', labelsize=25)
#     axs.set_aspect('equal','box')


def draw_fig2(obstacles, ax):
    for i, o in enumerate(obstacles):
        plot_vpolytope(ax, o.T)
    ax.set_xlabel('X',fontsize=42,labelpad=2)
    ax.set_ylabel('Y',fontsize=42,rotation=0,labelpad=10)
    ax.set_xlim([-2.7,2.7])
    ax.set_ylim([-1.8,3.2])
    ax.tick_params(axis='both', labelsize=24)
    ax.set_aspect('equal','box')


def draw_ee_traj(start, end, ax, color):
    T1 = T(start[0])
    T2 = T(start[1])
    T3 = T(start[2])

    o = np.array([0.0,0.0])
    p1 = (T1@off1)[:,0]
    p2 = p1 + ((T1@T2)@off2)[:,0]
    ee_start = p2 + ((T1@T2@T3)@off3)[:,0]

    T1 = T(end[0])
    T2 = T(end[1])
    T3 = T(end[2])

    o = np.array([0.0,0.0])
    p1 = (T1@off1)[:,0]
    p2 = p1 + ((T1@T2)@off2)[:,0]
    ee_end = p2 + ((T1@T2@T3)@off3)[:,0]

    ax.plot([ee_start[0],ee_end[0]],[ee_start[1],ee_end[1]], linewidth=3.0, color=color)


def draw_arm(tht1, tht2, tht3, ax, color='black', alpha=1.0, zorder=1):
    T1 = T(tht1)
    T2 = T(tht2)
    T3 = T(tht3)

    o = np.array([0.0,0.0])
    p1 = (T1@off1)[:,0]
    p2 = p1 + ((T1@T2)@off2)[:,0]
    p3 = p2 + ((T1@T2@T3)@off3)[:,0]

    ax.plot([o[0],p1[0]],[o[1],p1[1]], linewidth=4.0, color=color,alpha=(alpha+0.3)/1.3, zorder=zorder)
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]], linewidth=4.0, color=color,alpha=(alpha+0.3)/1.3, zorder=zorder)
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]], linewidth=4.0, color=color,alpha=(alpha+0.3)/1.3, zorder=zorder)

    ax.scatter(o[0],o[1],s=35.0,color=blue,zorder=4,alpha=(alpha+0.7)/1.7)
    ax.scatter(p1[0],p1[1],s=35.0,color=blue,zorder=4,alpha=(alpha+0.7)/1.7)
    ax.scatter(p2[0],p2[1],s=35.0,color=blue,zorder=4,alpha=(alpha+0.7)/1.7)


def plot_vpolytope(axis, V, **kwargs):
    axis.fill(*V, **kwargs)#, color='saddlebrown')
    # axs.fill(*V+np.array([[2.0*np.pi,0.0]]).T, **kwargs)
    # axs.fill(*V+np.array([[-2.0*np.pi,0.0]]).T, **kwargs)
    # axs.fill(*V+np.array([[0.0,2.0*np.pi]]).T, **kwargs)
    # axs.fill(*V+np.array([[0.0,-2.0*np.pi]]).T, **kwargs)

def plot_arm_traj(q, cost_f, samples, ax, fg):
    samples -= 1

    length_i = np.linalg.norm(top.tr_inv(top.tr(q[:-1])/top.tr(q[1:])),axis=1)
    tot_length = np.sum(length_i)
    length_per = tot_length/float(samples)

    ax.cla()
    draw_fig2(OBSTACLES, ax)

    i = 0
    acc = 0.0
    prev_change_q = 0.0
    change_q = length_i[i]

    q_interp = interp_traj(q, 250)
    for j in range(len(q_interp)-1):
        draw_ee_traj(q_interp[j], q_interp[j+1], ax, cmap((cost_f(q_interp[j+1])-0.29)/0.71))

    draw_arm(q[0,0], q[0,1], q[0,2], ax, alpha=0.0)

    while acc - tot_length < 1e-6:

        while acc - change_q > 1e-6:
            i += 1
            prev_change_q = change_q
            change_q += length_i[i]

        t = (acc-prev_change_q)/(change_q-prev_change_q)
        n = (1.0-t)*q[i] + t*q[i+1]

        alp = min(acc/tot_length,1.0)
        col = gray
        z_o = 3
        # if c_func(n) > np.inf: #0.96:
        #     alp = 1.0
        #     col='#ffa6c9'
        #     z_o = 2

        #     T1 = T(n[0])
        #     T2 = T(n[1])
        #     T3 = T(n[2])

        #     o = np.array([0.0,0.0])
        #     p1 = (T1@off1)[:,0]
        #     p2 = p1 + ((T1@T2)@off2)[:,0]
        #     p3 = p2 + ((T1@T2@T3)@off3)[:,0]
        #     faults = np.zeros((2,3))
        #     faults[:,0] = o
        #     faults[:,1] = p2
        #     faults[:,2] = p3

        #     ax.plot(faults[0,:], faults[1,:], color='crimson', linestyle=(0,(1,1)), linewidth=3.0, zorder=4, label='Robot After Joint 2 Failure')
        #     ax.scatter(faults[0,:-1], faults[1,:-1], color='crimson', s=22.0, zorder=4)

        draw_arm(n[0], n[1], n[2], ax, color=col, alpha=alp, zorder=z_o)

        acc += length_per

    fg.canvas.draw()


c_opt_out = np.load('../exp/3R/example/traj.npy')
l_opt_out = np.load('../exp/3R/example/traj_post_fail.npy')


def make_angles_continuous(angles):
    # The adjusted matrix of angles, initialized with the original angles
    adjusted_angles = angles.copy()

    # Iterate over each joint (column)
    for i in range(angles.shape[1]):
        # Calculate the difference between consecutive angles for the current joint
        d_angles = np.diff(angles[:, i])

        # Detect wraparounds, where the difference is greater than pi (jump from pi to -pi)
        # or less than -pi (jump from -pi to pi)
        wraparounds = np.where(d_angles > np.pi, -2*np.pi, np.where(d_angles < -np.pi, 2*np.pi, 0))

        # Adjust the wraparounds to make the angle transition continuous
        # We use a cumulative sum to adjust subsequent angles based on prior adjustments
        adjustments = np.cumsum(wraparounds)

        # Apply the adjustments to the original angle array for the current joint,
        # skipping the first angle because diff() results in one fewer element,
        # and we don't adjust the first angle.
        adjusted_angles[1:, i] += adjustments

    return adjusted_angles


c_opt_out = make_angles_continuous(c_opt_out)
l_opt_out = make_angles_continuous(l_opt_out)



def c_func(q):
    arm.q = q
    return 1.0 - arm.min_post_fail_sing

cost = line_integral(c_func, c_opt_out, step_size=0.001)
print('traj cost:', cost)
arr_colors = ['#ffa6c9', 'green', 'violet']
colors = ['dodgerblue', 'darkviolet', 'crimson', 'lightseagreen', 'slategray', 'lightcoral', 'seagreen', 'darkslateblue']


plot_arm_traj(c_opt_out, c_func, 3, axs2, fig2)
plot_arm_traj(l_opt_out, c_func, 3, axs, fig)

min_cost = np.inf
q_interp = interp_traj(c_opt_out, 250)
for q in q_interp:
    if c_func(q) < min_cost:
        min_cost = c_func(q)
print(f'min cost: {min_cost}')

# cbar.ax.set_xlabel('Cost', fontsize=42)

# norm = matplotlib.colors.Normalize(vmin=0.29,vmax=1.0)
# mappable = cm.ScalarMappable(norm=norm)
# cbar = fig.colorbar(mappable,ax=axs,shrink=0.745)
# cbar.ax.tick_params(axis='y', labelsize=24)

# norm = matplotlib.colors.Normalize(vmin=0.29,vmax=1.0)
# mappable = cm.ScalarMappable(norm=norm)
# cbar = fig2.colorbar(mappable,ax=axs2,shrink=0.745)
# cbar.ax.tick_params(axis='y', labelsize=24)

arm.q = c_opt_out[0]
p_start = arm.fk[[0,1],3]
arm.q = c_opt_out[-1]
p_end = arm.fk[[0,1],3]

axs2.scatter(p_start[0],p_start[1],color='green',s=60.0,zorder=25)
axs2.scatter(p_end[0],p_end[1],color='red',s=60.0,zorder=25)


# axs.legend(fontsize=32,markerscale=0.5)

# fig3 = plt.figure(figsize=(10,10))
# fig4 = plt.figure(figsize=(10,10))

# axs3 = fig3.add_subplot(1,1,1)
# axs4 = fig4.add_subplot(1,1,1)

# # axs3.plot(l_opt_out[:,0],l_opt_out[:,1],color=colors[0],linewidth=4.0,label='Shortest Distance')
# # axs4.plot(l_opt_out[:,1],l_opt_out[:,2],color=colors[0],linewidth=4.0,label='Shortest Distance')
axs3.plot(c_opt_out[:,0],c_opt_out[:,1],color=colors[0],linewidth=6.0,label='Lowest Cost')
axs4.plot(c_opt_out[:,1],c_opt_out[:,2],color=colors[0],linewidth=6.0,label='Lowest Cost')
axs3.scatter(c_opt_out[0,0],c_opt_out[0,1],color='green',s=120.0,zorder=2)
axs4.scatter(c_opt_out[0,1],c_opt_out[0,2],color='green',s=120.0,zorder=2)
axs3.scatter(c_opt_out[-1,0],c_opt_out[-1,1],color='red',s=120.0,zorder=2)
axs4.scatter(c_opt_out[-1,1],c_opt_out[-1,2],color='red',s=120.0,zorder=2)

axs3.grid(True)
axs4.grid(True)

axs3.set_xlim([-np.pi,np.pi])
axs3.set_ylim([-np.pi,np.pi])
axs4.set_xlim([-np.pi,np.pi])
axs4.set_ylim([-np.pi,np.pi])
axs3.set_xlabel(r'$\theta_{1}$',fontsize=60)
axs3.set_ylabel(r'$\theta_{2}$',fontsize=60, rotation=0, labelpad=12.0)
axs4.set_xlabel(r'$\theta_{2}$',fontsize=60)
axs4.set_ylabel(r'$\theta_{3}$',fontsize=60, rotation=0, labelpad=12.0)


axs3.tick_params(axis='x', labelsize=40)
axs3.tick_params(axis='y', labelsize=40)
axs4.tick_params(axis='x', labelsize=40)
axs4.tick_params(axis='y', labelsize=40)

axs3.yaxis.set_major_locator(MaxNLocator(5))
axs3.xaxis.set_major_locator(MaxNLocator(5))
axs4.yaxis.set_major_locator(MaxNLocator(5))
axs4.xaxis.set_major_locator(MaxNLocator(5))

# # axs3.legend(fontsize=18)
# # axs4.legend(fontsize=18)

# axs3.legend(fontsize=32,markerscale=0.5)
# axs4.legend(fontsize=32,markerscale=0.5)

fig3.tight_layout()
fig4.tight_layout()


plt.show()
