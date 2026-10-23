import numpy as np

import networkx as nx

import cvxpy as cp

import numbotics.robot as rob
import numbotics.spatial as spt
import numbotics.topology as top

from spp_ggcs import GGCS

from smm_map import build_smm_map
from ggcs_map import build_coll_free_map

from mapper import Map

from poly_funcs import (
    poly_contains,
)

from smm_proc import project_smm_to_constraint

from graph_helper import (
    random_path_multiple_goals,
    path_length,
    get_path_edges,
    phi_rounded_path_multiple_goals,
    pre_proc_graph_multiple_goals,
)

import pickle

arm_name = '3R'
# arm_name = '4R'
# arm_name = 'kinova'

arm = rob.Robot(f'./arms/{arm_name}.rob')


fail_probs = [np.random.rand() for _ in range(arm.n)]
print(fail_probs)
fail_probs = [p_i/sum(fail_probs) for p_i in fail_probs]
print(fail_probs)

with open(f'./regions/{arm_name}_c_free_polys.pkl', 'rb') as f:
    c_free = pickle.load(f)
c_free = c_free[:70]

start = np.array([1.5, 3.0, 2.5]) # np.array([2.0, -2.5, 2.0])

goal_loc = np.array([[-0.5, 1.5, 0.0]]).T # -0.5, 0.0]]).T

x_d = spt.trans_mat(pos=goal_loc)
s, smms = arm.all_smms(x_d, samples=250)
assert s

smms = top.tr_inv(smms)
smms = np.swapaxes(smms, 1, 2)

goals = []
for smm in smms:
    for q in smm:
        for r_i in c_free:
            if poly_contains(r_i, q):
                goals.append(q)
                break

failure = (1, -2.5)
post_fail_goals = project_smm_to_constraint(arm, smms, x_d, failure[0], failure[1], polytopes=c_free)
assert post_fail_goals.shape[0] > 0


####################################
####################################
#####                      #########
#                               ####
# PRE-FAILURE MOTION PLANNING   ####
#                               ####
#####                      #########
####################################
####################################


ft_map = build_smm_map(arm, c_free, smms, fail_probs)
ft_map.add_start_goals(start, goals)

search_paths = 100
max_opt_paths = 20
cost_scale = 0.95
max_paths = 1000


ft_mp_G, t = pre_proc_graph_multiple_goals(ft_map.graph, ft_map.s_ind, ft_map.g_inds, cost_scale, max_paths=max_paths, verbose=True)
print(f'fault-tolerant graph pre-process time: {t}')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
axs = fig.add_subplot(1,1,1, projection='3d')
for e in ft_mp_G.edges():
    v_i = ft_map.centers[e[0]]
    v_j = ft_map.centers[e[1]]

    axs.plot([v_i[0],v_j[0]], [v_i[1],v_j[1]], [v_i[2],v_j[2]], linewidth=1.25, color='red')

ft_ggcs = GGCS(ft_map, ft_mp_G, (len(ft_mp_G.nodes()),arm.n), remove_one_cycle=False)
ft_ggcs.solve(verbose=True)

opt_paths = set()
for i in range(search_paths):
    p = phi_rounded_path_multiple_goals(ft_mp_G, ft_ggcs.phi, ft_map.s_ind, ft_map.g_inds)
    if p is not None:
        opt_paths.add(p)
    if len(opt_paths) >= max_opt_paths:
        break
print(f'found {len(opt_paths)} potentially optimal paths')



def path_cost(p, x, xp):
    dist = 0.0
    for e in p:
        c = ft_mp_G[e[0]][e[1]]['cost']
        dist += c*np.linalg.norm(x[e]-xp[e])
    return dist

opt_results = []
for i, p_i in enumerate(opt_paths):
    x, xp, _ = ft_ggcs.solve_single_path(p_i, verbose=True)
    opt_results.append((p_i, x, xp))

opt_results.sort(key=lambda x: path_cost(x[0], x[1], x[2]))




axs.scatter(xs=start[0], ys=start[1], zs=start[2], s=50, c='green')
for goal in goals:
    axs.scatter(xs=goal[0], ys=goal[1], zs=goal[2], s=50, c='red')

p, x, xp = opt_results[0]
for e in p:
    v_i = x[e]
    v_j = xp[e]

    axs.plot([v_i[0],v_j[0]], [v_i[1],v_j[1]], [v_i[2],v_j[2]], linewidth=2.5, color='dodgerblue')


axs.set_xlim([-np.pi,np.pi])
axs.set_ylim([-np.pi,np.pi])
axs.set_zlim([-np.pi,np.pi])

axs.set_xlabel(r'$\theta_{1}$',fontsize=28)
axs.set_ylabel(r'$\theta_{2}$',fontsize=28)
axs.set_zlabel(r'$\theta_{3}$',fontsize=28,rotation=0)

plt.show()


res = []
for e in p:
    res.append(x[e])
res.append(xp[p[-1]])

with open('./exp/3R/example/traj.npy', 'wb') as f:
    np.save(f, np.array(res))




#####################################
#####################################
#####                       #########
#                                ####
# POST-FAILURE MOTION PLANNING   ####
#                                ####
#####                       #########
#####################################
#####################################

c_free_map = build_coll_free_map(arm, c_free, equality=failure)
c_free_map.add_start_goals(start, post_fail_goals)

search_paths = 100
max_opt_paths = 20
cost_scale = 0.9
max_paths = 1000


c_free_mp_G, t = pre_proc_graph_multiple_goals(c_free_map.graph, c_free_map.s_ind, c_free_map.g_inds, cost_scale, max_paths=max_paths, verbose=True)
print(f'collision-free graph pre-process time: {t}')


c_free_ggcs = GGCS(c_free_map, c_free_mp_G, (len(c_free_mp_G.nodes()), arm.n), remove_one_cycle=False, equality_cons=failure)
c_free_ggcs.solve(verbose=True)

opt_paths = set()
for i in range(search_paths):
    p = phi_rounded_path_multiple_goals(c_free_mp_G, c_free_ggcs.phi, c_free_map.s_ind, c_free_map.g_inds)
    if p is not None:
        opt_paths.add(p)
    if len(opt_paths) >= max_opt_paths:
        break
print(f'found {len(opt_paths)} potentially optimal paths')



def path_cost(p, x, xp):
    dist = 0.0
    for e in p:
        c = c_free_mp_G[e[0]][e[1]]['cost']
        dist += c*np.linalg.norm(x[e]-xp[e])
    return dist

opt_results = []
for i, p_i in enumerate(opt_paths):
    x, xp, _ = c_free_ggcs.solve_single_path(p_i, verbose=True, equality_cons=failure)
    opt_results.append((p_i, x, xp))

opt_results.sort(key=lambda x: path_cost(x[0], x[1], x[2]))



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
axs = fig.add_subplot(1,1,1, projection='3d')

axs.scatter(xs=start[0], ys=start[1], zs=start[2], s=50, c='green')
for goal in goals:
    axs.scatter(xs=goal[0], ys=goal[1], zs=goal[2], s=50, c='red')

p, x, xp = opt_results[0]
for e in p:
    v_i = x[e]
    v_j = xp[e]

    axs.plot([v_i[0],v_j[0]], [v_i[1],v_j[1]], [v_i[2],v_j[2]], linewidth=2.5, color='dodgerblue')


axs.set_xlim([-np.pi,np.pi])
axs.set_ylim([-np.pi,np.pi])
axs.set_zlim([-np.pi,np.pi])

axs.set_xlabel(r'$\theta_{1}$',fontsize=28)
axs.set_ylabel(r'$\theta_{2}$',fontsize=28)
axs.set_zlabel(r'$\theta_{3}$',fontsize=28,rotation=0)

plt.show()


res = []
for e in p:
    res.append(x[e])
res.append(xp[p[-1]])

with open('./exp/3R/example/traj_post_fail.npy', 'wb') as f:
    np.save(f, np.array(res))
