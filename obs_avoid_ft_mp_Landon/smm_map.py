import numpy as np

import networkx as nx

import numbotics.robot as rob
import numbotics.spatial as spt
import numbotics.topology as top

from rtree import index

from mapper import Map

from poly_funcs import *

from smm_proc import (
    smm_to_bb,
    cut_smm,
    union_intervals,
)

import itertools


EPSILON = 5e-2


def build_smm_map(arm, c_free, smms, fail_probs):

    goals = []
    limits = []
    _limits = []

    for i in range(arm.n):
        _limits.append([])

    for smm in smms:
        valid = []
        for q_i in smm:
            valid.append(False)
            for c_i in c_free:
                if poly_contains(c_i, q_i):
                    valid[-1] = True
                    goals.append(q_i)
                    break

        cut_smms = cut_smm(smm, np.array(valid))
        for c_smm in cut_smms:
            bb = smm_to_bb(c_smm)
            for i, bb_i in enumerate(bb):
                _limits[i].extend(bb_i)

    for k, l_i in enumerate(_limits):
        assert len(l_i) > 0

        limits.append(union_intervals(l_i))
        for i in range(len(limits[-1])):
            limits[-1][i].append(0.0)
        for j in range(len(limits[-1])-1):
            limits[-1].insert((2*j)+1, [limits[-1][2*j][1], limits[-1][(2*j)+1][0], fail_probs[k]])

        if limits[-1][0][0] - 1e-1 < -np.pi:
            limits[-1][0][0] = -3.25
        else:
            limits[-1].insert(0, [-3.25, limits[-1][0][0], fail_probs[k]])

        if limits[-1][-1][1] + 1e-1 > np.pi:
            limits[-1][-1][1] = 3.25
        else:
            limits[-1].insert(len(limits[-1]), [limits[-1][-1][1], 3.25, fail_probs[k]])

        for j in range(len(limits[-1])):
            limits[-1][j] = tuple(limits[-1][j])
        limits[-1] = tuple(limits[-1])

    limits = tuple(limits)
    costs = []
    Qs = []
    Q_trans = {}

    limit_inds = list(itertools.product(*[range(len(l_i)) for l_i in limits]))
    for i, l_inds in enumerate(limit_inds):
        ll = []
        ul = []
        cost = 0.0
        for j, k in enumerate(l_inds):
            ll.append(limits[j][k][0])
            ul.append(limits[j][k][1])
            cost += limits[j][k][2]
        cost = max(cost, EPSILON)
        costs.append(cost)
        Qs.append(limits_to_poly(ll, ul))


    for i in range(len(Qs)):
        Qs[i] = poly_scale_quick(Qs[i], 1.02)

    goals = np.array(goals)
    assert goals.shape[0] > 0

    inters = []
    inters_regions = []
    inters_offsets = []
    centers = []
    _verts = []


    offsets = [
        np.array([ 0.0,        0.0,        0.0      ]),
        np.array([ 2.0*np.pi,  0.0,        0.0      ]),
        np.array([-2.0*np.pi,  0.0,        0.0      ]),
        np.array([ 0.0,        2.0*np.pi,  0.0      ]),
        np.array([ 0.0,       -2.0*np.pi,  0.0      ]),
        np.array([ 0.0,        0.0,        2.0*np.pi]),
        np.array([ 0.0,        0.0,       -2.0*np.pi]),
    ]

    regions = []
    reg_costs = []
    for i in range(len(c_free)):
        for j in range(len(Qs)):
            c_free_reg = c_free[i]
            bb_reg = Qs[j]
            p_int = poly_intersect(c_free_reg, bb_reg)
            if not poly_empty(p_int):
                regions.append(p_int)
                reg_costs.append(costs[j])


    reg_aabb = []
    print('building cost region bounding boxes')
    reg_aabbs = index.Index(properties=index.Property(dimension=arm.n))
    for k, offset in enumerate(offsets):
        for i, r_i in enumerate(regions):
            r_i_trans = poly_translate(r_i, offset)
            reg_aabb.append(poly_aabb(r_i_trans))
            reg_aabbs.insert((k*len(regions))+i, reg_aabb[-1])

    for i in range(len(regions)):
        reg_inters = list(reg_aabbs.intersection(reg_aabb[i]))
        for _j in reg_inters:
            j = _j%len(regions)
            k = _j//len(regions)
            print(f'c free regions: {i}, {j}       \r', end='')
            if i == j:
                continue

            p_int = poly_intersect(regions[i], poly_translate(regions[j], offsets[k]))
            d = poly_cheb_center(p_int)
            if d is not None:
                inters.append((i,j))
                inters_regions.append(p_int)
                inters_offsets.append(offsets[k])
                centers.append(d)
    print()

    G = nx.DiGraph()
    for i, int_i in enumerate(inters):
        for k, int_j in enumerate(inters[i+1:]):
            j = i+k+1
            if int_i[0] == int_j[0]:
                print(f'forming edge {i}, {j}     \r', end='')

                vert_info_i = tuple(sorted(int_i))
                vert_info_j = tuple(sorted(int_j))

                if vert_info_i not in _verts:
                    _verts.append(vert_info_i)
                if vert_info_j not in _verts:
                    _verts.append(vert_info_j)

                int_i_ind = _verts.index(vert_info_i)
                int_j_ind = _verts.index(vert_info_j)

                if int_i_ind == int_j_ind:
                    continue

                dist = max(reg_costs[int_i[0]]*np.linalg.norm(centers[i]-centers[j]), 1e-6)

                G.add_edge(int_i_ind, int_j_ind, weight=dist, cost=reg_costs[int_i[0]], ggcs_data=(inters_regions[i], inters_regions[j], int_i, int_j, inters_offsets[i], inters_offsets[j]))
                G.add_edge(int_j_ind, int_i_ind, weight=dist, cost=reg_costs[int_i[0]], ggcs_data=(inters_regions[j], inters_regions[i], int_j, int_i, inters_offsets[j], inters_offsets[i]))
    print()

    return Map(G, regions, inters, inters_regions, inters_offsets, centers, _verts, reg_costs=reg_costs)
