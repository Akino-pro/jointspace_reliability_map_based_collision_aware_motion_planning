import numpy as np

import networkx as nx

import cvxpy as cp

import numbotics.robot as rob

from rtree import index

from poly_funcs import *

from spp_ggcs import GGCS

from mapper import Map


def poly_center_w_eq(p, eq):
    (A, b) = (p.A, p.b)
    y = cp.Variable((1,), nonneg=True)
    x = cp.Variable((p.n,))
    obj = cp.Maximize(y)
    cons = [(A@x) + (y*np.linalg.norm(A,axis=1)) <= b]
    cons += [x[eq[0]] == eq[1]]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.MOSEK)
        if prob.status == cp.OPTIMAL:
            return x.value
    except:
        pass
    return None


def build_coll_free_map(arm, regions, equality=None):

    reg_costs = [1.0 for _ in range(len(regions))]

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
            print(f'c free regions: {i}, {j}     \r', end='')
            if i == j:
                continue

            p_int = poly_intersect(regions[i], poly_translate(regions[j], offsets[k]))
            if equality is None:
                d = poly_cheb_center(p_int)
            else:
                d = poly_center_w_eq(p_int, equality)
            if d is not None:
                inters.append((i,j))
                inters_regions.append(p_int)
                inters_offsets.append(offsets[k])
                centers.append(d)

    G = nx.DiGraph()
    for i, int_i in enumerate(inters):
        for k, int_j in enumerate(inters[i+1:]):
            j = i+k+1
            if int_i[0] == int_j[0]:
                print(f'forming edge {i}, {j}    \r', end='')

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

    return Map(G, regions, inters, inters_regions, inters_offsets, centers, _verts)
