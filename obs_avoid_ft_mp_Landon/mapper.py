import numpy as np

import networkx as nx
# import igraph as ig

from poly_funcs import (
    poly_contains,
    poly_intersect,
    poly_empty
)

from graph_helper import (
    random_path,
    random_path_multiple_goals,
    path_length,
    get_path_edges,
    phi_rounded_path,
    phi_rounded_path_multiple_goals,
)

from constraints import (
    LinearInequality,
    ConstrainedLinearEquality,
    Point,
)

import time
from itertools import combinations
import pickle


class Map():
    def __init__(self, graph, regions, inters, inters_regions, inters_offsets, centers, _verts, reg_costs=None):
        self.graph = graph
        self.regions = regions
        self.inters = inters
        self.inters_regions = inters_regions
        self.inters_offsets = inters_offsets
        self.centers = centers
        self._verts = _verts

        self.reg_costs = reg_costs

        self.s_ind = None
        self.g_inds = None


    def add_start_goals(self, start, goals, verbose=False):

        for r_i in self.regions:
            if poly_contains(r_i, start):
                break
        else:
            raise ValueError(f'start configuration: {start} is not contained in any valid region')

        for goal in goals:
            for r_i in self.regions:
                if poly_contains(r_i, goal):
                    break
            else:
                raise ValueError(f'goal configuration: {goal} is not contained in any valid region')

        if self.s_ind or self.g_inds:
            self.remove_start_goal()

        s_ind = len(self._verts)
        g_inds = [s_ind + 1 + i for i in range(len(goals))]

        n_zero = np.zeros((self.regions[0].n,))
        if self.reg_costs is None:
            reg_costs = [1.0 for _ in range(len(self.regions))]
        else:
            reg_costs = self.reg_costs

        self.centers.append(start)
        for goal in goals:
            self.centers.append(goal)

        for i, r_i in enumerate(self.regions):
            if poly_contains(r_i, start):
                for (inter, inter_region, inter_offset, inter_center) in zip(self.inters, self.inters_regions, self.inters_offsets, self.centers):
                    if (inter[0] == i) or (inter[1] == i):
                        vert_info = tuple(sorted(inter))
                        int_ind = self._verts.index(vert_info)

                        dist = max(reg_costs[i]*np.linalg.norm(start-inter_center), 1e-6)
                        self.graph.add_edge(s_ind, int_ind, weight=dist, cost=reg_costs[i], ggcs_data=(start, inter_region, s_ind, inter, n_zero, inter_offset))

            for goal, g_ind in zip(goals, g_inds):
                if poly_contains(r_i, goal):
                    for (inter, inter_region, inter_offset, inter_center) in zip(self.inters, self.inters_regions, self.inters_offsets, self.centers):
                        if (inter[0] == i) or (inter[1] == i):
                            vert_info = tuple(sorted(inter))
                            int_ind = self._verts.index(vert_info)

                            dist = max(reg_costs[i]*np.linalg.norm(inter_center-goal), 1e-6)
                            self.graph.add_edge(int_ind, g_ind, weight=dist, cost=reg_costs[i], ggcs_data=(inter_region, goal, inter, g_ind, inter_offset, n_zero))

        self.s_ind = s_ind
        self.g_inds = g_inds


    def remove_start_goals(self):
        if self.g_inds is not None:
            for g_ind in self.g_inds[::-1]:
                del self.centers[g_ind]
                self.graph.remove_node(g_ind)
            self.g_inds = None

        if self.s_ind is not None:
            del self.centers[self.s_ind]
            self.graph.remove_node(self.s_ind)
            self.s_ind = None


    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            m = pickle.load(f)
        return m
