import numpy as np

import cvxpy as cp

import networkx as nx

from graph_helper import get_path_edges
from poly_funcs import poly_cheb_center


class GGCS():
    def __init__(self,
                 map_obj,
                 graph,
                 v_info,
                 remove_one_cycle=True,
                 equality_cons=None,
                 integer_phi=False): # will not support integer phi yet

        self.graph = graph

        self.centers = map_obj.centers

        self.source = map_obj.s_ind
        self.targets = map_obj.g_inds

        self.v_cnt = v_info[0]
        self.v_dim = v_info[1]

        self.edges = list(self.graph.edges())
        self.z = dict()
        self.zp = dict()
        self.phi = dict()
        self.l = dict()

        self.objective = 0
        self.consts = []

        source_phis = []
        target_phis = []
        source_in_phis = []
        target_out_phis = []

        for _e in self.graph.edges(data=True):

            e = (_e[0], _e[1])

            self.z[e] = cp.Variable((self.v_dim,))
            self.zp[e] = cp.Variable((self.v_dim,))
            self.phi[e] = cp.Variable(integer=integer_phi)
            self.l[e] = cp.Variable(nonneg=True)

            dist = _e[2]['cost'] * cp.norm(self.z[e]-self.zp[e])
            self.consts += [self.l[e] >= dist]
            self.consts += [self.phi[e] <= 1.0]

            p, p_prime, _, _, _, _ = _e[2]['ggcs_data'] # self.edge_verts[e]

            if e[0] == self.source:
                source_phis += [self.phi[e]]
                self.consts += [self.z[e] == p*self.phi[e]]
            elif e[1] == self.source:
                source_in_phis += [self.phi[e]]
                self.consts += [self.zp[e] == p_prime*self.phi[e]]
            else:
                self.consts += [p.A @ self.z[e] <= p.b*self.phi[e]]

            if e[1] in self.targets:
                target_phis += [self.phi[e]]
                self.consts += [self.zp[e] == p_prime*self.phi[e]]
            elif e[0] in self.targets:
                target_out_phis += [self.phi[e]]
                self.consts += [self.z[e] == p*self.phi[e]]
            else:
                self.consts += [p_prime.A @ self.zp[e] <= p_prime.b*self.phi[e]]

        self.objective = cp.sum([l for l in self.l.values()])

        self.consts += [cp.sum(source_phis) == 1.0]
        self.consts += [cp.sum(target_phis) == 1.0]

        self.consts += [cp.sum(source_in_phis) == 0.0]
        self.consts += [cp.sum(target_out_phis) == 0.0]

        if remove_one_cycle:
            for v in list(self.graph.nodes()):
                if (v is self.source) or (v in self.targets):
                    continue

                v_in_ids = list(self.graph.predecessors(v))
                v_out_ids = list(self.graph.successors(v))

                checks = [i for i in v_in_ids if i in v_out_ids and i is not self.source and i not in self.targets]
                checks.extend([i for i in v_out_ids if i in v_in_ids and i is not self.source and i not in self.targets])
                checks = list(set(checks))

                for w in checks:
                    w_out_ids = list(self.graph.successors(w))

                    v_flow = cp.sum([self.phi[(v,i)] for i in v_out_ids])
                    w_flow = cp.sum([self.phi[(w,i)] for i in w_out_ids])

                    self.consts += [self.phi[(v,w)] + self.phi[(w,v)] <= v_flow]
                    self.consts += [self.phi[(v,w)] + self.phi[(w,v)] <= w_flow]

        for v in list(self.graph.nodes()):
            in_ids = list(self.graph.predecessors(v))
            out_ids = list(self.graph.successors(v))

            if ((len(in_ids) > 0) or (len(out_ids) > 0)) and ((v != self.source) and (v not in self.targets)):

                if len(in_ids) == 0:
                    in_flow = 0.0
                    in_spt_flow = np.zeros((self.v_dim,))
                else:
                    in_flow = cp.sum([self.phi[(i,v)] for i in in_ids])
                    in_spt_flow = []

                    for i in in_ids:

                        _, _, _, ind_j, _, off_j = graph[i][v]['ggcs_data'] # self.edge_verts[(i,v)]

                        offset = np.zeros((self.v_dim,)) if ind_j[0] < ind_j[1] else off_j
                        in_spt_flow.append(self.zp[(i,v)] - (offset*self.phi[(i,v)]))

                    in_spt_flow = cp.sum(in_spt_flow)

                if len(out_ids) == 0:
                    out_flow = 0.0
                    out_spt_flow = np.zeros((self.v_dim,))
                else:
                    out_flow = cp.sum([self.phi[(v,i)] for i in out_ids])
                    out_spt_flow = []

                    for i in out_ids:

                        _, _, ind_i, _, off_i, _ = graph[v][i]['ggcs_data'] # self.edge_verts[(v,i)]

                        offset = np.zeros((self.v_dim,)) if ind_i[0] < ind_i[1] else off_i
                        out_spt_flow.append(self.z[(v,i)] - (offset*self.phi[(v,i)]))

                    out_spt_flow = cp.sum(out_spt_flow)


                self.consts += [in_flow == out_flow]
                self.consts += [in_spt_flow == out_spt_flow]
                self.consts += [out_flow <= 1.0]

            if equality_cons is not None:
                j_eq = equality_cons[0]
                j_val = equality_cons[1]
                for e in self.edges:
                    self.consts += [self.z[e][j_eq] == j_val*self.phi[e]]
                    self.consts += [self.zp[e][j_eq] == j_val*self.phi[e]]


    def solve(self, verbose=True):
        prob = cp.Problem(cp.Minimize(self.objective), self.consts)
        prob.solve(solver=cp.MOSEK, verbose=verbose)

        if verbose:
            print('gcs solve time:', prob.solver_stats.solve_time)

        if prob.value == np.inf:
            print('problem is infeasible')
            return None

        x = np.zeros((self.v_cnt,self.v_dim))
        for v in range(self.v_cnt):
            if ((v == self.source) or (v in self.targets)):
                x[v] = self.centers[v]

            else:
                try:
                    out_ids = list(self.graph.successors(v))
                    x[v] = sum([self.z[(v,i)].value for i in out_ids])

                    x[v] += (1.0 - sum([self.phi[(v,i)].value for i in out_ids]))*self.centers[v]
                except:
                    x[v] = self.centers[v]

        if verbose:
            return (x, prob.solver_stats.solve_time)
        return x


    def solve_single_path(self,
                          path,
                          basic_solver=True,
                          equality_cons=None,
                          verbose=False):

        x_opt = dict()
        xp_opt = dict()
        l_opt = dict()

        consts_opt = []

        for i, e in enumerate(path):

            x_opt[e] = cp.Variable((self.v_dim,))
            xp_opt[e] = cp.Variable((self.v_dim,))
            l_opt[e] = cp.Variable(nonneg=True)

            dist = self.graph[e[0]][e[1]]['cost'] * cp.norm(x_opt[e]-xp_opt[e])
            consts_opt += [l_opt[e] >= dist]

            p, p_prime, _, _, _, _ = self.graph[e[0]][e[1]]['ggcs_data']

            if e[0] == self.source:
                consts_opt += [x_opt[e] == p]
            elif e[1] == self.source:
                consts_opt += [xp_opt[e] == p_prime]
            else:
                consts_opt += [p.A @ x_opt[e] <= p.b]

            if e[1] in self.targets :
                consts_opt += [xp_opt[e] == p_prime]
            elif e[0] in self.targets:
                consts_opt += [x_opt[e] == p]
            else:
                consts_opt += [p_prime.A @ xp_opt[e] <= p_prime.b]

        for e0, e1 in zip(path[:-1], path[1:]):

            _, _, _, ind_j, _, off_j = self.graph[e0[0]][e0[1]]['ggcs_data']
            _, _, ind_i, _, off_i, _ = self.graph[e1[0]][e1[1]]['ggcs_data']

            offset_j = np.zeros((self.v_dim,)) if ind_j[0] < ind_j[1] else off_j
            offset_i = np.zeros((self.v_dim,)) if ind_i[0] < ind_i[1] else off_i

            consts_opt += [(xp_opt[e0]-offset_j) == (x_opt[e1]-offset_i)]

        if equality_cons is not None:
            j_eq = equality_cons[0]
            j_val = equality_cons[1]
            for e in path:
                consts_opt += [x_opt[e][j_eq] == j_val]
                consts_opt += [xp_opt[e][j_eq] == j_val]

        obj_opt = cp.sum([l for l in l_opt.values()])

        prob = cp.Problem(cp.Minimize(obj_opt), consts_opt)
        if basic_solver == True:
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.MOSEK, verbose=verbose, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3,
                                                                       'MSK_IPAR_INTPNT_SOLVE_FORM': 1,
                                                                       'MSK_DPAR_MIO_TOL_REL_GAP': 1e-3})

        for e in path:
            x_opt[e] = x_opt[e].value
            xp_opt[e] = xp_opt[e].value

        if verbose:
            print('path solve time:', prob.solver_stats.solve_time)

        if verbose:
            return x_opt, xp_opt, prob.solver_stats.solve_time
        return x_opt, xp_opt
