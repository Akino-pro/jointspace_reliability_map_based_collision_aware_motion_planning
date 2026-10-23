import numpy as np

import heapq

import time

import networkx as nx

import multiprocessing as mp
import os


def get_path_edges(p):
    return [e for e in zip(p[:-1], p[1:])]


def path_length(G, p):
    p_len = 0.0
    for e in p:
        p_len += G[e[0]][e[1]]['weight']
    return p_len


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x/np.sum(exp_x)


def random_path_multiple_goals(G, s, ts, v_to_t, e_set, cost_scale=0.9, max_cost=np.inf):

    p = []
    verts = {s}
    u = s
    v = None

    cost = 0.0

    while v not in ts:

        neighbors = [n for n in G.neighbors(u) if (n not in verts) and ((v, u) not in e_set)]
        neighbors = [n for n in neighbors if cost + cost_scale*(G[u][n]['weight'] + v_to_t[n]) < max_cost]
        if len(neighbors) == 0:
            return None
        n_weights = np.array([-(G[u][n]['weight'] + v_to_t[n]) for n in neighbors])

        n_w_norm = np.cumsum(softmax(n_weights))

        c = np.searchsorted(n_w_norm, np.random.uniform(0,1))

        v = neighbors[c]
        verts.add(v)

        cost += cost_scale*G[u][v]['weight']

        p.append((u,v))

        u = v

    if len(set(ts).intersection(verts)) == 0:
        return None

    return tuple(p)


def random_path(G, s, t, v_to_t, e_set, cost_scale=0.9, max_cost=np.inf):

    p = []
    verts = {s}
    u = s
    v = None

    cost = 0.0

    while v != t:

        neighbors = [n for n in G.neighbors(u) if (n not in verts) and ((v, u) not in e_set)]
        neighbors = [n for n in neighbors if cost + cost_scale*(G[u][n]['weight'] + v_to_t[n]) < max_cost]
        if len(neighbors) == 0:
            return None
        n_weights = np.array([-(G[u][n]['weight'] + v_to_t[n]) for n in neighbors])

        n_w_norm = np.cumsum(softmax(n_weights))

        c = np.searchsorted(n_w_norm, np.random.uniform(0,1))

        v = neighbors[c]
        verts.add(v)

        cost += cost_scale*G[u][v]['weight']

        p.append((u,v))

        u = v

    if t not in verts:
        return None

    return tuple(p)


def random_rev_path(G, s, t, v_to_t, e_set, cost_scale=0.1, max_cost=np.inf):

    p = []
    verts = {t}
    u = t
    v = None

    cost = 0.0

    while v != s:

        neighbors = [n for n in G.neighbors(u) if (n not in verts) and ((u, v) not in e_set)]
        neighbors = [n for n in neighbors if cost + cost_scale*(G[u][n]['weight'] + v_to_t[n]) < max_cost]
        if len(neighbors) == 0:
            return None
        n_weights = np.array([-(G[u][n]['weight'] + v_to_t[n]) for n in neighbors])

        n_w_norm = np.cumsum(softmax(n_weights))

        c = np.searchsorted(n_w_norm, np.random.uniform(0,1))

        v = neighbors[c]
        verts.add(v)

        cost += cost_scale*G[u][v]['weight']

        p.append((v,u))

        u = v


    if s not in verts:
        return None

    return tuple(p)


def phi_rounded_path(G, phis, s, t):

    p = []
    verts = {s}
    u = s
    v = None

    while v != t:

        neighbors = [n for n in G.neighbors(u) if n not in verts]
        if len(neighbors) == 0:
            return None
        n_weights = np.array([phis[(u,n)].value for n in neighbors])

        n_w_norm = np.cumsum(n_weights)

        c = np.searchsorted(n_w_norm, np.random.uniform(0,np.amax(n_w_norm)-1e-8))

        v = neighbors[c]
        verts.add(v)

        p.append((u,v))

        u = v

    if t not in verts:
        return None

    return tuple(p)


def phi_rounded_path_multiple_goals(G, phis, s, ts):

    p = []
    verts = {s}
    u = s
    v = None

    while v not in ts:

        neighbors = [n for n in G.neighbors(u) if n not in verts]
        if len(neighbors) == 0:
            return None
        n_weights = np.array([phis[(u,n)].value for n in neighbors])

        n_w_norm = np.cumsum(n_weights)

        c = np.searchsorted(n_w_norm, np.random.uniform(0,np.amax(n_w_norm)-1e-8))

        v = neighbors[c]
        verts.add(v)

        p.append((u,v))

        u = v

    if len(set(ts).intersection(verts)) == 0:
        return None

    return tuple(p)


def update_edge_costs(graph, costs, x_s, len_weight=1.0, cost_weight=1.0):
    for i, e in enumerate(graph.edges()):
        u = x_s[e[0]]
        v = x_s[e[1]]
        h, k = costs[e]
        w = (len_weight + (cost_weight*(0.5*h@(u + v)) + k))*np.linalg.norm(u-v)
        graph.add_edge(e[0],e[1],weight=max(w,1e-8))


def pre_proc_graph_multiple_goals(graph, s_ind, g_inds, cost_scale, max_paths=1000, verbose=False):
    mp_G = nx.DiGraph()
    G_rev = graph.reverse()
    new_edges = set()

    try:
        start_time = time.perf_counter_ns()
        v_to_t = nx.multi_source_dijkstra_path_length(G_rev, g_inds)
        p_short_len = np.inf
        g_short_len = g_inds[0]
        for g_ind in g_inds:
            p_len = v_to_t[g_ind]
            if p_len < p_short_len:
                p_short_len = p_len
                g_shortest = g_ind
        p_shortest = nx.shortest_path(graph, s_ind, g_shortest, weight='weight')
        paths = 0
        for i in range(len(p_shortest)-1):
            new_edges.add((p_shortest[i], p_shortest[i+1]))
        s_len = path_length(graph, get_path_edges(p_shortest))
        if verbose:
            print('lowest fixed path cost:', s_len)
        for i in range(max_paths):
            p_i = random_path_multiple_goals(graph, s_ind, g_inds, v_to_t, set(), cost_scale=cost_scale, max_cost=s_len)
            if p_i is not None:
                paths += 1
                for e_i in p_i:
                    if e_i not in new_edges:
                        new_edges.add(e_i)
            if verbose:
                print(f'found {paths} shortest paths  \r', end='')
        end_time = time.perf_counter_ns()

        for e in new_edges:
            mp_G.add_edge(e[0],e[1], **graph.get_edge_data(e[0], e[1]))

        if verbose:
            print()
            return mp_G, (end_time-start_time)/1e9
        return mp_G

    except:
        raise Exception




if __name__ == '__main__':

    ITERS = 5000

    A = np.random.uniform(0,1.0,(2000,2000))
    A = A.round()*A

    G = nx.DiGraph(A)
    min_G = nx.DiGraph(A*0.1)
    E = G.edges()

    s = 0
    t = 100

    s_len = nx.shortest_path_length(G, s, t, weight='weight')
    print('shortest path length:', s_len)

    curr_s_len = np.inf
    valid_paths = 0

    s_bf_time = time.perf_counter_ns()/1e9
    v_to_t = nx.single_source_bellman_ford_path_length(G, t)
    min_v_to_t = nx.single_source_bellman_ford_path_length(min_G, t)
    e_bf_time = time.perf_counter_ns()/1e9
    print('time to compute shortest distance to t:', e_bf_time-s_bf_time)

    args = [('G', s, t, 'v_to_t', 'min_v_to_t', 25.0*s_len) for _ in range(ITERS)]
    pool = mp.Pool(os.cpu_count())

    s_time = time.perf_counter_ns()/1e9

    paths = set()
    for p in pool.imap_unordered(rand_path_pool, args):
        if p is not None:
            paths.add(p)
            p_len = path_length(E, p)
            if p_len < curr_s_len:
                curr_s_len = p_len
                print('newest shortest path:', curr_s_len)

    print('valid paths:', len(paths))

    e_time = time.perf_counter_ns()/1e9
    print('runtime:', e_time-s_time)

    # for _ in range(ITERS):
    #     p = random_path(G, s, t, v_to_t, max_len=50)
    #     if p is not None:
    #         valid_paths += 1
    #         p_len = path_length(E, p)
    #         if p_len < curr_s_len:
    #             curr_s_len = p_len
    #             print('newest shortest path:', curr_s_len)
    # print('valid paths:', valid_paths)
