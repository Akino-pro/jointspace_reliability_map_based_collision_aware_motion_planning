import numpy as np

import cvxpy as cp

from scipy.optimize import (
    minimize,
    LinearConstraint,
    NonlinearConstraint,
)

try:
    from cyipopt import minimize_ipopt
except Exception as e:
    print(e)

import numbotics.robot as rob

from poly_funcs import (
    Poly,
    poly_intersect,
    limits_to_poly,
    poly_max_inscribed_ellipse,
    poly_contains_batch,
    poly_contains,
    poly_random_sample,
    poly_empty,
    poly_rem_redundant,
    poly_scale_quick,
    poly_hit_and_run_sample,
)

import pybullet as pyb

import pickle

from itertools import product


POLYS = 300
SHRINK = 0.2
RADIUS = (np.pi/2.0) - 1e-1



class DistanceOracle:
    def __init__():
        self.exclude_coll = {}
        self.n = 0

    def set_q(self, arm_id, q):
        for i in range(self.n):
            pyb.resetJointState(arm_id, i * 2, q[i])

    def __call__(self, q, arm_id, body_index, cmp_body_id, cmp_body_index):
        dist = 100.0
        self.set_q(arm_id, q)

        if cmp_body_id == arm_id:
            if (body_index == cmp_body_index) or (cmp_body_index in self.exclude_coll[body_index]):
                return dist

        c = pyb.getClosestPoints(
            bodyA=arm_id, bodyB=cmp_body_id, linkIndexA=body_index, linkIndexB=cmp_body_index, distance=100.0
        )
        if len(c) > 0:
            dist = c[0][8]
        return dist


class DistanceOracle3R(DistanceOracle):
    def __init__(self):
        self.exclude_coll = {
            0: [1,2,3,4,5,6,7],
            1: [0,2,3,4,5,6,7],
            2: [0,1,3,4,5,6,7],
            3: [0,1,2,4,5,6,7],
            4: [0,1,2,3,5,6,7],
            5: [0,1,2,3,4,6,7],
            6: [0,1,2,3,4,5,7],
            7: [0,1,2,3,4,5,6],
        }
        self.n = 3


class DistanceOracle4R(DistanceOracle):
    def __init__(self):
        self.exclude_coll = {
            0: [1],
            1: [0,2],
            2: [1,3],
            3: [2,4],
            4: [3,5],
            5: [4,6],
            6: [5,7],
            7: [6,8,9,10],
            8: [7,9,10],
            9: [7,8,10],
            10: [7,8,9],
        }
        self.n = 4


class DistanceOracleKinova(DistanceOracle):
    def __init__(self):
        self.exclude_coll = {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1, 3, 4],
            3: [1, 2, 4, 5],
            4: [2, 3, 5, 6],
            5: [3, 4, 6, 7],
            6: [4, 5, 7, 8],
            7: [5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
            8: [6, 7, 9, 10, 11, 12, 13, 14, 15],
            9: [7, 8, 10, 11, 12, 13, 14, 15],
            10: [7, 8, 8, 11, 12, 13, 14, 15],
            11: [7, 8, 9, 10, 12, 13, 14, 15],
            12: [7, 8, 9, 10, 11, 13, 14, 15],
            13: [7, 8, 9, 10, 11, 12, 14, 15],
            14: [7, 8, 9, 10, 11, 12, 13, 15],
            15: [7, 8, 9, 10, 11, 12, 13, 14],
        }
        self.n = 7

    def set_q(self, arm_id, q):
        for i in range(self.n):
            pyb.resetJointState(arm_id, i + 1, q[i])



distance_oracles = {
    '3R' : DistanceOracle3R(),
    '4R' : DistanceOracle4R(),
    'kinova': DistanceOracleKinova(),
}



def new_sep_hyperplane(q, C, d, delta):
    a = ((C.T@C) @ (q - d)) / np.linalg.norm((C.T@C) @ (q - d))
    b = (a @ q) - delta
    return (a, b)


def counter_ex_search(
        q_init,
        C,
        d,
        poly,
        dist_oracle,
        arm_id,
        body_index,
        cmp_body_id,
        cmp_body_index,
        method="ipopt"
):

    def cost(q):
        diff = q - d
        return diff @ ((C.T@C) @ diff)

    dist_const = NonlinearConstraint(
        lambda x: dist_oracle(x, arm_id, body_index, cmp_body_id, cmp_body_index),
        -np.inf,
        -1e-8
    )

    poly_const = LinearConstraint(poly.A, -np.inf, poly.b)

    if method == "ipopt":
        options = {
            "print_level": 1,
            "tol": 1e-6,
            "constr_viol_tol": 1e-8,
            "acceptable_tol": 1e-5,
            "linear_solver": "ma97",
            "max_iter": 500,
        }
        res = minimize_ipopt(
            cost,
            q_init,
            jac=None,
            constraints=(
                dist_const,
                poly_const,
            ),
            options=options,
        )

    elif method == "slsqp":
        res = minimize(cost, q_init, constraints=(dist_const, poly_const))

    else:
        raise Exception(f"unsupported optimization method selected: {method}")

    if res.success:
        return True, res.x
    return False, res.x


def counter_ex_search_poly(C, d, poly):
    x = cp.Variable((poly.A.shape[1],))
    obj = cp.Minimize(cp.quad_form(x - d, C.T@C))
    cons = [poly.A @ x <= poly.b]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.MOSEK)
    return x.value


def check_poly_contains_coll(p, d, dist_oracle, arm_bodies, obs_bodies, samples=100):
    q_sampled = poly_hit_and_run_sample(p, d, samples=samples)
    for q in q_sampled:
        if is_colliding(q, dist_oracle, arm_bodies, obs_bodies):
            return True
    return False


def bounding_box_from_ellipse(C, d):
    aabb_low = []
    aabb_high = []
    Q = np.linalg.inv(C.T@C)
    for i in range(Q.shape[0]):
        qii = Q[i,i]
        aabb_low.append(d[i] - np.sqrt(qii))
        aabb_high.append(d[i] + np.sqrt(qii))
    aabb_low.extend(aabb_high)
    return tuple(aabb_low)


def is_colliding(q, dist_oracle, arm_bodies, obs_bodies):
    for arm_id, body_index in arm_bodies:
        for cmp_body_id, cmp_body_index in arm_bodies + obs_bodies:
            if np.isclose(
                np.clip(dist_oracle(q, arm_id, body_index, cmp_body_id, cmp_body_index), 0.0, np.inf), 0.0
            ):
                return True
    return False


def collision_free_poly(
    p_init,
    q_init,
    dist_oracle,
    arm_bodies,
    obs_bodies,
    cspace_obs=[],
    prob_check=True,
    keep_seed_feasible=False,
    max_iters=100,
    cnt_ex_searches=15,
    max_cs_iters=10000,
    prob_checks=500,
    eps=5e-1,
    delta=5e-2,
    tol=1e-3,
    default_nl_sol="slsqp",
):

    if is_colliding(q_init, dist_oracle, arm_bodies, obs_bodies):
        print("initial configuration in collision")
        return None
    for p_c in cspace_obs:
        if poly_contains(p_c, q_init):
            print("initial configuration inside configration space polytope")
            return None

    n = q_init.shape[0]
    C_i = np.eye(p_init.A.shape[1]) * eps
    d_i = np.copy(q_init)

    for it in range(max_iters):
        p_i = Poly(p_init.A, p_init.b)

        for p_c in cspace_obs:
            p_int = poly_intersect(p_i, p_c)
            if not poly_empty(p_int):
                # print("poly counter example found!")
                q_cs = counter_ex_search_poly(C_i, d_i, p_c)
                a_j, b_j = new_sep_hyperplane(q_cs, C_i, d_i, delta)
                p_hyperplane = Poly(a_j[np.newaxis], np.array([b_j]))
                p_i = poly_intersect(p_i, p_hyperplane)

        q_sampled = poly_hit_and_run_sample(p_i, d_i, samples=max_cs_iters)

        all_bodies = arm_bodies + obs_bodies
        collision_pairs = 0

        for arm_id, body_index in arm_bodies:

            for close_probe in q_sampled:

                all_bodies.remove((arm_id, body_index))
                all_bodies.sort(key=lambda coll_pair: dist_oracle(close_probe, arm_id, body_index, coll_pair[0], coll_pair[1]))

                cs = 0
                for cmp_body_id, cmp_body_index in all_bodies:

                    if arm_id == cmp_body_id:
                        if cmp_body_index in dist_oracle.exclude_coll[body_index]:
                            continue

                    succ, q_cs = counter_ex_search(
                        close_probe,
                        C_i,
                        d_i,
                        p_i,
                        dist_oracle,
                        arm_id,
                        body_index,
                        cmp_body_id,
                        cmp_body_index,
                        method=default_nl_sol,
                    )

                    almost_in_p_i = np.all(
                        np.isclose(np.clip((p_i.A @ q_cs) - p_i.b, 0.0, np.inf), 0.0)
                    )
                    almost_collision = is_colliding(q_cs, dist_oracle, arm_bodies, obs_bodies)

                    if almost_in_p_i and almost_collision:
                        # print(f"found counter example for body {body_i+1}")
                        cs = 0
                        a_j, b_j = new_sep_hyperplane(q_cs, C_i, d_i, delta)
                        p_cs = Poly(a_j[np.newaxis], np.array([b_j]))
                        p_i = poly_intersect(p_i, p_cs)
                    else:
                        cs += 1

                    if cs >= cnt_ex_searches:
                        break

        prev_det = np.linalg.det(np.linalg.inv(C_i))
        C_i, d_i = poly_max_inscribed_ellipse(p_i)
        curr_det = np.linalg.det(C_i)
        C_i = np.linalg.inv(C_i)

        rel_diff = np.abs(curr_det - prev_det) / prev_det
        print(
            f"finishing iteration: {it}, with volume: {curr_det}, relative diff: {rel_diff}"
        )
        if prob_check:
            prob_cert = check_poly_contains_coll(p_i, d_i, dist_oracle, arm_bodies, obs_bodies, samples=prob_checks)
            print(f"prob certified: {prob_cert}")
            if (rel_diff < tol) and (it > 0) and prob_cert:
                break
        else:
            if (rel_diff < tol) and (it > 0):
                break

    return poly_rem_redundant(p_i)


if __name__ == "__main__":

    # arm_name = '3R'
    # arm_name = '4R'
    arm_name = "kinova"

    arm = rob.Robot(f"./arms/{arm_name}.rob")

    # pyb.connect(pyb.GUI)
    pyb.connect(pyb.DIRECT)

    arm_id = pyb.loadURDF(f"./arms/robot_{arm_name}_no_obs.urdf")

    ll = []
    ul = []
    for i in range(arm.n):
        if arm_name != "kinova":
            ll.append(pyb.getJointInfo(arm_id, i * 2)[8])
            ul.append(pyb.getJointInfo(arm_id, i * 2)[9])
        else:
            ll.append(pyb.getJointInfo(arm_id, i + 1)[8])
            ul.append(pyb.getJointInfo(arm_id, i + 1)[9])

    base_polys = []
    ranges = [(-3.2, -1.3), (-1.7, 1.7), (1.3, 3.2)]
    range_combs = product(*[ranges if ((ll[i] <= -np.pi) and (ul[i] >= np.pi)) else ((ll[i], ul[i]),) for i in range(arm.n)])
    for r in range_combs:
        base_ll = []
        base_ul = []
        for bounds in r:
            base_ll.append(bounds[0])
            base_ul.append(bounds[1])
        base_polys.append(limits_to_poly(base_ll, base_ul))

    N = pyb.getNumJoints(arm_id)

    obstacles = []
    obstacles.append(pyb.loadURDF(f"./arms/objects_{arm_name}.urdf"))
    distance_oracle = distance_oracles[arm_name]

    arm_bodies = [(arm_id, i) for i in range(N)]
    obs_bodies = []
    for obs_id in obstacles:
        obs_N = pyb.getNumJoints(obs_id)
        obs_bodies.extend([(obs_id, i) for i in range(obs_N)])


    # p_init = limits_to_poly(ll, ul)

    cspace_polys = []
    c_poly_shrink = []

    # try:
    #     with open(f'./regions/{arm_name}_c_free_polys.pkl', 'rb') as f:
    #         cspace_polys = pickle.load(f)

    #     for c_i in cspace_polys:
    #         c_poly_shrink.append(poly_scale_quick(c_i, 1.0-SHRINK))
    # except:
    #     pass

    print(f'initialized with {len(cspace_polys)} polys')

    seeds = np.random.uniform(ll, ul, (POLYS,arm.n))

    print()
    print(f'using {seeds.shape[0]} seeds!')
    for i, q_init in enumerate(seeds):
        valid_seed = True
        # q_init = np.random.uniform(ll, ul, (arm.n,))
        for p_c in cspace_polys:
            if poly_contains(p_c, q_init):
                print("seed inside existing polytope")
                valid_seed = False
                break
        if not valid_seed:
            continue

        print()
        print("NEW POLYTOPE COMPUTATION!!!")
        print("###########################")
        print()

        # p_init = poly_within_radius(ll, ul, q_init, RADIUS)
        for p_i in base_polys:
            if poly_contains(p_i, q_init):
                p_init = p_i
                break

        try:
            p_new = collision_free_poly(
                p_init,
                q_init,
                distance_oracle,
                arm_bodies,
                obs_bodies,
                cspace_obs=c_poly_shrink,
                keep_seed_feasible=True,
            )
            if p_new is not None:
                cspace_polys.append(p_new)
                c_poly_shrink.append(poly_scale_quick(p_new, 1.0 - SHRINK))

        except ValueError as e:
            print(e)
            continue

    # with open(f"./regions/{arm_name}_c_free_polys.pkl", "wb") as f:
    #     pickle.dump(cspace_polys, f)
