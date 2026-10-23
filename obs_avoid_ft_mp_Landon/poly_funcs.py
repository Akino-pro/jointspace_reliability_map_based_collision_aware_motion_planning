import numpy as np

import cvxpy as cp

from scipy.spatial import ConvexHull, Delaunay, HalfspaceIntersection
from scipy.special import gamma


class Poly():
    def __init__(self, A, b):
        self._A = np.copy(A)
        self._b = np.copy(b)

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def n(self):
        return self._A.shape[1]


def poly_intersect(p1, p2):
    (A1, b1, A2, b2) = (p1.A, p1.b, p2.A, p2.b)
    A_u = np.concatenate((A1, A2), axis=0)
    b_u = np.concatenate((b1, b2), axis=0)
    return Poly(A_u, b_u)


def poly_rem_redundant(p):
    (A, b) = (p.A, p.b)
    if poly_empty(p):
        return Poly(A, b)
    n = p.n
    x = cp.Variable((n,))
    A_par = cp.Parameter(A.shape)
    A_par.value = np.copy(A)
    b_par = cp.Parameter(b.shape)
    b_par.value = np.copy(b)
    a_T_par = cp.Parameter(n)
    a_T_par.value = np.copy(A[0])

    obj = cp.Maximize(a_T_par@x)
    cons = [A_par@x <= b_par]
    prob = cp.Problem(obj, cons)

    keep_inds = []

    for i in range(A.shape[0]):
        a_T_par.value = np.copy(A[i])
        A_par.value[i] = 0.0
        b_par.value[i] = 0.0

        keep = True
        try:
            prob.solve(solver=cp.MOSEK)
            if not (prob.status == 'unbounded'):
                if prob.objective.value < b[i]:
                    keep = False
        except:
            pass
        if keep:
            keep_inds.append(i)
            A_par.value[i] = np.copy(A[i])
            b_par.value[i] = np.copy(b[i])

    return Poly(A_par.value[keep_inds], b_par.value[keep_inds])


def poly_empty(p):
    (A, b) = (p.A, p.b)
    x = cp.Variable((p.n,))
    prob = cp.Problem(cp.Minimize(0), [A@x<=b])
    try:
        prob.solve(solver=cp.MOSEK)
        return (prob.status == cp.INFEASIBLE)
    except:
        return True


def poly_vol(p):
    if poly_empty(p):
        return 0.0
    try:
        vol = ConvexHull(poly_to_vert(p)).volume
        return vol
    except:
        return 0.0


def poly_vol_est(p):
    (A, b) = (p.A, p.b)
    n = p.n
    C, d = poly_max_inscribed_ellipse(p)
    fac = (np.pi**(float(n)/2.0))/gamma((float(n)/2.0)+1)
    return fac*np.linalg.det(C)


def poly_vol_voxel(p, samples=5):
    n = p.n
    p_aabb = poly_aabb(p)
    lbs, ubs = [], []
    box_vol = 1.0
    for lb, ub in zip(p_aabb[:n], p_aabb[n:]):
        lbs.append(lb)
        ubs.append(ub)
        box_vol *= (ub-lb)
    grids = [np.linspace(lbs[i], ubs[i], samples) for i in range(n)]
    mesh = np.meshgrid(*grids, indexing='ij')
    xs = np.vstack([m.ravel() for m in mesh]).T
    cont = poly_contains_batch(p, xs)
    print(cont)
    xs_cont = xs[cont]
    return box_vol*(float(xs_cont.shape[0])/float(xs.shape[0]))


def poly_vol_mc(p, lbs, ubs, samples=100000):
    n = p.n
    xs = np.random.uniform(lbs, ubs, (samples,n))
    box_vol = 1.0
    for lb, ub in zip(lbs, ubs):
        box_vol *= (ub-lb)
    cont = poly_contains_batch(p, xs)
    xs_cont = xs[cont]
    return box_vol*(float(xs_cont.shape[0])/float(xs.shape[0]))


def poly_intersect_vol(p1, p2):
    p_int = poly_intersect(p1, p2)
    try:
        return poly_vol(p_int)
    except:
        return 0.0


def poly_scale(p, scale):
    (A, b) = (p.A, p.b)
    C_k, d_k = poly_max_inscribed_ellipse(p)
    b_k = scale*(b - (A@d_k)) + (A@d_k)
    return Poly(A, b_k)


def poly_scale_quick(p, scale):
    (A, b) = (p.A, p.b)
    center = poly_cheb_center(p)
    b_k = scale*(b - (A@center)) + (A@center)
    return Poly(A, b_k)


def poly_scale_from_center(p, center, scale):
    (A, b) = (p.A, p.b)
    b_k = scale*(b - (A@center)) + (A@center)
    return Poly(A, b_k)


def poly_cheb_center(p):
    (A, b) = (p.A, p.b)
    y = cp.Variable((1,), nonneg=True)
    x = cp.Variable((p.n,))
    obj = cp.Maximize(y)
    cons = [(A@x) + (y*np.linalg.norm(A,axis=1)) <= b]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.MOSEK)
    if prob.status == cp.OPTIMAL:
        return x.value
    return None


def poly_max_inscribed_ellipse(p):
    (A, b) = (p.A, p.b)
    n = p.n
    C = cp.Variable((n,n), symmetric=True)
    d = cp.Variable((n,))
    constraints = [C >> 0]
    constraints += [cp.norm(A@C, axis=1) + (A@d) <= b]
    prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
    prob.solve(solver=cp.MOSEK)
    return C.value, d.value


def poly_max_inscribed_ellipse_w_point(p, x):
    (A, b) = (p.A, p.b)
    n = p.n
    C = cp.Variable((n,n), symmetric=True)
    d = cp.Variable((n,))
    constraints = [C >> 0]
    constraints += [cp.norm(A@C, axis=1) + (A@d) <= b]
    constraints += [cp.quad_form(x-d, C)]
    obj = cp.Minimize(cp.quad_form(x - d, C))
    prob = cp.Problem(cp.Minimize(-cp.atoms.log_det(C)), constraints)
    prob.solve(solver=cp.MOSEK)
    return C.value


def poly_max_inscribed_aa_ellipse(p):
    (A, b) = (p.A, p.b)
    n = p.n
    C = cp.Variable((n,), nonneg=True)
    d = cp.Variable((n,))
    constraints = [cp.norm(A@cp.diag(C), axis=1) + (A@d) <= b]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.log(C))), constraints)
    prob.solve(solver=cp.MOSEK)
    return C.value, d.value


def vert_to_poly(v):
    cvx_hull = ConvexHull(v)
    Ab = cvx_hull.equations
    return Poly(Ab[:,:-1], -Ab[:,-1])


def poly_contains(p, x):
    (A, b) = (p.A, p.b)
    return np.all(A@x <= b)


def poly_to_vert(p):
    (A, b) = (p.A, p.b)
    if poly_empty(p):
        return np.zeros((0,A.shape[1]))
    center = poly_cheb_center(p)
    try:
        Ab = np.concatenate((A,-b[:,np.newaxis]),axis=-1)
        sp_hp = HalfspaceIntersection(Ab, center)
        return sp_hp.intersections
    except:
        return np.zeros((0,A.shape[1]))


def poly_aabb(p):
    aabb = []
    v = poly_to_vert(p).T
    for v_i in v:
        aabb.append(np.amin(v_i))
    for v_i in v:
        aabb.append(np.amax(v_i))
    return tuple(aabb)


def poly_contains_sphere(p, rad):
    (A, b) = (p.A, p.b)
    x = cp.Variable((A.shape[1],))
    cons = [(A@x) + (rad*np.linalg.norm(A,axis=1)) <= b]
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver=cp.MOSEK)
    return not (prob.status == cp.INFEASIBLE)


def poly_translate(p, d):
    return Poly(p.A, p.b + (p.A@d))


def poly_feasible(p, thresh):
    n = p.n
    if n > 5:
        try:
            return poly_vol_est(p) > thresh
        except:
            return False
    else:
        return poly_vol(p) > thresh


def limits_to_poly(l_bnd, u_bnd):
    n = len(l_bnd)
    A_box = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    b_box = np.concatenate((np.ones((n))*u_bnd, -np.ones((n))*l_bnd), axis=0)
    return Poly(A_box, b_box)


def poly_contains_batch(p, xs):
    Ax = np.einsum('ij,Bj->Bi', p.A, xs)
    Ax_b = Ax <= p.b
    return np.all(Ax_b, axis=1)


def poly_random_sample(p, max_samples=100):
    aabb = poly_aabb(p)
    n = p.A.shape[1]
    for _ in range(max_samples):
        x = np.random.uniform(aabb[:n], aabb[n:], (n,))
        if poly_contains(p, x):
            return x
    return None


def poly_hit_and_run_sample(p, d, samples=100):
    n = p.A.shape[1]
    A = p.A
    b = p.b
    res = np.zeros((1,n))
    res[0] = d

    if not poly_contains(p, d):
        return None

    def lam_min_max(A, b, x, u):
        lams = (b - A@x) / (A@u)
        lams_pos = lams[lams > 0]
        lams_neg = lams[lams < 0]

        lam_max = np.min(lams_pos)
        lam_min = np.max(lams_neg)

        return lam_min, lam_max

    for _ in range(samples-1):
        x_i = res[-1]
        u = np.random.randn(n)
        u /= np.linalg.norm(u)

        lam_min, lam_max = lam_min_max(A, b, x_i, u)

        lam = np.random.uniform(lam_min, lam_max)

        x_prime = x_i + lam * u
        res = np.concatenate((res, x_prime[np.newaxis]), axis=0)

    return res





