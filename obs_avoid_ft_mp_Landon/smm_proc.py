import numpy as np

from scipy.optimize import minimize, LinearConstraint

import numbotics.topology as top

from poly_funcs import poly_contains



def union_intervals(a):
    b = []
    for begin, end in sorted(a):
        if b and b[-1][1] >= begin - 1:
            b[-1][1] = max(b[-1][1], end)
        else:
            b.append([begin, end])
    return b


def smm_to_bb(smm, gap_factor=5):
    bounds = []
    for dim in range(smm.shape[1]):
        sorted_dim_smm = np.sort(smm[:, dim])
        diffs = np.diff(sorted_dim_smm)

        max_gap_index = np.argmax(diffs)
        max_gap_value = diffs[max_gap_index]

        average_gap = np.mean(diffs)
        if max_gap_value > gap_factor * average_gap:
            bound1 = [sorted_dim_smm[0], sorted_dim_smm[max_gap_index]]
            bound2 = [sorted_dim_smm[max_gap_index + 1], sorted_dim_smm[-1]]
            bounds.append([bound1, bound2])
        else:
            bounds.append([[sorted_dim_smm[0], sorted_dim_smm[-1]]])

    return bounds


def cut_smm(smm, valid, valid_percentage=0.92, min_len=0.02):
    B, n = smm.shape
    threshold = int(np.ceil(valid_percentage * B))
    min_thresh = int(np.ceil(min_len * B))
    cuts = []
    start_index = 0

    while start_index < B:
        while start_index < B and not valid[start_index]:
            start_index += 1

        if start_index >= B:
            break

        end_index = start_index + 1
        while end_index - start_index < threshold and end_index < B:
            if not valid[end_index]:
                valid_run_length = np.sum(valid[start_index:end_index])
                if valid_run_length >= valid_percentage * (end_index - start_index):
                    break
            end_index += 1

        if (np.sum(valid[start_index:end_index]) >= valid_percentage * (end_index - start_index)) and (end_index-start_index > min_thresh):
            cuts.append(smm[start_index:end_index])

        start_index = end_index

    return cuts


def project_smm_to_constraint(arm, smms, x_d, failed_j, q_fail, polytopes=[]):

    def obj(q):
        arm.q = q
        return np.linalg.norm(arm.fk_err(x_d, mask=True))

    A = np.zeros((1,arm.n))
    A[0, failed_j] = 1.0
    projection = LinearConstraint(
        A,
        q_fail,
        q_fail
    )

    qs = []

    for smm in smms:
        for q in smm:

            res = minimize(obj, q, method='slsqp', constraints=(projection,))
            q_opt = res.x

            for poly in polytopes:
                if poly_contains(poly, q_opt):
                    break
            else:
                if len(polytopes) > 0:
                    continue

            if (obj(q_opt) < 1e-6) and res.success:
                for q_prev in qs:
                    if np.linalg.norm(top.tr_inv(top.tr(q_prev)/top.tr(q_opt))) < 1e-3:
                        break
                else:
                    q_opt = top.tr_inv(top.tr(q_opt))
                    q_opt[failed_j] = q_fail
                    qs.append(q_opt)

    return np.array(qs)
