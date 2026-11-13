"""Optional direct SOCP backend for cached affine-slack / OPD-norm models.

Clarabel uses Ax+s=b. Its status is checked along with independently evaluated
model constraints; the caller must still re-trace every physical candidate.
"""
import importlib.util

import numpy as np
from scipy import sparse


def available():
    return importlib.util.find_spec('clarabel') is not None


def unsupported(model):
    if model.use_cones and hasattr(model.owner, 'order_scale'):
        for key, vector in [('wfe_order', 'wave_vector'), ('spot_order', 'spot_vector')]:
            if key in model.keys and vector in model.vectors:
                return 'Differences of residual norms are not convex SOCP constraints'
    return None


def solve_proposal(model, low, high, tolerance, scales, selected, phase,
                   variant, damping, hard_rows):
    import clarabel

    reason = unsupported(model)
    if reason:
        raise ValueError(reason + '; use subproblem_backend="slsqp" or "auto"')
    n, k = len(low), len(model.keys)
    modes = model.modes
    total = n + k + 1 if phase == 'feasibility' else n + 1
    scalar_m, scalar_c, cones, blocks, rhs = [], [], [], [], []

    def linear(coefficients, constant):
        scalar_m.append(np.atleast_2d(coefficients))
        scalar_c.append(np.atleast_1d(constant))

    def norm_bound(B, b, offset, upper):
        # offset + upper@x >= ||B@x[:n] + b||, represented without linearizing the norm.
        A = np.zeros((len(b) + 1, total))
        A[0] = -upper
        A[1:, :n] = -B
        blocks.append(sparse.csc_matrix(A))
        rhs.append(np.r_[offset, b])
        cones.append(clarabel.SecondOrderConeT(len(b) + 1))

    bounds = np.zeros((2 * n, total))
    bounds[:n, :n], bounds[n:, :n] = modes, -modes
    linear(bounds, np.r_[-low, high])
    objective = np.zeros(total)
    if phase == 'feasibility':
        objective[n:n+k], objective[-1] = variant, 1.
        nonnegative = np.zeros((k + 1, total))
        nonnegative[:, n:] = np.eye(k + 1)
        linear(nonnegative, np.zeros(k + 1))
        maxima = np.zeros((k, total))
        maxima[:, n:n+k], maxima[:, -1] = -np.eye(k), 1.
        linear(maxima, np.zeros(k))
    else:
        objective[-1] = -1.
    selected = set(map(int, selected))
    hard_rows = set(map(int, hard_rows))
    for block_index, key in enumerate(model.keys):
        begin, end = model.slices[block_index:block_index+2]
        spec = model.norm_specs.get(key)
        if spec:
            compressed = model.vectors[spec['vector']]
            if spec.get('axis', False):
                compressed = compressed[:, 0]
            compressed = compressed.reshape(end-begin, compressed.shape[-2], n+1)
            gain = np.broadcast_to(spec['gain'], model.arrays[block_index].shape).ravel()
            offset = np.broadcast_to(spec['offset'], model.arrays[block_index].shape).ravel()
        for local, row in enumerate(range(begin, end)):
            upper = np.zeros(total)
            if phase == 'feasibility':
                upper[n+block_index] = 1.
                adjustment = 0.
            else:
                adjustment = tolerance[row]
            if spec:
                B, b = gain[local]*compressed[local, :, :-1]@modes, gain[local]*compressed[local, :, -1]
                norm_bound(B, b, offset[local]+adjustment, upper)
                if phase == 'margin' and row in selected:
                    upper = np.zeros(total); upper[-1] = -scales[row]
                    norm_bound(B, b, offset[local], upper)
                if row in hard_rows:
                    norm_bound(B, b, offset[local], np.zeros(total))
            else:
                upper[:n] = model.j[row]@modes
                linear(upper, model.g[row]+adjustment)
                if phase == 'margin' and row in selected:
                    entry = upper.copy(); entry[-1] = -scales[row]
                    linear(entry, model.g[row])
                if row in hard_rows:
                    entry = np.zeros(total); entry[:n] = model.j[row]@modes
                    linear(entry, model.g[row])
    # Nonnegative slacks first, followed by the individually assembled SOCs.
    matrices = [sparse.csc_matrix(-np.vstack(scalar_m)), *blocks]
    constants = [np.concatenate(scalar_c), *rhs]
    cones = [clarabel.NonnegativeConeT(len(constants[0])), *cones]
    P = sparse.block_diag((2*damping*modes.T@modes, np.zeros((total-n, total-n))), format='csc')
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 150
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    answer = clarabel.DefaultSolver(sparse.triu(P).tocsc(), objective,
        sparse.vstack(matrices, format='csc'), np.concatenate(constants), cones, settings).solve()
    x = np.asarray(answer.x, dtype=float)
    finite = x.shape == (total,) and np.isfinite(x).all()
    violation, value = None, None
    delta = np.zeros(n)
    if finite:
        delta = np.clip(modes@x[:n], low, high)
        prediction = model.prediction(delta)[0]
        checks = [delta-low, high-delta, prediction[list(hard_rows)]]
        if phase == 'feasibility':
            relax = np.repeat(x[n:n+k], np.diff(model.slices))
            checks.extend([prediction+relax, x[-1]-x[n:n+k], x[n:]])
            value = x[-1]+variant*x[n:n+k].sum()+damping*(delta@delta)
            deficits = [max(0., -float(model.g[a:b].min())) for a, b in zip(model.slices[:-1], model.slices[1:])]
            baseline = max(deficits)+variant*sum(deficits)
        else:
            indices = sorted(selected)
            checks.extend([prediction+tolerance, prediction[indices]-scales[indices]*x[-1]])
            value = -x[-1]+damping*(delta@delta)
            baseline = -float(np.min(model.g[indices]/scales[indices]))
        violation = max(0., -float(np.min(np.concatenate(checks))))
    usable = bool(finite and str(answer.status) in ('Solved', 'AlmostSolved')
                  and violation <= 2e-7 and value <= baseline+2e-8)
    return (delta if usable else np.zeros(n)), dict(
        backend='clarabel', success=str(answer.status) == 'Solved', status=str(answer.status),
        message=str(answer.status), iterations=int(answer.iterations), model_violation=violation,
        usable=usable, objective=None if value is None else float(value),
        solve_seconds=float(answer.solve_time))
