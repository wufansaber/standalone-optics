"""Bounded exact-constraint projection before an independently re-traced trial."""
import numpy as np
from scipy.optimize import minimize


def project_feasible(candidate, lower, upper, values, jacobian, *, max_iterations=60, feasibility_buffer=1e-10):
    """Find a nearby point with every supplied original slack >=0, or return None.

The callback computes actual nonlinear constraints; no numerical solver success
flag supplies feasibility credit. This does not perform optical acceptance.
    """
    candidate, lower, upper = [np.asarray(v, dtype=float) for v in (candidate, lower, upper)]
    if (candidate.ndim != 1 or lower.shape != candidate.shape or upper.shape != candidate.shape
            or not all(np.isfinite(v).all() for v in (candidate, lower, upper))
            or np.any(lower > upper) or np.any(candidate < lower) or np.any(candidate > upper)
            or type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(feasibility_buffer) or feasibility_buffer < 0.):
        raise ValueError('Use finite in-box candidate and ordered bounds, with positive iteration budget')

    def gates(q):
        g = np.asarray(values(q), dtype=float)
        if g.ndim != 1 or not g.size or not np.isfinite(g).all():
            raise ValueError('Projection requires nonempty finite nonlinear slacks')
        return g

    initial = gates(candidate)
    if np.all(initial >= 0.):
        return candidate.copy()

    def derivative(q):
        j = np.asarray(jacobian(q), dtype=float)
        if j.shape != (len(initial), len(candidate)) or not np.isfinite(j).all():
            raise ValueError('Projection Jacobian shape/values do not match slacks')
        return j

    answer = minimize(lambda q: (.5*float((q-candidate)@(q-candidate)), q-candidate),
        candidate, jac=True, bounds=np.c_[lower, upper], method='SLSQP',
        constraints=[dict(type='ineq', fun=lambda q: gates(q)-feasibility_buffer, jac=derivative)],
        options=dict(maxiter=max_iterations, ftol=1e-13))
    q = np.asarray(answer.x, dtype=float)
    if (q.shape == candidate.shape and np.isfinite(q).all() and np.all(q >= lower)
            and np.all(q <= upper) and np.all(gates(q) >= 0.)):
        return q.copy()
    return None
