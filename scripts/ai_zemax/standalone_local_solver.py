"""Cached, constrained local refinement for ray-tracing problem adapters.

The adapter supplies ``batch(qs)``, ``slacks(row)`` (nonnegative is feasible),
finite ``bounds`` and ``model(q)``. Parameters must already have useful scales.
No lens prescription, optical specification, GPU, or commercial solver is
embedded here. Numerical convergence is explicitly separate from acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize


def _write(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def _layout(blocks):
    keys = list(blocks)
    arrays = [np.asarray(blocks[key], dtype=float) for key in keys]
    if not arrays or any(not a.size or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Constraint blocks must be nonempty and finite')
    slices = np.r_[0, np.cumsum([a.size for a in arrays])]
    return keys, arrays, slices, np.concatenate([a.ravel() for a in arrays])


def _expanded(values, keys, arrays, default):
    if values is None or np.isscalar(values):
        return np.full(sum(a.size for a in arrays), default if values is None else float(values))
    unknown = set(values) - set(keys) - {'__default__'}
    if unknown:
        raise ValueError('Unknown constraint blocks: ' + ', '.join(sorted(unknown)))
    return np.concatenate([np.broadcast_to(values.get(k, values.get('__default__', default)), a.shape).ravel()
                           for k, a in zip(keys, arrays)])


@dataclass
class _Linearization:
    g: np.ndarray
    j: np.ndarray
    keys: list
    arrays: list
    slices: np.ndarray
    measured: list | None
    steps: np.ndarray | None
    vectors: dict
    modes: np.ndarray
    frozen: np.ndarray
    owner: object
    use_cones: bool
    norm_specs: dict
    _last_delta: np.ndarray | None = field(default=None, init=False)
    _last_prediction: tuple | None = field(default=None, init=False)

    def prediction(self, delta):
        """Evaluate the affine-slack / signed-residual-norm local model."""
        if self._last_delta is not None and np.array_equal(delta, self._last_delta):
            return self._last_prediction
        g, j = self.g + self.j @ delta, self.j.copy()
        if not self.use_cones:
            self._last_delta, self._last_prediction = delta.copy(), (g, j)
            return g, j
        lengths = {}
        derivative = {}
        for name, compressed in self.vectors.items():
            v = compressed @ np.r_[delta, 1.]
            length = np.linalg.norm(v, axis=-1)
            deriv = np.einsum('...i,...ij->...j', v, compressed[..., :-1]) / np.maximum(length[..., None], 1e-14)
            lengths[name], derivative[name] = length, deriv
        for ix, key in enumerate(self.keys):
            begin, end = self.slices[ix:ix+2]
            if key in self.norm_specs:
                spec = self.norm_specs[key]
                name = spec['vector']
                length, deriv = lengths[name], derivative[name]
                if spec.get('axis', False):
                    length, deriv = length[:, 0], deriv[:, 0]
                gain, offset = np.asarray(spec['gain']), np.asarray(spec['offset'])
                g[begin:end] = (offset-gain*length).ravel()
                j[begin:end] = (-gain[..., None]*deriv).reshape(-1, self.j.shape[1])
            elif key in ('wfe_order', 'spot_order') and hasattr(self.owner, 'order_scale'):
                name = 'spot_vector' if key == 'spot_order' else 'wave_vector'
                if name not in lengths:
                    continue
                length, deriv = lengths[name], derivative[name]
                scale, offset = self.owner.order_scale(key == 'spot_order')
                result = (length[:, 1:]-length[:, :1])*scale-offset
                if result.size == end-begin:
                    g[begin:end] = result.ravel()
                    j[begin:end] = ((deriv[:, 1:]-deriv[:, :1])*np.asarray(scale)[..., None]).reshape(-1, self.j.shape[1])
        self._last_delta, self._last_prediction = delta.copy(), (g, j)
        return g, j


class EfficientSolveMixin:
    """Add cached trust-region refinement to a tracing adapter.

    ``jacobian_provider(q, base, keys)`` may return a full slack Jacobian, or
    ``{'slacks': J, 'vectors': {'wave_vector': dVdq, ...}}``. Vector derivatives
    keep the norm-aware model without finite-difference ray probes. Optional
    ``strict_validator(model)`` performs the *independent* project acceptance;
    no validator means final acceptance is explicitly ``not_run``.
    """

    def solve(self, directory, iters, start_trust, variant, damping, use_cones=True, second_order=False,
              *, tolerances=None, phase2_iterations=0, margin_keys=None, margin_scales=None,
              jacobian_provider=None, callback=None, initial_q=None, strict_validator=None,
              difference_step=2.5e-4, batch_size=8, min_trust=2e-6, max_trust=2.5,
              stagnation_retries=6, verbose=True, subproblem_backend="slsqp", hard_gate_keys=(),
              trial_retractor=None):
        if subproblem_backend not in ("slsqp", "clarabel", "auto"):
            raise ValueError("Unknown subproblem_backend")
        if not np.isfinite([variant, damping]).all() or variant < 0 or damping < 0:
            raise ValueError("variant and damping must be finite and nonnegative")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError('Finite, ordered parameter bounds are required')
        n = len(bounds)
        q = np.zeros(n) if initial_q is None else np.asarray(initial_q, dtype=float).copy()
        if q.shape != (n,) or not np.isfinite(q).all() or np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
            raise ValueError('Initial parameters must be finite and inside bounds')
        if iters < 0 or phase2_iterations < 0 or difference_step <= 0 or batch_size < 1 or start_trust <= 0:
            raise ValueError('Invalid iteration, difference, batch or trust settings')
        counters = dict(batch_calls=0, traced_candidates=0, linearizations=0, derivative_cache_hits=0,
                        qr_builds=0, svd_builds=0, proposals=0, rejected_steps=0, accepted_steps=0,
                        derivative_failures=0, cone_linear_fallbacks=0, retraction_calls=0)
        times = dict(trace=0., derivatives=0., compression=0., subproblem=0., retraction=0., validation=0.,
                     optimization_seconds=0., seconds=0.)
        start = perf_counter()

        def trace(qs):
            tick = perf_counter()
            out = self.batch(np.asarray(qs))
            counters['batch_calls'] += 1
            counters['traced_candidates'] += len(qs)
            times['trace'] += perf_counter()-tick
            if len(out) != len(qs):
                raise ValueError('batch must return one row or None per candidate')
            return out

        base = trace([q])[0]
        if base is None:
            raise RuntimeError('Starting prescription fails real-ray evaluation')
        keys, arrays, slices, g = _layout(self.slacks(base))
        if set(hard_gate_keys)-set(keys):
            raise ValueError("hard_gate_keys must select existing constraint blocks")
        hard_rows = np.array([i for key, a, b in zip(keys, slices[:-1], slices[1:])
                              if key in hard_gate_keys for i in range(a, b)], dtype=int)
        if np.any(g[hard_rows] < 0.):
            raise ValueError("Initial hard gates must be feasible")
        tolerance = _expanded(tolerances, keys, arrays, 1e-8)
        scales = _expanded(margin_scales, keys, arrays, 1.)
        if np.any(tolerance < 0) or not np.isfinite(tolerance).all() or np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError('Tolerances must be finite/nonnegative and margin scales finite/positive')
        selected = list(margin_keys) if margin_keys is not None else [k for k in keys if not k.endswith('_order')]
        if set(selected)-set(keys) or not selected:
            raise ValueError('margin_keys must select existing constraint blocks')
        selected_rows = np.concatenate([np.arange(slices[keys.index(k)], slices[keys.index(k)+1]) for k in selected])

        def flatten(row):
            current_keys, current_arrays, _, current = _layout(self.slacks(row))
            if current_keys != keys or [a.shape for a in current_arrays] != [a.shape for a in arrays]:
                raise ValueError('Constraint layout changed; restart solve after changing the active sample set')
            return current

        def cost(v):
            deficit = np.array([max(0., -float(v[a:b].min())) for a, b in zip(slices[:-1], slices[1:])])
            return float(deficit.max()+variant*deficit.sum())

        def margin(v):
            return float(np.min(v[selected_rows]/scales[selected_rows]))

        def report(v):
            return dict(worst_slack=float(v.min()), merit=cost(v), margin=margin(v),
                        strict_feasible=bool(np.all(v >= 0.)), numerical_converged=bool(np.all(v >= -tolerance)),
                        gates={key: float(v[a:b].min()) for key, a, b in zip(keys, slices[:-1], slices[1:])})

        baseline = report(g)
        history, cache = [], None
        trust = min(float(start_trust), float(max_trust))
        phase = 'feasibility'
        phase2_floor = None
        remaining_margin = int(phase2_iterations)
        stalled = 0
        status = 'iteration_limit'
        subproblem_history = []
        # The baseline is always saved: immediate numerical convergence still
        # produces an inspectable prescription and an honest strict status.
        _write(directory/'BEST.json', self.model(q))

        def linearize():
            tick = perf_counter()
            frozen = bounds[:, 0] == bounds[:, 1]
            measured, steps = None, None
            vector_derivatives = {}
            if jacobian_provider is not None:
                supplied = jacobian_provider(q.copy(), base, tuple(keys))
                if isinstance(supplied, dict):
                    j = np.array(supplied['slacks'], dtype=float, copy=True)
                    vector_derivatives = supplied.get('vectors', {})
                else:
                    j = np.array(supplied, dtype=float, copy=True)
            else:
                steps = np.where(q+difference_step <= bounds[:, 1], difference_step, -difference_step)
                steps = np.where(steps > 0., np.minimum(steps, bounds[:, 1]-q), np.maximum(steps, bounds[:, 0]-q))
                # A boundary interval narrower than the nominal step still has
                # a usable one-sided derivative.
                for ix in np.flatnonzero((steps == 0.) & ~frozen):
                    steps[ix] = min(difference_step, bounds[ix, 1]-q[ix]) or max(-difference_step, bounds[ix, 0]-q[ix])
                measured = [base]*n
                indices = np.flatnonzero(~frozen)
                for begin in range(0, len(indices), batch_size):
                    current = indices[begin:begin+batch_size]
                    probes = np.repeat(q[None], len(current), axis=0)
                    probes[np.arange(len(current)), current] += steps[current]
                    for ix, row in zip(current, trace(probes)):
                        measured[ix] = row
                for ix in indices:
                    if measured[ix] is not None:
                        continue
                    for alternate in (-steps[ix], .25*steps[ix], -.25*steps[ix]):
                        if bounds[ix, 0] <= q[ix]+alternate <= bounds[ix, 1]:
                            probe = q.copy()
                            probe[ix] += alternate
                            row = trace([probe])[0]
                            if row is not None:
                                measured[ix], steps[ix] = row, alternate
                                break
                    if measured[ix] is None:
                        measured[ix] = base
                        frozen[ix] = True
                        counters['derivative_failures'] += 1
                safe_step = np.where(frozen, 1., steps)
                j = np.column_stack([(flatten(row)-g)/step for row, step in zip(measured, safe_step)])
                for name in ('wave_vector', 'spot_vector'):
                    if name in base:
                        vector_derivatives[name] = np.stack([(np.asarray(row[name])-base[name])/step
                                                             for row, step in zip(measured, safe_step)], axis=-1)
            if j.shape != (len(g), n) or not np.isfinite(j).all():
                raise ValueError('Jacobian has wrong shape or nonfinite entries')
            j[:, frozen] = 0.
            times['derivatives'] += perf_counter()-tick
            tick = perf_counter()
            vectors = {}
            if use_cones:
                for name, derivative in vector_derivatives.items():
                    derivative = np.array(derivative, dtype=float, copy=True)
                    if name not in base or derivative.shape != np.asarray(base[name]).shape+(n,) or not np.isfinite(derivative).all():
                        raise ValueError('Invalid residual vector Jacobian: '+name)
                    derivative[..., frozen] = 0.
                    vectors[name] = np.linalg.qr(np.concatenate((derivative, np.asarray(base[name])[..., None]), axis=-1), mode='r')
                    counters['qr_builds'] += 1
            modes = np.eye(n)
            norm_specs = {}
            explicit_specs = getattr(self, 'local_norm_specs', {})
            for ix, key in enumerate(keys):
                explicit = key in explicit_specs
                if explicit:
                    spec = dict(explicit_specs[key])
                elif key in ('wfe', 'axis', 'spot'):
                    target = getattr(self, {'wfe': 'wfe_target', 'axis': 'axis_target', 'spot': 'spot_target'}[key], None)
                    if target is None:
                        continue
                    spec = dict(vector='spot_vector' if key == 'spot' else 'wave_vector',
                                axis=key == 'axis', gain=1./target, offset=1.)
                else:
                    continue
                if spec.get('vector') not in vectors:
                    continue
                value = np.linalg.norm(base[spec['vector']], axis=-1)
                if spec.get('axis', False):
                    value = value[:, 0]
                gain, offset = np.asarray(spec['gain']), np.asarray(spec['offset'])
                predicted = offset-gain*value
                current_slack = g[slices[ix]:slices[ix+1]].reshape(arrays[ix].shape)
                matches = (predicted.shape == current_slack.shape and np.isfinite(predicted).all() and
                           np.all(gain > 0.) and np.allclose(predicted, current_slack, rtol=1e-9, atol=1e-10))
                if matches:
                    norm_specs[key] = spec
                elif explicit:
                    raise ValueError('Explicit local_norm_specs does not match slacks: '+key)
                else:
                    # Do not overwrite custom weighted constraints with a
                    # guessed unit-normalized cone. Their actual slack/J is
                    # retained unless the adapter declares the norm model.
                    counters['cone_linear_fallbacks'] += 1
            condition = float(getattr(self, 'condition_scale', 0.) or 0.)
            if condition > 0 and 'wave_vector' in vectors:
                response = vectors['wave_vector'][..., :-1].reshape(-1, n) / np.sqrt(
                    max(1, np.prod(vectors['wave_vector'].shape[:-2])))
                # Keep all parameter directions, including the response's
                # nullspace. Economy SVD drops those directions when there
                # are fewer response rows than parameters. Only request the
                # full basis in that small-row case: a full U for the usual
                # many-ray response would allocate an unnecessary huge matrix.
                _, singular, basis = np.linalg.svd(response, full_matrices=response.shape[0] < n)
                gains = np.ones(n)
                gains[:len(singular)] = 1./np.sqrt(1.+(singular/condition)**2)
                modes = basis.T*gains
                counters['svd_builds'] += 1
            times['compression'] += perf_counter()-tick
            counters['linearizations'] += 1
            safe_steps = None if steps is None else np.where(steps == 0., 1., steps)
            return _Linearization(g.copy(), j, keys, arrays, slices, measured, safe_steps, vectors, modes, frozen, self, use_cones, norm_specs)

        for iteration in range(int(iters)+int(phase2_iterations)):
            converged = bool(np.all(g >= -tolerance))
            if phase == 'feasibility' and converged:
                if remaining_margin <= 0:
                    status = 'numerical_feasibility'
                    break
                phase = 'margin'
                # Already feasible physical gates stay strictly feasible. A
                # numerical residual present on entry may not get worse.
                phase2_floor = np.minimum(g, 0.)
                cache = None
            if phase == 'feasibility' and iteration >= iters:
                status = 'iteration_limit'
                break
            if phase == 'margin':
                if remaining_margin <= 0:
                    status = 'margin_iteration_limit'
                    break
                remaining_margin -= 1
            if cache is None:
                cache = linearize()
            else:
                counters['derivative_cache_hits'] += 1
            low, high = np.maximum(-trust, bounds[:, 0]-q), np.minimum(trust, bounds[:, 1]-q)
            low[cache.frozen] = high[cache.frozen] = 0.
            tick = perf_counter()
            gate_tolerance = tolerance if phase == 'feasibility' else -phase2_floor
            delta, sub = self._local_proposal(cache, low, high, gate_tolerance, scales, selected_rows,
                                             phase, variant, damping, backend=subproblem_backend, hard_rows=hard_rows)
            times['subproblem'] += perf_counter()-tick
            counters['proposals'] += 1
            subproblem_history.append(sub)
            accepted = False
            rho = None
            old_score = cost(g) if phase == 'feasibility' else -margin(g)
            if sub['usable'] and np.max(abs(delta), initial=0.) > 1e-14:
                trials = q+np.array([1., .72, .47, .28, .14, .055])[:, None]*delta
                if trial_retractor is not None:
                    tick = perf_counter()
                    projected = []
                    for trial in trials:
                        counters['retraction_calls'] += 1
                        corrected = trial_retractor(trial.copy(), (q+low).copy(), (q+high).copy())
                        if corrected is None:
                            continue
                        corrected = np.asarray(corrected, dtype=float)
                        if (corrected.shape != (n,) or not np.isfinite(corrected).all()
                                or np.any(corrected < q+low) or np.any(corrected > q+high)):
                            raise ValueError('trial_retractor returned a point outside the original trust box')
                        projected.append(corrected)
                    trials = np.asarray(projected).reshape(-1, n)
                    times['retraction'] += perf_counter()-tick
                rows = trace(trials) if len(trials) else []
                if second_order and len(trials) and cache.measured is not None and hasattr(self, 'compensate_trial'):
                    refined = np.asarray(self.compensate_trial(q, trials[:4], rows[:4], base,
                                                              cache.measured, cache.steps, trust))
                    if refined.size:
                        refined = refined.reshape(-1, n)
                        okay = np.isfinite(refined).all(axis=1) & (refined >= q+low-1e-12).all(axis=1) & (refined <= q+high+1e-12).all(axis=1)
                        refined = refined[okay]
                        if len(refined):
                            trials = np.vstack((trials, refined))
                            rows.extend(trace(refined))
                choices = []
                for ix, row in enumerate(rows):
                    if row is None:
                        continue
                    v = flatten(row)
                    if np.any(v[hard_rows] < 0.):
                        continue
                    if phase == 'margin' and not np.all(v >= phase2_floor):
                        continue
                    score = cost(v) if phase == 'feasibility' else -margin(v)
                    predicted = cache.prediction(trials[ix]-q)[0]
                    prediction = old_score-(cost(predicted) if phase == 'feasibility' else -margin(predicted))
                    actual = old_score-score
                    numerical_floor = max(1e-13, 1e-8*max(abs(old_score), 1e-12))
                    if actual > numerical_floor and prediction > numerical_floor:
                        ratio = actual/prediction
                        # Reject only this poorly predicted trial. Another,
                        # smaller trial can have a useful model-agreement ratio
                        # even when the best raw score is unreliable.
                        if ratio >= .05:
                            choices.append((score, ix, v, ratio))
                if choices:
                    _, ix, new_g, rho = min(choices, key=lambda x: (x[0], x[1]))
                    if rho >= .05:
                        taken = np.max(abs(trials[ix]-q), initial=0.)
                        q, base, g = trials[ix].copy(), rows[ix], new_g
                        cache = None
                        counters['accepted_steps'] += 1
                        accepted = True
                        stalled = 0
                        if rho < .25:
                            trust *= .5
                        elif rho > .75 and taken >= .8*trust:
                            trust = min(max_trust, trust*1.6)
                        _write(directory/'BEST.json', self.model(q))
            if not accepted:
                counters['rejected_steps'] += 1
                stalled += 1
                trust *= .46
            item = dict(iteration=iteration, phase=phase, accepted=accepted,
                        rho=None if rho is None else float(rho), trust=float(trust),
                        seconds=perf_counter()-start, subproblem=sub, **report(g))
            history.append(item)
            if verbose:
                print(item, flush=True)
            if callback is not None:
                callback(q.copy(), base, item)
            _write(directory/'PROGRESS.json', dict(baseline=baseline, history=history,
                                                   counters=counters, timings=times))
            if trust < min_trust:
                status = 'trust_region_exhausted'
                break
            if stalled >= stagnation_retries:
                status = 'stagnation'
                break
        else:
            if np.all(g >= -tolerance) and phase2_iterations == 0:
                status = 'numerical_feasibility'
        self.last_solution_q, self.last_solution_row = q.copy(), base
        times['optimization_seconds'] = perf_counter()-start
        acceptance = dict(status='not_run', passed=None)
        if strict_validator is not None:
            validation_start = perf_counter()
            checked = strict_validator(self.model(q))
            if isinstance(checked, (bool, np.bool_)):
                acceptance = dict(status='completed', passed=bool(checked))
            elif isinstance(checked, dict) and isinstance(checked.get('passed'), (bool, np.bool_)):
                acceptance = dict(checked, status='completed', passed=bool(checked['passed']))
            else:
                raise ValueError('strict_validator must return bool or a dict containing boolean passed')
            times['validation'] = perf_counter()-validation_start
        times['seconds'] = perf_counter()-start
        result = dict(status=status, q=q.tolist(), **report(g), baseline=baseline, history=history,
                      counters=counters, timings=times, margin_keys=selected,
                      optimizer_tolerances={key: tolerance[a:b].reshape(array.shape).tolist()
                                            for key, a, b, array in zip(keys, slices[:-1], slices[1:], arrays)},
                      final_acceptance=acceptance,
                      subproblem_backend=subproblem_backend, hard_gate_keys=list(hard_gate_keys),
                      derivative_backend='external' if jacobian_provider is not None else 'finite_difference')
        _write(directory/'PROGRESS.json', result)
        return result

    def _local_proposal(self, model, low, high, tolerance, scales, selected, phase, variant, damping,
                        *, backend="slsqp", hard_rows=()):
        """Solve one cached local model; retain useful inexact SLSQP answers."""
        fallback = None
        if backend in ('clarabel', 'auto'):
            from .standalone_conic import available, unsupported, solve_proposal
            if not available():
                if backend == 'clarabel':
                    raise ImportError('Install optional dependency clarabel>=0.10,<0.12 for this backend')
                fallback = 'clarabel_not_installed'
            elif unsupported(model):
                if backend == 'clarabel':
                    raise ValueError(unsupported(model))
                fallback = unsupported(model)
            else:
                return solve_proposal(model, low, high, tolerance, scales, selected, phase,
                                      variant, damping, hard_rows)
        hard_rows = np.asarray(hard_rows, dtype=int)
        n, k = len(low), len(model.keys)
        modes = model.modes
        relax = np.repeat(np.eye(k), np.diff(model.slices), axis=0)
        finite_bounds = [(None, None)]*n
        if phase == 'feasibility':
            deficits = np.array([max(0., -float(model.g[a:b].min())) for a, b in zip(model.slices[:-1], model.slices[1:])])
            seed = np.r_[np.zeros(n), deficits+1e-7, deficits.max()+2e-7]
            finite_bounds += [(0., None)]*(k+1)
            def objective(x):
                return float(x[-1]+variant*x[n:-1].sum()+damping*np.sum((modes@x[:n])**2))
            def objective_jac(x):
                return np.r_[2*damping*modes.T@(modes@x[:n]), np.full(k, variant), 1.]
            def constraint(x):
                pred = model.prediction(modes@x[:n])[0]
                return np.r_[pred+relax@x[n:-1], pred[hard_rows], x[-1]-x[n:-1], modes@x[:n]-low, high-modes@x[:n]]
            def constraint_jac(x):
                j = model.prediction(modes@x[:n])[1]@modes
                return np.vstack((np.column_stack((j, relax, np.zeros(len(j)))),
                                  np.column_stack((j[hard_rows], np.zeros((len(hard_rows), k+1)))),
                                  np.column_stack((np.zeros((k, n)), -np.eye(k), np.ones(k))),
                                  np.column_stack((modes, np.zeros((n, k+1)))),
                                  np.column_stack((-modes, np.zeros((n, k+1))))))
        else:
            seed = np.r_[np.zeros(n), np.min(model.g[selected]/scales[selected])-1e-7]
            finite_bounds += [(None, None)]
            def objective(x):
                return float(-x[-1]+damping*np.sum((modes@x[:n])**2))
            def objective_jac(x):
                return np.r_[2*damping*modes.T@(modes@x[:n]), -1.]
            def constraint(x):
                pred = model.prediction(modes@x[:n])[0]
                return np.r_[pred+tolerance, pred[hard_rows], pred[selected]/scales[selected]-x[-1],
                             modes@x[:n]-low, high-modes@x[:n]]
            def constraint_jac(x):
                j = model.prediction(modes@x[:n])[1]@modes
                return np.vstack((np.column_stack((j, np.zeros(len(j)))),
                                  np.column_stack((j[hard_rows], np.zeros(len(hard_rows)))),
                                  np.column_stack((j[selected]/scales[selected, None], -np.ones(len(selected)))),
                                  np.column_stack((modes, np.zeros(n))),
                                  np.column_stack((-modes, np.zeros(n)))))
        solution = minimize(objective, seed, jac=objective_jac, bounds=finite_bounds, method='SLSQP',
                            constraints=[dict(type='ineq', fun=constraint, jac=constraint_jac)],
                            options=dict(maxiter=85, ftol=2e-9))
        candidate = np.asarray(solution.x, dtype=float)
        finite = candidate.shape == seed.shape and np.isfinite(candidate).all()
        violation = max(0., -float(np.min(constraint(candidate)))) if finite else None
        # success=False can still contain a useful finite feasible approximation.
        # A successful flag never overrides an infeasible/nonfinite answer.
        usable = finite and violation <= 2e-7 and objective(candidate) <= objective(seed)+2e-8
        delta = modes@candidate[:n] if usable else np.zeros(n)
        if usable:
            delta = np.clip(delta, low, high)
        return delta, dict(backend="slsqp", fallback_reason=fallback, success=bool(solution.success), status=int(solution.status),
                           message=str(solution.message), iterations=int(getattr(solution, 'nit', 0)),
                           model_violation=violation, usable=bool(usable),
                           objective=float(objective(candidate)) if finite else None)
