"""Adaptive working sets with an independent, fixed dense audit.

Samples are JSON objects supplied by an optical adapter (for example a field
coordinate and a named pupil quadrature). A quadrature is promoted as a whole:
isolated pupil points must not be assigned arbitrary wavefront integration
weights. This controller never substitutes a numerical optimizer status for
the independent audit and makes no continuous-domain certification claim.
"""
from copy import deepcopy
import json
import math
from time import perf_counter

import numpy as np


def _identity(sample):
    if not isinstance(sample, dict) or not sample:
        raise ValueError('Each sample must be a nonempty JSON object')
    return json.dumps(sample, sort_keys=True, separators=(',', ':'), allow_nan=False)


def run_constraint_exchange(solve_round, audit, active_samples, initial_q, *,
                            max_rounds=4, add_per_round=8, callback=None):
    """Solve on a working set, scan a fixed domain, and add its worst failures.

``solve_round(q, samples, round_index)`` returns ``(q, JSON-safe report)``.
``audit(q)`` returns nonempty records with ``sample``, ``gates`` (signed
slacks, positive means passing), and optional ``valid``. Every audit must
cover the same predeclared final domain; adapters may attach metadata to
records but must not silently drop failed samples. Nonfinite gates fail
closed. The caller controls all physical thresholds and exact ray tracing.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
    if type(add_per_round) is not int or add_per_round < 1:
        raise ValueError('add_per_round must be a positive integer')
    q = np.asarray(initial_q, dtype=float).copy()
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
        raise ValueError('initial_q must be a finite nonempty vector')
    active = {_identity(s): deepcopy(s) for s in active_samples}
    if not active:
        raise ValueError('The initial working set cannot be empty')
    history = []
    start = perf_counter()
    domain = None
    gate_layout = None
    passed = False
    reason = 'round_limit'
    for iteration in range(max_rounds):
        trial, solver_report = solve_round(q.copy(), deepcopy(list(active.values())), iteration)
        trial = np.asarray(trial, dtype=float)
        if trial.shape != q.shape or not np.isfinite(trial).all():
            raise ValueError('The local solver returned an invalid parameter vector')
        q = trial.copy()
        records = list(audit(q.copy()))
        if not records:
            raise ValueError('Independent audit returned no samples')
        identities = [_identity(r['sample']) for r in records]
        if len(set(identities)) != len(identities):
            raise ValueError('Independent audit contains duplicate samples')
        current_domain = set(identities)
        current_layout = {identity: {k: np.asarray(v).shape for k, v in r.get('gates', {}).items()}
                          for identity, r in zip(identities, records)}
        if domain is None:
            domain = current_domain
            gate_layout = current_layout
        elif current_domain != domain:
            raise ValueError('Independent audit domain changed between rounds')
        elif current_layout != gate_layout:
            raise ValueError('Independent audit gate layout changed between rounds')
        failed = []
        minima = {}
        per_gate = {}
        for identity, record in zip(identities, records):
            gates = record.get('gates')
            if not isinstance(gates, dict) or not gates:
                raise ValueError('Each audit sample requires nonempty signed gates')
            if 'valid' in record and not isinstance(record['valid'], (bool, np.bool_)):
                raise ValueError('Audit valid flags must be boolean')
            values = {}
            for key, value in gates.items():
                number = float(np.min(np.asarray(value, dtype=float)))
                values[key] = number if math.isfinite(number) else -math.inf
            if not record.get('valid', True):
                values['valid_ray_trace'] = -math.inf
            for key, value in values.items():
                minima[key] = min(minima.get(key, math.inf), value)
                if value < 0.:
                    old = per_gate.get(key)
                    if old is None or value < old[0]:
                        per_gate[key] = (value, identity, record['sample'])
            worst = min(values.values())
            if worst < 0.:
                failed.append((worst, identity, record['sample']))
        passed = not failed
        # First retain each gate's worst location, then fill by violation.
        candidates = sorted(per_gate.values()) + sorted(failed)
        added = []
        for _, identity, sample in candidates:
            if identity not in active:
                active[identity] = deepcopy(sample)
                added.append(deepcopy(sample))
                if len(added) == add_per_round:
                    break
        item = dict(round=iteration, working_samples=len(active)-len(added),
                    audited_samples=len(records), failed_samples=len(failed),
                    gate_minima={k: (v if math.isfinite(v) else None) for k, v in minima.items()},
                    added_samples=added, independent_audit_passed=passed,
                    solver=solver_report, seconds=perf_counter()-start)
        history.append(item)
        if callback is not None:
            callback(deepcopy(item))
        if passed:
            reason = 'independent_audit_passed'
            break
        if not added:
            reason = 'active_set_stalled'
            break
    return dict(q=q.tolist(), passed=passed, termination=reason, history=history,
                active_samples=list(active.values()), seconds=perf_counter()-start,
                continuous_domain_certified=False)
