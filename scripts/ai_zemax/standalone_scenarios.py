"""Deterministic multi-scenario optimization with explicit bounded compensators.

Every scenario keeps every gate. A failed scenario invalidates the candidate.
Compensators are shared across the fields/waves in a scenario, not assigned per
measurement. This adapter does not infer distributions or manufacturing yield.
"""
from copy import deepcopy

import numpy as np

from .standalone_local_solver import EfficientSolveMixin


def _bounds(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 2))
    if (array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all()
            or np.any(array[:, 0] > array[:, 1])):
        raise ValueError('Use finite ordered bounds with shape (parameters, 2)')
    return array.copy()


class ScenarioProblem(EfficientSolveMixin):
    """Wrap a physical evaluator; all callback parameters use scaled coordinates.

``evaluate(design, scenario_payload, controls)`` returns None for invalid rays,
or ``{'gates': {name: signed_array}, 'vectors': {name: signed_residual_array}}``.
``model_factory(design)`` returns the shared physical design.
Optional ``local_jacobian(design, payload, controls, evaluated_row)`` returns
``{'gates': {name: J}, 'vectors': {name: J}}`` with last axis design+local controls.
Use ``problem.jacobian`` as solve's jacobian_provider to lift these derivatives.
``compensator_bounds`` maps every scenario ID to its allowed control bounds;
an empty list means that scenario cannot be compensated. No controls are added
by default. User callbacks must apply controls to the physical model/trace.
    """

    def __init__(self, design_bounds, scenarios, evaluate, model_factory, *,
                 compensator_bounds=None, local_jacobian=None, local_norm_specs=None):
        self.design_bounds = _bounds(design_bounds)
        self.design_size = len(self.design_bounds)
        if not self.design_size or not isinstance(scenarios, dict) or not scenarios:
            raise ValueError('Provide design variables and a nonempty named scenario mapping')
        if any(not isinstance(key, str) or not key for key in scenarios):
            raise ValueError('Scenario IDs must be nonempty strings')
        self.scenarios = deepcopy(scenarios)
        self.evaluate = evaluate
        self.model_factory = model_factory
        self.local_jacobian = local_jacobian
        self.local_norm_specs = deepcopy(local_norm_specs or {})
        if compensator_bounds is not None and set(compensator_bounds) != set(scenarios):
            raise ValueError('Specify compensator bounds for every scenario; use [] when forbidden')
        bounds, self.controls = [self.design_bounds], {}
        start = self.design_size
        for key in scenarios:
            current = _bounds([] if compensator_bounds is None else compensator_bounds[key])
            self.controls[key] = slice(start, start+len(current))
            bounds.append(current)
            start += len(current)
        self.bounds = np.concatenate(bounds)
        self._layout = None

    def _evaluate(self, q):
        if q.shape != (len(self.bounds),) or not np.isfinite(q).all():
            return None
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            return None
        rows, layout = [], None
        for key, payload in self.scenarios.items():
            row = self.evaluate(q[:self.design_size].copy(), deepcopy(payload), q[self.controls[key]].copy())
            if row is None:
                return None
            if not isinstance(row, dict):
                raise ValueError('Scenario evaluators must return a row dictionary or None')
            if 'valid' in row:
                if not isinstance(row['valid'], (bool, np.bool_)):
                    raise ValueError('Scenario valid flags must be boolean')
                if not row['valid']:
                    return None
            gates, vectors = row.get('gates'), row.get('vectors', {})
            if not isinstance(gates, dict) or not gates or not isinstance(vectors, dict):
                raise ValueError('Each scenario needs nonempty signed gates and optional vector mapping')
            gates = {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in gates.items()}
            vectors = {name: np.asarray(value, dtype=float) for name, value in vectors.items()}
            if set(gates) & set(vectors) or 'gates' in vectors or 'scenario_rows' in vectors:
                raise ValueError('Gate and vector names must be distinct and unreserved')
            values = list(gates.values())+list(vectors.values())
            if any(not value.size or not np.isfinite(value).all() for value in values):
                return None
            current = ([(name, value.shape) for name, value in gates.items()],
                       [(name, value.shape) for name, value in vectors.items()])
            if layout is None:
                layout = current
            elif current != layout:
                raise ValueError('All scenarios must preserve the same ordered gate/vector layout')
            rows.append(dict(row, gates=gates, vectors=vectors))
        if self._layout is None:
            self._layout = layout
        elif layout != self._layout:
            raise ValueError('Scenario layout changed; restart with a new problem')
        result = {'gates': {name: np.stack([row['gates'][name] for row in rows]) for name, _ in layout[0]},
                  'scenario_rows': rows}
        result.update({name: np.stack([row['vectors'][name] for row in rows]) for name, _ in layout[1]})
        return result

    def batch(self, qs):
        return [self._evaluate(np.asarray(q, dtype=float)) for q in qs]

    @staticmethod
    def slacks(row):
        return row['gates']

    def model(self, q):
        q = np.asarray(q, dtype=float)
        return dict(design=self.model_factory(q[:self.design_size].copy()),
                    scenarios=[dict(id=key, controls=q[self.controls[key]].tolist()) for key in self.scenarios],
                    compensation_scope='One bounded control vector per named physical scenario',
                    statistical_yield_estimated=False)

    def jacobian(self, q, base, keys):
        if self.local_jacobian is None:
            raise ValueError('No local_jacobian supplied; omit jacobian_provider for finite differences')
        gates, vectors = {key: [] for key in keys}, {key: [] for key in base if key not in ('gates', 'scenario_rows')}
        from .standalone_streaming import validate_compressed,compress_affine
        for index, (key, payload) in enumerate(self.scenarios.items()):
            controls = self.controls[key]
            columns = np.r_[np.arange(self.design_size), np.arange(controls.start, controls.stop)]
            local = self.local_jacobian(np.asarray(q[:self.design_size]).copy(), deepcopy(payload),
                                        np.asarray(q[controls]).copy(), deepcopy(base['scenario_rows'][index]))
            if set(local.get('vectors',{}))&set(local.get('compressed_vectors',{})):
                raise ValueError('A scenario vector cannot have both dense and compressed Jacobians')
            for label, destination in [('gates', gates), ('vectors', vectors)]:
                for name in destination:
                    expected = base['scenario_rows'][index][label][name].shape
                    if label=='vectors'and name in local.get('compressed_vectors',{}):
                        value=validate_compressed(local['compressed_vectors'][name],base['scenario_rows'][index]['vectors'][name],len(columns))
                        lifted=np.zeros(value.shape[:-1]+(len(self.bounds)+1,))
                        lifted[...,columns]=value[...,:-1];lifted[...,-1]=value[...,-1]
                        destination[name].append(('compressed',lifted));continue
                    value = np.asarray(local[label][name], dtype=float)
                    if value.shape != expected+(len(columns),) or not np.isfinite(value).all():
                        raise ValueError('Invalid scenario Jacobian for '+name)
                    lifted = np.zeros(expected+(len(self.bounds),))
                    lifted[..., columns] = value
                    destination[name].append(('dense',lifted)if label=='vectors'else lifted)
        dense,compressed={},{}
        for name,values in vectors.items():
            if all(mode=='dense'for mode,value in values):dense[name]=np.stack([value for mode,value in values])
            else:
                blocks=[value if mode=='compressed'else compress_affine(value,base['scenario_rows'][i]['vectors'][name])for i,(mode,value)in enumerate(values)]
                rows=max(block.shape[-2]for block in blocks)
                compressed[name]=np.stack([np.pad(block,[(0,0)]*(block.ndim-2)+[(0,rows-block.shape[-2]),(0,0)])for block in blocks])
        return dict(slacks=np.concatenate([np.stack(gates[key]).reshape(-1, len(self.bounds)) for key in keys]),
                    vectors=dense,compressed_vectors=compressed)
