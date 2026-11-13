"""Deterministic native-free coarse screening of complete lens prescriptions."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from itertools import islice
import math
import multiprocessing
import os
from threading import Lock

import numpy as np

from .standalone import disk_quadrature
from .standalone_manufacturing import audit_manufacturing, audit_ray_envelope
from .standalone_system import OpticalSystem


MAX_CANDIDATES_PER_BATCH = 4096
_BLAS_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
_SPAWN_ENV_LOCK = Lock()
_OPTICAL_ERRORS = (KeyError, TypeError, ValueError, RuntimeError, OverflowError,
                   FloatingPointError, np.linalg.LinAlgError)


@contextmanager
def _single_thread_spawn_environment():
    """Set worker BLAS limits before the fresh Python process imports NumPy."""
    with _SPAWN_ENV_LOCK:
        previous = {name: os.environ.get(name) for name in _BLAS_ENV}
        try:
            os.environ.update(dict.fromkeys(_BLAS_ENV, '1'))
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _selection_indices(values, name):
    if values is None:
        return None
    indices = tuple(values)
    if (not indices or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError(f'{name} must contain distinct nonnegative zero-based integers')
    return indices


def _limit(value, name):
    if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0.):
        raise ValueError(f'{name} must be finite and positive')
    return None if value is None else float(value)


def _candidate_model(model, candidate):
    unknown = candidate.keys() - {'id', 'model', 'curvature_deltas'}
    if unknown:
        raise ValueError(f'Unsupported candidate keys: {sorted(map(str, unknown))}')
    if 'model' in candidate:
        if 'curvature_deltas' in candidate:
            raise ValueError('Supply either a complete model or curvature deltas')
        return candidate['model']
    if model is None:
        raise ValueError('Curvature candidates require a baseline model')
    rows = list(model['surface_rows'])
    seen = set()
    for surface, delta in candidate.get('curvature_deltas', ()):
        if type(surface) is not int or not 1 <= surface < len(rows) or surface in seen:
            raise ValueError('Use distinct 1-based optical surface numbers before the image plane')
        if isinstance(delta, bool) or not math.isfinite(delta):
            raise ValueError('Curvature deltas must be finite')
        index = surface-1
        row = dict(rows[index])
        curvature = row.get('curvature_mm_inv', 0.)+delta
        if not math.isfinite(curvature):
            raise ValueError('Perturbed curvature must be finite')
        row['curvature_mm_inv'] = curvature
        rows[index] = row
        seen.add(surface)
    return dict(model, surface_rows=rows)


def _manufacturing_policy(value):
    if value is not None and not isinstance(value, dict):
        raise ValueError('Manufacturing policy must be an object')
    return deepcopy(value)


def _manufacturing_preflight(model, settings, result):
    """Block impossible solids and enforce an explicitly selected project policy."""
    policy = settings['manufacturing_policy']
    report = audit_manufacturing(model, policy)
    result['manufacturing'] = report
    if not report['eligible']:
        result['reasons'].append({'code': 'manufacturing_failed',
                                  'detail': 'Candidate violates physical geometry or configured limits'})
        return False
    if policy is not None and not report['passed']:
        result['reasons'].append({'code': 'manufacturing_incomplete',
                                  'detail': 'Configured nominal-geometry checks have missing inputs'})
        return False
    return True


def _manufacturing_ray_preflight(system, settings, result, pupil):
    """Use one shared strict CPU envelope gate for CPU and CUDA acceptance."""
    geometry = result['manufacturing']
    if settings['manufacturing_policy'] is None and not any(
            row['mechanical_radius_mm'] is not None for row in geometry['surfaces']):
        return True
    report = audit_ray_envelope(system, geometry, pupil)
    result['manufacturing'] = report
    if not report['eligible']:
        result['reasons'].append({'code': 'manufacturing_ray_envelope_failed',
                                  'detail': 'Fresh full-field/full-spectrum ray paths violate configured dimensions or fail tracing'})
        return False
    return True


def _score_candidate(model, settings, pupil, weights, task):
    input_index, candidate = task
    result = {'id': candidate['id'], 'input_index': input_index, 'status': 'rejected',
              'rank': None, 'worst_wfe_waves': None, 'mean_wfe_waves': None,
              'worst_spot_um': None, 'mean_spot_um': None, 'groups': [], 'reasons': []}
    reasons = result['reasons']
    try:
        candidate_model = _candidate_model(model, candidate)
        if not _manufacturing_preflight(candidate_model, settings, result):
            return result
        system = OpticalSystem(candidate_model)
        fields = settings['field_indices'] or range(len(system.fields))
        waves = settings['wave_indices'] or range(len(system.wavelengths))
        if max(fields) >= len(system.fields) or max(waves) >= len(system.wavelengths):
            raise ValueError('Selected field or wavelength index is outside the candidate model')
        if not _manufacturing_ray_preflight(system, settings, result, pupil):
            return result
    except _OPTICAL_ERRORS as error:
        reasons.append({'code': 'invalid_model', 'detail': str(error)})
        return result

    groups = result['groups']
    for i in fields:
        for j in waves:
            try:
                score, _, _ = system.evaluate(system.fields[i], j, pupil, weights)
                wfe = float(score['wavefront_rms_waves'])
                spot = float(score['spot_rms_um'])
                if not math.isfinite(wfe) or not math.isfinite(spot):
                    raise ValueError('Nonfinite RMS wavefront or spot result')
            except _OPTICAL_ERRORS as error:
                reasons.append({'code': 'trace_failed', 'field_index': i, 'wave_index': j,
                                'detail': str(error)})
                return result
            groups.append({'field_index': i, 'wave_index': j, 'wfe_waves': wfe, 'spot_um': spot})
    wfe = [row['wfe_waves'] for row in groups]
    spot = [row['spot_um'] for row in groups]
    result.update(worst_wfe_waves=max(wfe), mean_wfe_waves=math.fsum(value/len(wfe) for value in wfe),
                  worst_spot_um=max(spot), mean_spot_um=math.fsum(value/len(spot) for value in spot))
    for metric, threshold in (('worst_wfe_waves', settings['wfe_limit_waves']),
                              ('worst_spot_um', settings['spot_limit_um'])):
        if threshold is not None and result[metric] > threshold:
            reasons.append({'code': 'screen_limit_exceeded', 'metric': metric,
                            'value': result[metric], 'limit': threshold})
    if not reasons:
        result['status'] = 'calculated'
    return result


def _initialize_worker(model, settings):
    global _worker_state
    pupil, weights = disk_quadrature(settings['radial'], settings['azimuthal'])
    _worker_state = (model, settings, pupil, weights)


def _worker_score(task):
    return _score_candidate(*_worker_state, task)


def _rank_key(result):
    return (result['worst_wfe_waves'], result['mean_wfe_waves'],
            result['worst_spot_um'], result['input_index'])


class CandidateScreen:
    """Keep model data and one BLAS thread per spawned worker across batches.

    Candidate IDs are unique within each batch. A candidate specifies an optional
    list of [one-based optical surface, delta curvature in 1/mm] pairs applied
    to the baseline, or a complete normalized model. Fields and wavelengths use
    zero-based indices and default to all of the entries of each model. Keep the
    optical evaluation conditions the same for all candidates in a ranking.
    """

    def __init__(self, model, *, workers=1, radial=6, azimuthal=24, field_indices=None,
                 wave_indices=None, wfe_limit_waves=None, spot_limit_um=None,
                 manufacturing_policy=None):
        if type(workers) is not int or not 1 <= workers <= 61:
            raise ValueError('Worker count must be an integer from 1 to 61')
        if (type(radial) is not int or not 1 <= radial <= 128
                or type(azimuthal) is not int or not 3 <= azimuthal <= 512):
            raise ValueError('Use 1–128 radial nodes and 3–512 azimuthal samples')
        self.model = deepcopy(model)
        self.workers = workers
        self.settings = dict(radial=radial, azimuthal=azimuthal,
                             field_indices=_selection_indices(field_indices, 'Field indices'),
                             wave_indices=_selection_indices(wave_indices, 'Wavelength indices'),
                             wfe_limit_waves=_limit(wfe_limit_waves, 'Wavefront limit'),
                             spot_limit_um=_limit(spot_limit_um, 'Spot limit'),
                             manufacturing_policy=_manufacturing_policy(manufacturing_policy))
        self._pool = None
        self._pupil = None

    def __enter__(self):
        if self._pool is not None or self._pupil is not None:
            raise RuntimeError('Candidate screen is already open')
        if self.workers == 1:
            self._pupil = disk_quadrature(self.settings['radial'], self.settings['azimuthal'])
        else:
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                             mp_context=multiprocessing.get_context('spawn'),
                                             initializer=_initialize_worker,
                                             initargs=(self.model, self.settings))
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        self._pupil = None

    def screen(self, candidates):
        """Evaluate a bounded batch, preserving input order to break ties."""
        if self._pool is None and self._pupil is None:
            raise RuntimeError('Use CandidateScreen as a context manager')
        batch = list(islice(candidates, MAX_CANDIDATES_PER_BATCH+1))
        if not 1 <= len(batch) <= MAX_CANDIDATES_PER_BATCH:
            raise ValueError(f'Use 1–{MAX_CANDIDATES_PER_BATCH} candidates per batch')
        if any(not isinstance(item, dict) or not isinstance(item.get('id'), str)
               or not item['id'].strip() for item in batch):
            raise ValueError('Every candidate needs a nonempty string id')
        if len({item['id'] for item in batch}) != len(batch):
            raise ValueError('Candidate ids must be unique within a batch')
        if self._pool is None:
            results = [_score_candidate(self.model, self.settings, *self._pupil, task)
                       for task in enumerate(batch)]
        else:
            # ProcessPoolExecutor starts new processes during task submission.
            with _single_thread_spawn_environment():
                futures = self._pool.map(_worker_score, enumerate(batch), chunksize=1)
            results = list(futures)
        ranked = sorted((row for row in results if row['status'] == 'calculated'), key=_rank_key)
        rejected = [row for row in results if row['status'] != 'calculated']
        for rank, row in enumerate(ranked, 1):
            row['rank'] = rank
        return {'status': 'calculated' if ranked else 'no_viable_candidates',
                'candidate_count': len(results), 'ranked_count': len(ranked),
                'rejected_count': len(rejected), 'settings': {**self.settings, 'workers': self.workers,
                                                             'pupil_rays': self.settings['radial']*self.settings['azimuthal']},
                'ranked': [row['id'] for row in ranked], 'results': ranked+rejected}


def screen_candidates(model, candidates, *, workers=1, radial=6, azimuthal=24,
                      field_indices=None, wave_indices=None, wfe_limit_waves=None, spot_limit_um=None,
                      manufacturing_policy=None):
    """Run one coarse screening batch with the same results as a reusable pool."""
    with CandidateScreen(model, workers=workers, radial=radial, azimuthal=azimuthal,
                         field_indices=field_indices, wave_indices=wave_indices,
                         wfe_limit_waves=wfe_limit_waves, spot_limit_um=spot_limit_um,
                         manufacturing_policy=manufacturing_policy) as screen:
        return screen.screen(candidates)
