"""Optional CUDA evaluation of bounded batches of candidate pupil bundles."""

from collections import defaultdict
from copy import deepcopy
from itertools import islice
import math

import numpy as np

from .standalone import directions_from_slopes, disk_quadrature
from .standalone_parallel_screen import (MAX_CANDIDATES_PER_BATCH, _OPTICAL_ERRORS, _candidate_model,
                                         _limit, _rank_key, _score_candidate, _selection_indices)
from .standalone_system import OpticalSystem

# A bundle always includes its chief in addition to the quadrature rays.
_MAX_DEVICE_RAYS = 131_072
_MAX_DEVICE_GROUPS = 256
_HOST_CANDIDATE_BATCH = 32
_WFE_REPLAY_TOL = 2e-8
_SPOT_REPLAY_TOL = 1e-6


def _new_result(index, candidate):
    return {'id': candidate['id'], 'input_index': index, 'status': 'rejected',
            'rank': None, 'worst_wfe_waves': None, 'mean_wfe_waves': None,
            'worst_spot_um': None, 'mean_spot_um': None, 'groups': [], 'reasons': []}


def _launch_infinite(system, wave, pupil, slopes, direction):
    """Build the same parallel phase plane as OpticalSystem._shoot."""
    first = system.surfaces[wave][0]
    entrance_z = system.entrance_pupil_z_mm
    at_vertex = system.entrance_radius_mm*pupil-entrance_z*slopes
    sag = first.sag_and_gradient(np.sum(at_vertex*at_vertex, axis=1))[0]
    finite_sag = sag[np.isfinite(sag)]
    clearance = 2.*system.entrance_radius_mm+1.
    if len(finite_sag):
        clearance += float(np.max(abs(finite_sag)))
    points = np.c_[at_vertex-clearance*slopes, np.full(len(pupil), first.vertex_mm-clearance)]
    reference = np.r_[-(clearance+entrance_z)*slopes, first.vertex_mm-clearance]
    points -= ((points-reference)@direction)[:,None]*direction
    opl_offset = system.indices[wave,0]*float(reference@direction)
    return points, opl_offset


def _launch_bundle(system, wave, origin, slopes, pupil_with_chief, positions, directions, offsets):
    if not system.infinite:
        positions[:,:2] = origin
        positions[:,2] = 0.
        directions[:] = directions_from_slopes(slopes+system.pupil_scale*pupil_with_chief)
        offsets[:] = 0.
        return
    direction = directions_from_slopes(np.asarray(slopes)[None,:])[0]
    directions[:] = direction
    # The chief and full pupil use their own phase-plane clearance, as on CPU.
    positions[:1], offsets[0] = _launch_infinite(system, wave, pupil_with_chief[:1], slopes, direction)
    positions[1:], offsets[1:] = _launch_infinite(system, wave, pupil_with_chief[1:], slopes, direction)


def _weighted_center(xp, values, weights):
    relative = values-values[:, :1]
    return relative-xp.sum(weights*relative,axis=1,keepdims=True)


def _reduce_scores(xp, traced, offsets, ref_distance, image_index, wavelengths, gaussian,
                   pupil_r2, base_weights):
    """Reduce strict ray traces to RMS metrics without returning ray data to host."""
    points = traced.points[:, 1:]
    directions = traced.directions[:, 1:]
    chief_points = traced.points[:, :1]
    radius_signed = ref_distance/traced.directions[:,0,2]
    radius = xp.abs(radius_signed)[:,None]
    relative = points-chief_points
    projection = xp.sum(relative*directions,axis=2)
    perpendicular = relative-projection[...,None]*directions
    radial_squared = xp.sum(perpendicular*perpendicular,axis=2)
    outside_sphere = xp.any(radial_squared > radius*radius,axis=1)
    root = xp.sqrt(xp.maximum(radius*radius-radial_squared,0.))
    correction = -projection+xp.sign(radius_signed)[:,None]*radial_squared/(radius+root)
    opl = traced.opl_mm+offsets
    opd = (opl[:,1:]-opl[:,:1])+image_index[:,None]*correction
    waves = opd/(wavelengths[:,None]*1e-3)

    apodization = -gaussian[:,None]*pupil_r2[None,:]
    phase_weight = base_weights[None,:]*xp.exp(apodization)
    spot_weight = base_weights[None,:]*xp.exp(2.*apodization)
    phase_sum = phase_weight.sum(axis=1,keepdims=True)
    spot_sum = spot_weight.sum(axis=1,keepdims=True)
    phase_weight /= xp.where(phase_sum > 0., phase_sum, 1.)
    spot_weight /= xp.where(spot_sum > 0., spot_sum, 1.)

    # Remove piston and fit direction-cosine tilt. Center and scale the two
    # columns before solving the 2-by-2 weighted normal equations.
    x = _weighted_center(xp,directions[:,:,0],phase_weight)
    y = _weighted_center(xp,directions[:,:,1],phase_weight)
    target = _weighted_center(xp,waves,phase_weight)
    sx = xp.sqrt(xp.sum(phase_weight*x*x,axis=1,keepdims=True))
    sy = xp.sqrt(xp.sum(phase_weight*y*y,axis=1,keepdims=True))
    x = x/xp.where(sx > 0.,sx,1.)
    y = y/xp.where(sy > 0.,sy,1.)
    xy = xp.sum(phase_weight*x*y,axis=1,keepdims=True)
    xx = xp.sum(phase_weight*x*x,axis=1,keepdims=True)
    yy = xp.sum(phase_weight*y*y,axis=1,keepdims=True)
    xt = xp.sum(phase_weight*x*target,axis=1,keepdims=True)
    yt = xp.sum(phase_weight*y*target,axis=1,keepdims=True)
    determinant = xx*yy-xy*xy
    fit_valid = (determinant[:,0] > 1e-8) & (sx[:,0] > 1e-12) & (sy[:,0] > 1e-12)
    inverse = 1./xp.where(fit_valid[:,None],determinant,1.)
    residual = target-x*((yy*xt-xy*yt)*inverse)-y*((xx*yt-xy*xt)*inverse)
    wfe = xp.sqrt(xp.sum(phase_weight*residual*residual,axis=1))

    px = _weighted_center(xp,points[:,:,0],spot_weight)
    py = _weighted_center(xp,points[:,:,1],spot_weight)
    spot = 1000.*xp.sqrt(xp.sum(spot_weight*(px*px+py*py),axis=1))
    # Degenerate/reference edge cases use the exact existing CPU calculation,
    # which supplies the same result or diagnostic as the CPU screen.
    valid = (xp.all(traced.valid,axis=1) & ~outside_sphere & fit_valid
             & (phase_sum[:,0] > 0.) & (spot_sum[:,0] > 0.) & (radius[:,0] > 0.)
             & xp.isfinite(radius[:,0])
             & xp.isfinite(wfe) & xp.isfinite(spot))
    return xp.column_stack((wfe,spot,valid))


def _finalize(result, settings):
    if result['reasons']:
        return
    groups = result['groups']
    for index, group in enumerate(groups):
        if 'error' not in group:
            continue
        result['reasons'].append({'code': 'trace_failed', 'field_index': group['field_index'],
                                  'wave_index': group['wave_index'], 'detail': group['error']})
        del groups[index:]
        return
    wfe = [row['wfe_waves'] for row in groups]
    spot = [row['spot_um'] for row in groups]
    result.update(worst_wfe_waves=max(wfe), mean_wfe_waves=math.fsum(value/len(wfe) for value in wfe),
                  worst_spot_um=max(spot), mean_spot_um=math.fsum(value/len(spot) for value in spot))
    for metric, threshold in (('worst_wfe_waves', settings['wfe_limit_waves']),
                              ('worst_spot_um', settings['spot_limit_um'])):
        if threshold is not None and result[metric] > threshold:
            result['reasons'].append({'code': 'screen_limit_exceeded', 'metric': metric,
                                      'value': result[metric], 'limit': threshold})
    if not result['reasons']:
        result['status'] = 'calculated'


class CudaCandidateScreen:
    """Score sequential prescriptions on one CUDA GPU, with the CPU report schema."""

    def __init__(self, model, *, device=0, radial=6, azimuthal=24, field_indices=None,
                 wave_indices=None, wfe_limit_waves=None, spot_limit_um=None):
        if type(device) is not int or device < 0:
            raise ValueError('CUDA device must be a nonnegative integer')
        if (type(radial) is not int or not 1 <= radial <= 128
                or type(azimuthal) is not int or not 3 <= azimuthal <= 512):
            raise ValueError('Use 1–128 radial nodes and 3–512 azimuthal samples')
        self.model = deepcopy(model)
        self.device = device
        self.settings = dict(radial=radial, azimuthal=azimuthal,
                             field_indices=_selection_indices(field_indices,'Field indices'),
                             wave_indices=_selection_indices(wave_indices,'Wavelength indices'),
                             wfe_limit_waves=_limit(wfe_limit_waves,'Wavefront limit'),
                             spot_limit_um=_limit(spot_limit_um,'Spot limit'))
        self._cp = None

    def __enter__(self):
        if self._cp is not None:
            raise RuntimeError('CUDA candidate screen is already open')
        from .standalone_gpu_trace import require_cupy

        cp = require_cupy(self.device)
        try:
            with cp.cuda.Device(self.device):
                self._pupil, self._weights = disk_quadrature(self.settings['radial'],self.settings['azimuthal'])
                self._with_chief = np.vstack((np.zeros((1,2)),self._pupil))
                self._weights_gpu = cp.asarray(self._weights)
                self._r2_gpu = cp.asarray(np.sum(self._pupil*self._pupil,axis=1))
                self._group_limit = min(_MAX_DEVICE_GROUPS, _MAX_DEVICE_RAYS//len(self._with_chief))
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError(f'CUDA screening could not initialize device {self.device}: {error}') from error
        self._cp = cp
        return self

    def __exit__(self, *exc):
        self._weights_gpu = self._r2_gpu = None
        self._cp = None

    def _cpu_group(self, system, group):
        self._replayed_groups += 1
        try:
            score, _, _ = system.evaluate(system.fields[group['field_index']], group['wave_index'],
                                          self._pupil,self._weights)
            wfe = float(score['wavefront_rms_waves'])
            spot = float(score['spot_rms_um'])
            if not math.isfinite(wfe) or not math.isfinite(spot):
                raise ValueError('Nonfinite RMS wavefront or spot result')
            group.update(wfe_waves=wfe,spot_um=spot)
        except _OPTICAL_ERRORS as error:
            group['error'] = str(error)

    def _score_jobs(self, jobs):
        from .standalone_gpu_trace import trace_gpu_batch

        cp = self._cp
        count = len(jobs)
        size = len(self._with_chief)
        positions = np.empty((count,size,3))
        directions = np.empty_like(positions)
        offsets = np.empty((count,size))
        surfaces = []
        object_indices = []
        controls = []
        launched = []
        for system,group,origin,slopes in jobs:
            wave = group['wave_index']
            i = len(launched)
            with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
                _launch_bundle(system,wave,origin,slopes,self._with_chief,
                               positions[i],directions[i],offsets[i])
            if (not np.isfinite(positions[i]).all() or not np.isfinite(directions[i]).all()
                    or not np.isfinite(offsets[i]).all() or np.any(directions[i,:,2] <= 0.)):
                self._cpu_group(system,group)
                continue
            launched.append((system,group))
            surfaces.append(system.surfaces[wave])
            object_indices.append(system.indices[wave,0])
            controls.append((system.reference_distance_mm,system.indices[wave,-1],
                             system.wavelengths[wave],system.gaussian_apodization))
        active = len(launched)
        if not active:
            return
        try:
            traced = trace_gpu_batch(surfaces,positions[:active],directions[:active],object_indices,
                                     return_device=True,device=self.device)
            scores = _reduce_scores(cp,traced,cp.asarray(offsets[:active]),*cp.asarray(controls).T,
                                    self._r2_gpu,self._weights_gpu)
            host_scores = cp.asnumpy(scores)
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError,
                cp.cuda.compiler.CompileException) as error:
            raise RuntimeError(f'CUDA screening failed on device {self.device}: {error}') from error
        for (system,group),(wfe,spot,valid) in zip(launched,host_scores,strict=True):
            if valid:
                group.update(wfe_waves=float(wfe),spot_um=float(spot))
            else:
                self._cpu_group(system,group)

    def _score_chunk(self, chunk, first):
        pending = defaultdict(list)
        rows = []
        for index,candidate in enumerate(chunk,first):
            row = _new_result(index,candidate)
            rows.append(row)
            try:
                system = OpticalSystem(_candidate_model(self.model,candidate))
                fields = self.settings['field_indices'] or range(len(system.fields))
                waves = self.settings['wave_indices'] or range(len(system.wavelengths))
                if max(fields) >= len(system.fields) or max(waves) >= len(system.wavelengths):
                    raise ValueError('Selected field or wavelength index is outside the candidate model')
            except _OPTICAL_ERRORS as error:
                row['reasons'].append({'code': 'invalid_model', 'detail': str(error)})
                continue
            topology = len(system.surfaces[0])
            for field in fields:
                for wave in waves:
                    group = {'field_index': field, 'wave_index': wave}
                    row['groups'].append(group)
                    try:
                        origin, slopes = system.launch_field(system.fields[field],wave)
                    except _OPTICAL_ERRORS as error:
                        group['error'] = str(error)
                        continue
                    pending[topology].append((system,group,origin,slopes))
                    if len(pending[topology]) == self._group_limit:
                        self._score_jobs(pending.pop(topology))
        for jobs in pending.values():
            self._score_jobs(jobs)
        for row in rows:
            _finalize(row,self.settings)
        return rows

    def _replay_boundaries(self, rows, candidates):
        # Small double-precision arithmetic differences must not choose between
        # effectively tied rows or flip a configured screening gate.
        replay = set()
        for row in rows:
            for metric,threshold,tolerance in (
                    ('worst_wfe_waves',self.settings['wfe_limit_waves'],_WFE_REPLAY_TOL),
                    ('worst_spot_um',self.settings['spot_limit_um'],_SPOT_REPLAY_TOL)):
                if threshold is not None and row[metric] is not None and abs(row[metric]-threshold) <= tolerance:
                    replay.add(row['input_index'])
        ranked = sorted((row for row in rows if row['status']=='calculated'),key=_rank_key)
        for a,b in zip(ranked,ranked[1:]):
            left,right = _rank_key(a)[:3],_rank_key(b)[:3]
            if left != right and abs(left[0]-right[0]) <= _WFE_REPLAY_TOL:
                replay.update((a['input_index'],b['input_index']))
        for index in sorted(replay):
            rows[index] = _score_candidate(self.model,self.settings,self._pupil,self._weights,
                                           (index,candidates[index]))
        return len(replay)

    def screen(self, candidates):
        """Evaluate a bounded candidate batch, with CPU tie and gate rechecks."""
        if self._cp is None:
            raise RuntimeError('Use CudaCandidateScreen as a context manager')
        batch = list(islice(candidates,MAX_CANDIDATES_PER_BATCH+1))
        if not 1 <= len(batch) <= MAX_CANDIDATES_PER_BATCH:
            raise ValueError(f'Use 1–{MAX_CANDIDATES_PER_BATCH} candidates per batch')
        if any(not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'].strip()
               for item in batch):
            raise ValueError('Every candidate needs a nonempty string id')
        if len({item['id'] for item in batch}) != len(batch):
            raise ValueError('Candidate ids must be unique within a batch')
        self._replayed_groups = 0
        rows = []
        with self._cp.cuda.Device(self.device):
            for first in range(0,len(batch),_HOST_CANDIDATE_BATCH):
                rows.extend(self._score_chunk(batch[first:first+_HOST_CANDIDATE_BATCH],first))
        replay_count = self._replay_boundaries(rows,batch)
        ranked = sorted((row for row in rows if row['status']=='calculated'),key=_rank_key)
        rejected = [row for row in rows if row['status']!='calculated']
        for rank,row in enumerate(ranked,1):
            row['rank'] = rank
        return {'status': 'calculated' if ranked else 'no_viable_candidates',
                'candidate_count': len(rows), 'ranked_count': len(ranked), 'rejected_count': len(rejected),
                'settings': {**self.settings,'workers': 1,'backend': 'cuda','device': self.device,
                             'pupil_rays': len(self._pupil),'cpu_replayed_candidates': replay_count,
                             'cpu_replayed_groups': self._replayed_groups},
                'ranked': [row['id'] for row in ranked],'results': ranked+rejected}
