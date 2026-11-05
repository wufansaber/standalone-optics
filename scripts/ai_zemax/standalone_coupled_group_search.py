"""Search group gaps with explicit, bounded, signed curvature modes."""

from collections import Counter
import math

import numpy as np

from .standalone_candidate_generation import curvature_candidates
from .standalone_group_search import search_group_gaps
from .standalone_parallel_screen import MAX_CANDIDATES_PER_BATCH


def _validated_modes(modes, surface_count):
    names = set()
    definitions = []
    for mode in modes:
        name = mode['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Curvature modes require distinct nonempty names')
        names.add(name)
        coefficients = []
        seen = set()
        for surface, coefficient in mode['curvature_per_unit_mm_inv']:
            if type(surface) is not int or not 1 <= surface < surface_count or surface in seen:
                raise ValueError('Mode terms must refer to distinct one-based surfaces before the image')
            if isinstance(coefficient, bool) or not math.isfinite(coefficient):
                raise ValueError('Mode coefficients must be finite curvature changes per unit mode')
            seen.add(surface)
            coefficients.append([surface, float(coefficient)])
        if not any(coefficient for _, coefficient in coefficients):
            raise ValueError('Every curvature mode must have a nonzero coefficient')
        low, high = mode['min'], mode['max']
        if (isinstance(low, bool) or isinstance(high, bool) or not math.isfinite(low)
                or not math.isfinite(high) or not low <= 0. <= high or low >= high
                or not math.isfinite(high-low)):
            raise ValueError('Each finite curvature-mode interval must contain zero and have positive width')
        definitions.append({'name': name, 'min': float(low), 'max': float(high),
                            'curvature_per_unit_mm_inv': coefficients})
    if not definitions:
        raise ValueError('Supply at least one curvature mode')
    return definitions


def _mode_samples(modes, count, random_seed):
    # Reuse the same baseline, endpoint and stratified sequence as curvature screening.
    variables = [{'surface': i, 'min_delta_mm_inv': mode['min'], 'max_delta_mm_inv': mode['max']}
                 for i, mode in enumerate(modes, 1)]
    for row in curvature_candidates(variables, count, random_seed=random_seed):
        coordinates = dict(row['curvature_deltas'])
        yield {mode['name']: coordinates.get(i, 0.) for i, mode in enumerate(modes, 1)}


def _source_hash(model):
    for name in ('source_sha256', 'coupled_group_search', 'group_gap_search'):
        value = model.get(name)
        if name == 'source_sha256' and value:
            return value
        if isinstance(value, dict) and value.get('parent_source_sha256'):
            return value['parent_source_sha256']
    return None


def _perturb(model, modes, values, source_hash):
    contributions = {}
    for mode in modes:
        coordinate = values[mode['name']]
        for surface, coefficient in mode['curvature_per_unit_mm_inv']:
            contributions.setdefault(surface, []).append(coordinate*coefficient)
    rows = list(model['surface_rows'])
    changes = []
    for surface, terms in sorted(contributions.items()):
        before = rows[surface-1].get('curvature_mm_inv', 0.)
        if not all(math.isfinite(term) for term in terms):
            raise ValueError('Curvature mode generated a nonfinite perturbation')
        after = before+math.fsum(terms)
        if not math.isfinite(after):
            raise ValueError('Curvature mode generated a nonfinite surface curvature')
        if after != before:
            rows[surface-1] = dict(rows[surface-1], curvature_mm_inv=after)
            changes.append([surface, after-before])
    if not changes:
        return model, changes
    changed = dict(model, surface_rows=rows)
    changed.pop('source_sha256', None)
    changed.pop('seed_realization', None)
    changed.pop('coupled_group_search', None)
    changed['group_gap_search'] = {'parent_source_sha256': source_hash}
    return changed, changes


def _coverage_order(candidates, baseline, bounds, modes, count):
    if len(candidates) <= count:
        return range(len(candidates))
    mode_span = np.array([mode['max']-mode['min'] for mode in modes])
    mode_points = np.array([list(row['mode_values'].values()) for row in candidates])/mode_span
    gap_span = np.maximum(np.ptp(bounds, axis=1), 1e-9)
    gap_points = (np.asarray([row['gap_mm'] for row in candidates])-baseline)/gap_span
    features = np.column_stack((mode_points/np.sqrt(mode_points.shape[1]),
                                gap_points/np.sqrt(gap_points.shape[1])))
    origin = next((i for i, row in enumerate(candidates) if row['id']=='baseline'), None)
    if origin is None:
        origin = int(np.argmin(np.linalg.norm(features, axis=1)))
    selected = [origin]
    distance = np.linalg.norm(features-features[origin], axis=1)
    for _ in range(count-1):
        distance[selected] = -1.
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(features-features[index], axis=1))
    return selected


def search_coupled_group_gaps(model, *, split_after_surfaces, gap_bounds_mm, curvature_modes,
                             track_mm=None, magnification=None, defocus_b_mm=None,
                             mode_samples=24, gap_samples=64, layouts_per_mode=4, count=64,
                             random_seed=0):
    """Build thick-group layouts after applying bounded curvature modes.

    Each mode scales an explicit set of signed surface curvature changes.
    Curvatures and air gaps are the only optical inputs changed. Every mode
    uses the same primary-wavelength A/B/track target, derived from the original
    model unless explicitly supplied. This is a paraxial generator. Evaluate
    its output with strict real-ray screening and the applicable mechanical
    limits before using a prescription.
    """
    limits = ((mode_samples, 256, 'mode samples'), (gap_samples, 4096, 'gap samples'),
              (layouts_per_mode, 64, 'layouts per mode'), (count, MAX_CANDIDATES_PER_BATCH, 'candidates'))
    for value, maximum, label in limits:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'Use 1–{maximum} {label}')
    if mode_samples*gap_samples > 100_000 or mode_samples*layouts_per_mode > MAX_CANDIDATES_PER_BATCH:
        raise ValueError('Requested curvature-mode and gap layout batch exceeds the search budget')
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError('Random seed must be a nonnegative integer')
    modes = _validated_modes(curvature_modes, len(model['surface_rows']))
    if len(modes) > 128:
        raise ValueError('Use at most 128 curvature modes')
    group_settings = dict(split_after_surfaces=split_after_surfaces, gap_bounds_mm=gap_bounds_mm,
                          samples=gap_samples, count=layouts_per_mode, random_seed=random_seed)
    reference = search_group_gaps(model, track_mm=track_mm, magnification=magnification,
                                  defocus_b_mm=defocus_b_mm, **group_settings)
    targets = reference['targets']
    source_hash = _source_hash(model)
    accepted = []
    mode_rows = []
    seen_curvatures = set()
    rejections = Counter()
    unique_layouts = 0
    for index, values in enumerate(_mode_samples(modes, mode_samples, random_seed)):
        identity = f'mode_{index:04d}'
        entry = {'id': identity, 'mode_values': values}
        mode_rows.append(entry)
        try:
            changed, deltas = _perturb(model, modes, values, source_hash)
            entry['curvature_deltas'] = deltas
            key = tuple(map(tuple, deltas))
            if key in seen_curvatures:
                entry.update(status='duplicate_curvatures', candidate_count=0)
                rejections['duplicate_curvatures'] += 1
                continue
            seen_curvatures.add(key)
            if index == 0:
                search = reference
            else:
                search = search_group_gaps(changed, **targets, **group_settings)
        except (ValueError, OverflowError, np.linalg.LinAlgError) as error:
            entry.update(status='invalid_curvatures', candidate_count=0, reason=str(error))
            rejections['invalid_curvatures'] += 1
            continue
        entry.update(status=search['status'], candidate_count=len(search['candidates']),
                     unique_accepted_count=search['unique_accepted_count'],
                     group_matrices_by_wave=search['group_matrices_by_wave'],
                     gap_rejections=search['rejections'])
        rejections.update(search['rejections'])
        unique_layouts += search['unique_accepted_count']
        if not search['candidates']:
            rejections['modes_without_layouts'] += 1
        for row in search['candidates']:
            layout_id = row['id']
            baseline = not deltas and layout_id == 'baseline'
            candidate = row['model']
            if not baseline:
                candidate.pop('source_sha256', None)
                candidate['coupled_group_search'] = {'parent_source_sha256': source_hash, 'mode_values': values,
                                                     'curvature_deltas': deltas, 'gap_mm': row['gap_mm']}
            accepted.append(dict(row, id='baseline' if baseline else f'{identity}_{layout_id}',
                                 mode_id=identity, group_layout_id=layout_id,
                                 mode_values=values, curvature_deltas=deltas))
    selected = _coverage_order(accepted, reference['baseline']['gap_mm'], reference['gap_bounds_mm'], modes, count)
    candidates = [accepted[i] for i in selected]
    result = {key: value for key, value in reference.items() if key not in ('candidates', 'rejections', 'status')}
    return dict(result, status='candidates_found' if candidates else 'no_feasible_candidates',
                method='coupled_thick_group_curvature_and_air_gap_search',
                curvature_modes=modes, mode_sample_count=mode_samples, layouts_per_mode=layouts_per_mode,
                unique_accepted_count=unique_layouts, generated_candidate_count=len(accepted),
                mode_reports=mode_rows, rejections=dict(rejections), candidates=candidates)
