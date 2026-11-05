"""Search rigid multi-element group spacings in a finite-conjugate prescription."""

from copy import deepcopy

import numpy as np
from numpy.polynomial import Polynomial

from .standalone import paraxial_transfer
from .standalone_system import OpticalSystem


def _translation(gap):
    return np.array(((1., gap), (0., 1.)))


def _compose(groups, gaps):
    result = _translation(gaps[0])
    for group, gap in zip(groups, gaps[1:]):
        result = _translation(gap) @ group @ result
    return result


def _kernel(groups, inner):
    """Transfer from the first group entrance to the last group exit."""
    return _compose(groups, (0., *inner, 0.))


def _roots(polynomial, bounds, anchor):
    """Enumerate in-bound real roots, with anchors for an identity equation."""
    coefficients = polynomial.coef
    scale = float(np.max(abs(coefficients)))
    low, high = bounds
    if scale < 1e-13:
        return list(dict.fromkeys((float(np.clip(anchor, low, high)), (low+high)/2., low, high)))
    polynomial = Polynomial(coefficients/scale).trim(tol=1e-13)
    if polynomial.degree() == 0:
        return []
    return [float(np.clip(root.real, low, high)) for root in polynomial.roots()
            if abs(root.imag) < 1e-9 and low-1e-9 <= root.real <= high+1e-9]


def _track_polynomial(a, c, d, total, m):
    # det(K)=1 removes K[0,1] when eliminating the object and image gaps.
    return total*c + (m+1./m) - a-d


def _solve_one(groups, inner, pivot, bounds, baseline, track_constant, m):
    """Fix all but one intergroup gap, and solve its track quadratic."""
    trial = inner.copy()
    trial[pivot] = 0.
    offset = _kernel(groups, trial)
    trial[pivot] = 1.
    slope = _kernel(groups, trial)-offset
    a = Polynomial((offset[0, 0], slope[0, 0]))
    c = Polynomial((offset[1, 0], slope[1, 0]))
    d = Polynomial((offset[1, 1], slope[1, 1]))
    total = Polynomial((track_constant+inner.sum()-inner[pivot], 1.))
    for value in _roots(_track_polynomial(a, c, d, total, m), bounds[pivot], baseline[pivot]):
        row = inner.copy()
        row[pivot] = value
        yield row


def _bilinear_kernel(groups, inner, p, q):
    trial = inner.copy()
    trial[p], trial[q] = 0., 0.
    base = _kernel(groups, trial)
    trial[p] = 1.
    x = _kernel(groups, trial)-base
    trial[p], trial[q] = 0., 1.
    y = _kernel(groups, trial)-base
    trial[p] = 1.
    xy = _kernel(groups, trial)-base-x-y
    return np.stack((base, x), axis=-1), np.stack((y, xy), axis=-1)


def _solve_two(groups, inner, pivots, bounds, baseline, outer, available, m, b):
    """Eliminate a second interior gap when one or both outer gaps are fixed."""
    p, q = pivots
    base, slope = _bilinear_kernel(groups, inner, p, q)
    a0, c0, d0 = (Polynomial(base[i, j]) for i, j in ((0, 0), (1, 0), (1, 1)))
    a1, c1, d1 = (Polynomial(slope[i, j]) for i, j in ((0, 0), (1, 0), (1, 1)))
    x = Polynomial((0., 1.))
    fixed_sum = inner.sum()-inner[p]-inner[q]
    object_gap, image_gap = outer
    if object_gap is not None and image_gap is not None:
        y = available-fixed_sum-object_gap-image_gap-x
        first = a0+y*a1+image_gap*(c0+y*c1)-m
        second = d0+y*d1+(object_gap-b/m)*(c0+y*c1)-1./m
        strongest = max((first, second), key=lambda value: np.max(abs(value.coef)))
        for value in _roots(strongest, bounds[p], baseline[p]):
            row = inner.copy()
            row[p], row[q] = value, y(value)
            yield row
        return

    if object_gap is not None:
        fixed = object_gap-b/m
        numerator, denominator = -(d0+fixed*c0-1./m), d1+fixed*c1
    else:
        numerator, denominator = -(a0+image_gap*c0-m), a1+image_gap*c1
    track_constant = b/m-available
    if np.max(abs(denominator.coef)) < 1e-13:
        values = _roots(numerator, bounds[p], baseline[p])
    else:
        total = (track_constant+fixed_sum+x)*denominator+numerator
        a = a0*denominator+a1*numerator
        c = c0*denominator+c1*numerator
        d = d0*denominator+d1*numerator
        equation = total*c + (m+1./m)*denominator**2-(a+d)*denominator
        values = _roots(equation, bounds[p], baseline[p])
        # Elimination multiplies by the denominator, so check its removable zeros.
        values.extend(_roots(denominator, bounds[p], baseline[p]))
    for value in values:
        row = inner.copy()
        row[p] = value
        divisor = denominator(value)
        if abs(divisor) > 1e-12:
            row[q] = numerator(value)/divisor
            yield row
        elif abs(numerator(value)) < 1e-10:
            yield from _solve_one(groups, row, q, bounds, baseline, track_constant, m)


def _outer_gaps(kernel, inner_sum, available, outer_bounds, anchor, m, b):
    """Find the outer gaps from track and the top-row matrix constraints."""
    total = available-inner_sum
    low = max(outer_bounds[1, 0], total-outer_bounds[0, 1])
    high = min(outer_bounds[1, 1], total-outer_bounds[0, 0])
    if low > high+1e-10:
        return None
    if high-low <= 1e-12:
        image = (low+high)/2.
    else:
        a, bk, c, d = kernel.ravel()
        column = np.array((c, d-m))
        target = np.array((m-a, b-bk-m*total))
        normal = column @ column
        image = float((column @ target)/normal) if normal > 1e-24 else float(np.clip(anchor, low, high))
        if not low-1e-9 <= image <= high+1e-9:
            return None
        image = float(np.clip(image, low, high))
    return total-image, image


def _group_templates(system, cuts):
    ends = (*cuts, len(system.surfaces[0])-1)
    starts = (0, *cuts)
    waves = []
    for index, surfaces in enumerate(system.surfaces):
        groups = [paraxial_transfer(surfaces[start:end], surfaces[start].vertex_mm, system.indices[index, start])
                  for start, end in zip(starts, ends)]
        waves.append(groups)
    return np.asarray(waves), list(zip(starts, ends))


def _shift_groups(model, original, gaps, sections):
    candidate = deepcopy(model)
    vertices = np.asarray(model['vertices_mm'], dtype=float).copy()
    offsets = np.cumsum(np.asarray(gaps)-original)
    for offset, (start, end) in zip(offsets, sections):
        vertices[start:end] += offset
    vertices[-1] += offsets[-1]
    candidate['vertices_mm'] = vertices.tolist()
    if np.any(abs(offsets) > 1e-10):
        previous = candidate.pop('group_gap_search', {})
        parent = candidate.pop('source_sha256', previous.get('parent_source_sha256'))
        candidate.pop('seed_realization', None)
        candidate['group_gap_search'] = {'parent_source_sha256': parent, 'gap_mm': list(gaps)}
    return candidate


def _anchors(bounds, baseline, dimensions, samples, random_seed):
    anchor = np.clip(baseline, bounds[:, 0], bounds[:, 1])
    yield anchor
    if not dimensions:
        return
    if len(dimensions) == 1:
        i = dimensions[0]
        for value in np.linspace(*bounds[i], samples):
            row = anchor.copy()
            row[i] = value
            yield row
        return
    rng = np.random.default_rng(random_seed)
    cells = np.column_stack([rng.permutation(samples) for _ in dimensions])
    unit = (cells+rng.random(cells.shape))/samples
    limits = bounds[dimensions]
    for values in limits[:, 0]+unit*np.diff(limits, axis=1)[:, 0]:
        row = anchor.copy()
        row[dimensions] = values
        yield row


def _candidate_indices(gaps, baseline, bounds, count):
    """Keep the closest layout and then maximize spacing coverage."""
    metric = np.asarray(gaps)/(np.maximum(np.ptp(bounds, axis=1), 1e-9))
    closest = int(np.argmin(np.linalg.norm(np.asarray(gaps)-baseline, axis=1)))
    selected = [closest]
    distance = np.linalg.norm(metric-metric[closest], axis=1)
    for _ in range(min(count, len(gaps))-1):
        distance[selected] = -1.
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(metric-metric[index], axis=1))
    return selected


def search_group_gaps(model, *, split_after_surfaces, gap_bounds_mm, track_mm=None,
                      magnification=None, defocus_b_mm=None, samples=1024, count=24,
                      random_seed=0):
    """Search air-gap changes while preserving every internal group surface.

    Supply at least two increasing one-based split surface numbers. The gap
    bounds are ordered object, intergroup gaps, and image. Defaults preserve
    the input track and primary ABCD top row; supply B=0 for paraxial focus.
    The engine solves one or two bounded variables analytically, and samples
    remaining free interior gaps deterministically. All group matrices at all
    wavelengths are retained in the report for independent composition.
    """
    system = OpticalSystem(model)
    if system.infinite:
        raise ValueError('Group gap search requires a finite object in air')
    vertices = np.asarray(model['vertices_mm'], dtype=float)
    cuts = tuple(split_after_surfaces)
    if (len(cuts) < 2 or any(type(cut) is not int or not 1 <= cut < len(vertices)-1 for cut in cuts)
            or tuple(sorted(set(cuts))) != cuts):
        raise ValueError('Supply at least two distinct increasing one-based group split surfaces')
    if not np.allclose(system.indices[:, [0, *cuts, -2, -1]], 1., rtol=0., atol=1e-12):
        raise ValueError('Object, selected intergroup gaps, and image must be in air at every wavelength')
    bounds = np.asarray(gap_bounds_mm, dtype=float)
    if (bounds.shape != (len(cuts)+2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds < 0.) or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError('Supply ordered nonnegative air-gap bound pairs for all group gaps')
    if type(samples) is not int or not 1 <= samples <= 50_000 or type(count) is not int or not 1 <= count <= 4096:
        raise ValueError('Use 1–50000 samples and retain 1–4096 candidates')
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError('Random seed must be a nonnegative integer')

    groups, sections = _group_templates(system, cuts)
    original = np.array([vertices[0], *(vertices[i]-vertices[i-1] for i in cuts), vertices[-1]-vertices[-2]])
    direct = np.asarray([paraxial_transfer(wave, 0., index[0])
                         for wave, index in zip(system.surfaces, system.indices)])
    rebuilt = np.asarray([_compose(wave, original) for wave in groups])
    primary = system.primary
    track = float(vertices[-1] if track_mm is None else track_mm)
    m = float(direct[primary, 0, 0] if magnification is None else magnification)
    b_mm = float(direct[primary, 0, 1] if defocus_b_mm is None else defocus_b_mm)
    if not np.isfinite([track, m, b_mm]).all() or track <= 0. or abs(m) < 1e-12:
        raise ValueError('Track, signed nonzero magnification, and B must be finite')
    internal_length = float(sum(vertices[end-1]-vertices[start] for start, end in sections))
    available = (track-internal_length)/track
    b = b_mm/track
    normalized_bounds = bounds/track
    original_n = original/track
    matrix = groups[primary].copy()
    matrix[:, 0, 1] /= track
    matrix[:, 1, 0] *= track
    inner_bounds = normalized_bounds[1:-1]
    outer_bounds = normalized_bounds[[0, -1]]
    fixed_outer = [float(pair[0]) if pair[1] == pair[0] else None for pair in outer_bounds]
    free = np.flatnonzero(np.ptp(inner_bounds, axis=1)>0.).tolist()
    n_pivots = min(len(free), 2 if any(x is not None for x in fixed_outer) else 1)
    pivots = sorted(free, key=lambda i: -np.ptp(inner_bounds[i]))[:n_pivots]
    sampled = [i for i in free if i not in pivots]
    rejection = {'out_of_bounds': 0, 'matrix_residual': 0, 'duplicate': 0}
    accepted = []
    seen = set()

    def record(gap):
        if (not np.isfinite(gap).all() or np.any(gap < normalized_bounds[:, 0]-1e-10)
                or np.any(gap > normalized_bounds[:, 1]+1e-10)):
            rejection['out_of_bounds'] += 1
            return False
        gap = np.clip(gap, normalized_bounds[:, 0], normalized_bounds[:, 1])
        a_actual, b_actual = _compose(matrix, gap)[0]
        if (abs(gap.sum()-available) > 1e-10
                or abs(a_actual-m) > 1e-9*max(1., abs(m))
                or abs(b_actual-b) > 1e-9*max(1., abs(b))):
            rejection['matrix_residual'] += 1
            return False
        key = tuple(np.round(gap, 11))
        if key in seen:
            rejection['duplicate'] += 1
            return False
        accepted.append(gap*track)
        seen.add(key)
        return True

    baseline_recorded = False
    if (np.all(original_n >= normalized_bounds[:, 0]-1e-12)
            and np.all(original_n <= normalized_bounds[:, 1]+1e-12)):
        baseline_recorded = record(original_n)
    if normalized_bounds[:, 0].sum()-1e-12 <= available <= normalized_bounds[:, 1].sum()+1e-12:
        for inner in _anchors(inner_bounds, original_n[1:-1], sampled, samples, random_seed):
            if not pivots:
                possibilities = (inner,)
            elif len(pivots) == 1:
                possibilities = _solve_one(matrix, inner, pivots[0], inner_bounds, original_n[1:-1], b/m-available, m)
            else:
                possibilities = _solve_two(matrix, inner, pivots, inner_bounds, original_n[1:-1], fixed_outer,
                                           available, m, b)
            for choice in possibilities:
                outer = _outer_gaps(_kernel(matrix, choice), choice.sum(), available, outer_bounds, original_n[-1], m, b)
                if outer is None:
                    rejection['out_of_bounds'] += 1
                    continue
                record(np.r_[outer[0], choice, outer[1]])

    candidates = []
    if accepted:
        for rank, index in enumerate(_candidate_indices(accepted, original, bounds, count), 1):
            gap = accepted[index]
            baseline = index == 0 and baseline_recorded
            if baseline:
                gap = original
            candidate = _shift_groups(model, original, gap, sections)
            full_matrices = np.asarray([_compose(wave, gap) for wave in groups])
            candidates.append({'id': 'baseline' if baseline else f'group_{rank:04d}',
                               'gap_mm': gap.tolist(), 'distance_from_baseline_mm': float(np.linalg.norm(gap-original)),
                               'matrix_by_wave': full_matrices.tolist(),
                               'track_error_mm': float(candidate['vertices_mm'][-1]-track),
                               'magnification_error': float(full_matrices[primary, 0, 0]-m),
                               'defocus_b_error_mm': float(full_matrices[primary, 0, 1]-b_mm),
                               'model': candidate, 'status': 'generated'})

    return {
        'status': 'candidates_found' if candidates else 'no_feasible_candidates',
        'method': 'rigid_thick_group_air_gap_search',
        'group_surface_ranges': [[start+1, end] for start, end in sections],
        'gap_after_surfaces': [0, *cuts, len(vertices)-1],
        'group_matrices_by_wave': groups.tolist(),
        'wavelengths_um': system.wavelengths.tolist(),
        'primary_wave': primary,
        'targets': {'track_mm': track, 'magnification': m, 'defocus_b_mm': b_mm},
        'gap_bounds_mm': bounds.tolist(),
        'baseline': {'gap_mm': original.tolist(), 'track_mm': float(vertices[-1]),
                     'matrix_by_wave': direct.tolist(), 'reconstructed_matrix_by_wave': rebuilt.tolist(),
                     'max_abcd_entry_delta': float(np.max(abs(rebuilt-direct)))},
        'sample_count': samples, 'random_seed': random_seed,
        'sampled_gaps_after_surfaces': [[0, *cuts, len(vertices)-1][i+1] for i in sampled],
        'rejections': rejection, 'unique_accepted_count': len(accepted), 'candidates': candidates,
    }
