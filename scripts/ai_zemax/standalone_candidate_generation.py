"""Generate deterministic curvature candidates for first-pass prescription screening."""

import numpy as np

from .standalone_parallel_screen import MAX_CANDIDATES_PER_BATCH


def curvature_candidates(variables, count, *, random_seed=0):
    """Include the baseline, one-variable endpoints, and a stratified batch."""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_CANDIDATES_PER_BATCH:
        raise ValueError(f'Candidate count must be an integer from 1 to {MAX_CANDIDATES_PER_BATCH}')
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError('Random seed must be a nonnegative integer')
    if not variables:
        raise ValueError('Supply at least one curvature variable')
    surfaces = [variable['surface'] for variable in variables]
    if (any(not isinstance(surface, int) or isinstance(surface, bool) or surface <= 0 for surface in surfaces)
            or len(surfaces) != len(set(surfaces))):
        raise ValueError('Curvature variables must have distinct positive surface numbers')
    bounds = np.asarray([[variable['min_delta_mm_inv'], variable['max_delta_mm_inv']]
                         for variable in variables], dtype=float)
    if (not np.isfinite(bounds).all() or np.any(bounds[:, 0] > 0.) or np.any(bounds[:, 1] < 0.)
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Each nonempty curvature interval must include zero')

    endpoints = [np.zeros(len(surfaces))]
    for i, limits in enumerate(bounds):
        for limit in limits:
            if limit:
                row = np.zeros(len(surfaces))
                row[i] = limit
                endpoints.append(row)
    points = np.asarray(endpoints[:count])
    remaining = count-len(points)
    if remaining:
        random = np.random.default_rng(random_seed)
        cells = np.column_stack([random.permutation(remaining) for _ in surfaces])
        unit = (cells+random.random(cells.shape))/remaining
        points = np.vstack((points, bounds[:, 0]+unit*np.diff(bounds, axis=1)[:, 0]))
    return [
        {'id': 'baseline' if i == 0 else f'candidate_{i:05d}',
         'curvature_deltas': [[surface, float(value)] for surface, value in zip(surfaces, point) if value]}
        for i, point in enumerate(points)
    ]
