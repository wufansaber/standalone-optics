"""Select an MTF calculation consistent with the sampled optical regime."""

import numpy as np

from .standalone_mtf import diffraction_mtf
from .standalone_mtf_geometric import geometric_mtf


def calculate_mtf(system, field_xy, *, samples=256, wave=None, method='auto',
                  maximum_frequency_lpmm=None):
    """Use diffraction FFT for resolved wavefronts and ray MTF for severe blur."""
    if method not in ('auto', 'fft', 'geometric'):
        raise ValueError('Choose auto, fft, or geometric MTF')
    frequency = None
    if maximum_frequency_lpmm is not None:
        if not np.isfinite(maximum_frequency_lpmm) or maximum_frequency_lpmm <= 0.:
            raise ValueError('MTF maximum frequency must be finite and positive')
        frequency = np.linspace(0., maximum_frequency_lpmm, 101)
    kwargs = dict(samples=samples, wave=wave, frequency_lpmm=frequency)
    if method == 'geometric':
        return geometric_mtf(system, field_xy, **kwargs)
    diffraction = diffraction_mtf(system, field_xy, **kwargs)
    if method == 'fft':
        return diffraction
    warnings = set(diffraction['warnings'])
    severe = 'large_transverse_ray_aberration_scalar_fft' in warnings
    unresolved = ('phase_grid_may_alias_increase_sampling' in warnings
                  and 'large_opd_span_scalar_fft' in warnings)
    if severe or unresolved:
        result = geometric_mtf(system, field_xy, **kwargs)
        result['selection'] = {'requested_method': 'auto', 'fft_diagnostics': sorted(warnings)}
        return result
    return diffraction
