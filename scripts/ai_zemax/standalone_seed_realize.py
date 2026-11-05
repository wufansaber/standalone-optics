"""Turn a three-group thin seed into a traceable thick singlet prototype."""

import numpy as np

from .standalone import Surface, paraxial_transfer


def _vector(value, length, name):
    values = np.asarray(value, dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain {length} finite values')
    return values


def _translation(distance):
    return np.array([[1., distance], [0., 1.]])


def _group(power, reduced_thickness):
    root_squared = 1. - reduced_thickness * power
    if root_squared <= 0.:
        raise ValueError('The thick singlet cannot realize the requested power')
    root = np.sqrt(root_squared)
    surface_power = power / (1. + root)
    diagonal = 1. - reduced_thickness * surface_power
    matrix = np.array([[diagonal, reduced_thickness], [-power, diagonal]])
    diagonal_derivative = -reduced_thickness / (2. * root)
    derivative = np.array([[diagonal_derivative, 0.], [-1., diagonal_derivative]])
    return matrix, derivative, surface_power


def _matched_powers(initial, air_gaps, reduced_thickness, target, track):
    """Match the first matrix row while retaining the middle group's power."""
    translations = [_translation(gap / track) for gap in air_gaps]
    reduced = reduced_thickness / track
    powers = initial * track
    signs = np.sign(powers[[0, 2]])
    middle = _group(powers[1], reduced[1])[0]
    internal = translations[2] @ middle @ translations[1]
    desired = np.array([target, 0.])

    def evaluate(trial):
        first, first_derivative, _ = _group(trial[0], reduced[0])
        last, last_derivative, _ = _group(trial[2], reduced[2])
        before_last = internal @ first @ translations[0]
        after_first = translations[3] @ last @ internal
        residual = (translations[3] @ last @ before_last)[0] - desired
        jacobian = np.column_stack(((after_first @ first_derivative @ translations[0])[0],
                                    (translations[3] @ last_derivative @ before_last)[0]))
        return residual, jacobian

    tolerance = 2e-12 * max(1., abs(target))
    for _ in range(30):
        residual, jacobian = evaluate(powers)
        norm = np.linalg.norm(residual)
        if norm <= tolerance:
            return powers / track
        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as error:
            raise ValueError('Thick realization has a singular conjugate match') from error
        for scale in 0.5 ** np.arange(18):
            trial = powers.copy()
            trial[[0, 2]] -= scale * correction
            updated = trial[[0, 2]]
            if np.any((signs != 0.) & (signs * updated <= 0.)):
                continue
            if np.any(1. - reduced * trial <= 0.):
                continue
            if np.linalg.norm(evaluate(trial)[0]) < norm:
                powers = trial
                break
        else:
            raise ValueError('Could not match thick conjugates on the seed power branch')
    raise ValueError('Thick conjugate match did not converge')


def _clear_radii(surfaces, stop, fields, na, primary, margin):
    entrance = paraxial_transfer(surfaces[primary][:stop], 0.)
    a, b = entrance[0]
    if abs(b) < 1e-12:
        raise ValueError('The selected stop does not define a finite chief-ray launch')
    chief_slope_scale = -a / b
    pupil_slope = na / np.sqrt(1. - na**2)
    maximum_field = float(np.max(np.linalg.norm(fields, axis=1)))
    radii = np.zeros(6)
    for sequence in surfaces:
        for i in range(6):
            a, b = paraxial_transfer(sequence[:i + 1], 0.)[0]
            radii[i] = max(radii[i], abs(a + b * chief_slope_scale) * maximum_field
                           + abs(b) * pupil_slope)
    return np.maximum(1e-6, margin * np.max(radii.reshape(3, 2), axis=1))


def _edge_gaps(surfaces, radii, image_z):
    """Minimum axial separation at the center or common spherical edge."""
    vertices = np.array([0., *(surface.vertex_mm for surface in surfaces[:6]), image_z])
    curvatures = np.array([0., *(surface.curvature_mm_inv for surface in surfaces[:6]), 0.])
    radius = np.r_[radii[0], np.repeat(radii, 2), radii[-1]]
    overlap = np.minimum(radius[:-1], radius[1:])
    def sag(curvature):
        return curvature * overlap**2 / (1. + np.sqrt(1. - (curvature * overlap)**2))

    at_center = np.diff(vertices)
    at_edge = at_center + sag(curvatures[1:]) - sag(curvatures[:-1])
    return np.minimum(at_center, at_edge)


def realize_seed(seed, *, refractive_indices=((1.52, 1.52, 1.52),),
                 wavelengths_um=(.5876,), center_thicknesses_mm=(3., 3., 3.),
                 object_na=.02, fields=((0., 0.), (0., 1.)), primary_wave=0,
                 stop_group=None, wavelength_weights=None, aperture_margin=1.15,
                 minimum_air_gap_mm=.05, minimum_edge_thickness_mm=.05,
                 max_group_radius_mm=None):
    """Build a neutral prescription with finite object and fixed track in air.

    Each seed lens plane becomes the center of a symmetric spherical singlet.
    First and last effective powers are adjusted to retain signed magnification
    and conjugate focus at the primary wavelength. Clear radii use a paraxial
    envelope and mechanical axial gap checks. Real-ray screening follows.
    """
    gaps = _vector(seed['gaps_mm'], 4, 'Thin gaps')
    powers = _vector(seed['powers_mm_inv'], 3, 'Thin powers')
    thickness = _vector(center_thicknesses_mm, 3, 'Center thicknesses')
    target_matrix = np.asarray(seed['matrix'], dtype=float)
    indices = np.asarray(refractive_indices, dtype=float)
    wavelengths = np.asarray(wavelengths_um, dtype=float)
    fields = np.asarray(fields, dtype=float)
    if target_matrix.shape != (2, 2) or not np.isfinite(target_matrix).all():
        raise ValueError('Seed must provide a finite 2 by 2 paraxial matrix')
    if np.any(gaps <= 0.) or np.any(thickness <= 0.):
        raise ValueError('Thin gaps and center thicknesses must be positive')
    if wavelengths.ndim != 1 or not len(wavelengths) or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0.):
        raise ValueError('Wavelengths must be a nonempty positive finite vector')
    if (indices.shape != (len(wavelengths), 3) or not np.isfinite(indices).all()
            or np.any(indices <= 1.)):
        raise ValueError('Supply one row of three glass indices greater than one per wavelength')
    if not isinstance(primary_wave, int) or not 0 <= primary_wave < len(wavelengths):
        raise ValueError('Primary wavelength index is out of range')
    if fields.ndim != 2 or fields.shape[1] != 2 or not len(fields) or not np.isfinite(fields).all():
        raise ValueError('Fields must be a nonempty list of finite object-space XY pairs')
    controls = [object_na, aperture_margin, minimum_air_gap_mm, minimum_edge_thickness_mm]
    if (not np.isfinite(controls).all() or not 0. < object_na < 1. or aperture_margin < 1.
            or minimum_air_gap_mm <= 0. or minimum_edge_thickness_mm <= 0.):
        raise ValueError('Require 0 < object NA < 1, aperture margin >= 1, and positive minimum gaps')
    radius_limit = None
    if max_group_radius_mm is not None:
        limits = np.asarray(max_group_radius_mm, dtype=float)
        radius_limit = np.full(3, limits.item()) if limits.ndim == 0 else limits
        if radius_limit.shape != (3,) or not np.isfinite(radius_limit).all() or np.any(radius_limit <= 0.):
            raise ValueError('Supply one positive clear-radius limit or one limit for each group')
    if stop_group is None:
        stop_group = seed.get('stop_group', 2)
    if not isinstance(stop_group, int) or not 1 <= stop_group <= 3:
        raise ValueError('Stop group must be 1, 2, or 3')
    weights = np.ones(len(wavelengths)) if wavelength_weights is None else _vector(
        wavelength_weights, len(wavelengths), 'Wavelength weights')
    if np.any(weights < 0.) or not np.any(weights > 0.):
        raise ValueError('Wavelength weights must be nonnegative with a positive sum')

    track = float(gaps.sum())
    centers = np.cumsum(gaps)[:3]
    vertices = np.column_stack((centers - thickness / 2., centers + thickness / 2.)).ravel()
    air_gaps = np.diff(np.r_[0., vertices, track])[::2]
    if np.any(air_gaps < minimum_air_gap_mm):
        raise ValueError('Thick vertex air gaps are below the minimum')
    if np.any(thickness < minimum_edge_thickness_mm):
        raise ValueError('Singlet center thickness is below the specified minimum')
    target = float(target_matrix[0, 0])
    if abs(target_matrix[0, 1]) > 1e-8 * track or target == 0.:
        raise ValueError('Thin seed does not image at its prescribed track')
    primary_indices = indices[primary_wave]
    reduced = thickness / primary_indices
    fitted = _matched_powers(powers, air_gaps, reduced, target, track)
    front = np.array([_group(p, t)[2] for p, t in zip(fitted, reduced)]) / (primary_indices - 1.)
    curvature = np.column_stack((front, -front)).ravel()
    vertices = np.r_[vertices, track]
    curvature = np.r_[curvature, 0.]
    index_rows = np.ones((len(wavelengths), 8))
    index_rows[:, 1:6:2] = indices
    surfaces = [
        [Surface(z, n, c) for z, n, c in zip(vertices, row[1:], curvature)]
        for row in index_rows
    ]
    stop = 2 * stop_group - 1
    radii = _clear_radii(surfaces, stop, fields, object_na, primary_wave, aperture_margin)
    if radius_limit is not None and np.any(radii > radius_limit):
        raise ValueError('Thick realization exceeds the maximum group radius')
    if np.any(np.abs(front) * radii >= 1.):
        raise ValueError('Estimated clear radius extends beyond a spherical surface domain')
    edge_gaps = _edge_gaps(surfaces[primary_wave], radii, track)
    if np.any(edge_gaps[::2] < minimum_air_gap_mm):
        raise ValueError('Thick realization violates the minimum clear-aperture air gap')
    if np.any(edge_gaps[1::2] < minimum_edge_thickness_mm):
        raise ValueError('Thick realization violates the minimum edge thickness')
    transfer = paraxial_transfer(surfaces[primary_wave], 0.)
    if not np.allclose(transfer[0], [target, 0.], rtol=1e-9, atol=1e-9 * max(1., track)):
        raise ValueError('Thick realization failed its independent paraxial transfer check')
    return {
        'vertices_mm': vertices.tolist(),
        'surface_rows': [
            dict(curvature_mm_inv=float(c), aperture_radius_mm=float(radius))
            for c, radius in zip(curvature[:6], np.repeat(radii, 2))
        ] + [dict(curvature_mm_inv=0., aperture_radius_mm=None)],
        'indices_by_wave': index_rows.tolist(),
        'wavelengths_um': wavelengths.tolist(),
        'wavelength_weights': weights.tolist(),
        'primary_wave': primary_wave,
        'stop_surface': stop,
        'object_at_infinity': False,
        'field_type': 'object_height',
        'fields': fields.tolist(),
        'field_normalization': 'radial',
        'aperture_type': 'obna',
        'aperture_value': float(object_na),
        'seed_realization': {
            'method': 'three_symmetric_singlets',
            'target_magnification': target,
            'track_mm': track,
            'primary_transfer': transfer.tolist(),
            'thin_powers_mm_inv': powers.tolist(),
            'realized_powers_mm_inv': fitted.tolist(),
            'group_centers_mm': centers.tolist(),
            'clear_radii_mm': radii.tolist(),
            'minimum_air_gaps_mm': edge_gaps[::2].tolist(),
            'minimum_glass_thicknesses_mm': edge_gaps[1::2].tolist(),
        },
    }
