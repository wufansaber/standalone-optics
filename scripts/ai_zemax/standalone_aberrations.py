"""Geometric aberration curves for a fixed coaxial optical prescription."""

import numpy as np

from .standalone import disk_quadrature, paraxial_transfer
from .standalone_analysis import best_focus


ZEMAX_AXIAL_PROBE = .005
ZEMAX_PARABASAL_PROBE = .001


def axis_crossings_image_um(points, directions):
    """Find the signed axial displacement to each on-axis meridional crossing."""
    points = np.asarray(points, dtype=float)
    directions = np.asarray(directions, dtype=float)
    radial = directions[:, :2]
    radial_norm_sq = np.sum(radial**2, axis=1)
    product = -1000. * np.sum(points[:, :2]*radial, axis=1)*directions[:, 2]
    return np.divide(product, radial_norm_sq, out=np.full(len(points), np.nan),
                     where=radial_norm_sq > 1e-28)


def _relative(value, baseline):
    return None if value is None or baseline is None else float(value - baseline)


def _finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _focus_um(rays, weights, tangential):
    values = best_focus(rays.points[:, :2], rays.directions, weights, tangential_axis=tangential)
    return {key: None if value is None else 1000.*value for key, value in values.items()}


def _required(rays):
    if not rays.valid.all():
        index = np.flatnonzero(~rays.valid)[0]
        raise ValueError(f'{np.count_nonzero(~rays.valid)} rays failed. First failure at surface '
                         f'{rays.failed_surface[index]+1}: {rays.reason[index]}')
    return rays


def _trace_diagnostics(rays, sample_names):
    failures = []
    for name, result in zip(sample_names, rays):
        for sample in np.flatnonzero(~result.valid):
            failures.append({'set': name, 'sample': int(sample), 'surface': int(result.failed_surface[sample])+1,
                             'reason': str(result.reason[sample])})
    return failures


def _first_order(system, wave):
    matrix = paraxial_transfer(system.surfaces[wave], 0., system.indices[wave, 0])
    object_index, image_index = system.indices[wave, [0, -1]]
    column = 0 if system.infinite else 1
    height, reduced_slope = matrix[:, column]
    if abs(reduced_slope) <= 1e-14:
        return matrix, None, None
    focus = -image_index*height/reduced_slope
    if system.infinite:
        return matrix, focus, None
    magnification = matrix[0, 0] + focus/image_index*matrix[1, 0]
    scale = image_index/object_index*magnification**2
    return matrix, focus, scale


def _fixed_image_scale(system, wave, matrix):
    if system.infinite:
        return None
    n_object, n_image = system.indices[wave, [0, -1]]
    chief_magnification = matrix[0, 0]+matrix[0, 1]*n_object*system.chief_slope_scale
    return n_image/n_object*chief_magnification**2 if abs(chief_magnification) > 1e-14 else None


def _parabasal_focus(system, wave, launch, axes, *, one_sided=False):
    if one_sided:
        pupil = np.concatenate((np.zeros_like(axes), ZEMAX_PARABASAL_PROBE*axes))
    else:
        step = 1e-4
        pupil = np.concatenate((-step*axes, step*axes))
    rays = _required(system._shoot(wave, pupil, *launch))
    slopes = rays.directions[:, :2]/rays.directions[:, 2, None]
    delta_slope = np.sum((slopes[2:]-slopes[:2])*axes, axis=1)
    delta_height = np.sum((rays.points[2:, :2]-rays.points[:2, :2])*axes, axis=1)
    result = np.divide(-1000.*delta_height, delta_slope, out=np.full(2, np.nan),
                       where=np.abs(delta_slope) > 1e-14)
    return dict(zip(('tangential', 'sagittal'), _finite_list(result)))


def _axial_wave(system, wave, pupil, weights, radii, focus=None):
    launch = (np.zeros(2), np.zeros(2))
    if focus is None:
        bundle = _required(system._shoot(wave, pupil, *launch))
        focus = _focus_um(bundle, weights, np.array([0., 1.]))
    traced_radii = np.r_[ZEMAX_AXIAL_PROBE, radii[1:]]
    fan = system._shoot(wave, np.column_stack((np.zeros(len(traced_radii)), traced_radii)), *launch)
    crossings = axis_crossings_image_um(fan.points, fan.directions)
    crossings[~fan.valid] = np.nan
    matrix, paraxial_shift, scale = _first_order(system, wave)
    paraxial_um = None if paraxial_shift is None else 1000.*paraxial_shift
    finite_probe_curve = _finite_list(crossings)
    image_shift = [paraxial_um, *finite_probe_curve[1:]]
    return matrix, scale, {
        'wavelength_um': float(system.wavelengths[wave]),
        'paraxial_focus_shift_image_um': paraxial_um,
        'geometric_rms_focus_shift_image_um': focus['mean_shift'],
        'axis_crossing_shift_image_um': image_shift,
        'longitudinal_spherical_image_um': [_relative(value, paraxial_um) for value in image_shift],
        'finite_probe_axis_crossing_shift_image_um': finite_probe_curve,
        'finite_probe_longitudinal_spherical_image_um': [
            _relative(value, finite_probe_curve[0]) for value in finite_probe_curve],
        'failures': _trace_diagnostics((fan,), ('axial_marginal_fan',)),
        'negative_segment_rays': int(np.count_nonzero(fan.negative_segment_count)),
    }


def _distortion_percent(system, wave, matrix, launch, chief_xy):
    height, slopes = launch
    if system.infinite:
        height = -(system.surfaces[wave][0].vertex_mm+system.entrance_pupil_z_mm)*slopes
    reference = matrix[0, 0]*height + matrix[0, 1]*system.indices[wave, 0]*slopes
    reference_sq = reference @ reference
    if reference_sq <= 1e-24:
        return None
    return float(100.*((chief_xy-reference) @ reference)/reference_sq)


def _field_wave(system, wave, launch, axes, chief, primary_chief, pupil,
                weights, fan_coordinates, axial, matrix, scale):
    bundle = system._shoot(wave, pupil, *launch)
    focus = _focus_um(bundle, weights, axes[0]) if bundle.valid.all() else None
    curvature = _relative(focus['mean_shift'], axial['geometric_rms_focus_shift_image_um']) if focus else None
    astigmatism = focus['astigmatic_separation'] if focus else None
    fans = tuple(system._shoot(wave, fan_coordinates[:, None]*axis, *launch) for axis in axes)
    fan_values = {}
    fan_even = {}
    fan_names = ('tangential', 'sagittal')
    for name, fan, axis in zip(fan_names, fans, axes):
        values = 1000.*((fan.points[:, :2]-chief) @ axis)
        values[~fan.valid] = np.nan
        fan_values[name] = _finite_list(values)
        fan_even[name] = _finite_list(((values + values[::-1])/2.)[len(values)//2:])
    fixed_axes = np.eye(2)
    fixed_fans = tuple(system._shoot(wave, fan_coordinates[:, None]*axis, *launch) for axis in fixed_axes)
    fixed_values = {}
    for coord, (fan, name) in enumerate(zip(fixed_fans, ('x', 'y'))):
        values = 1000.*(fan.points[:, coord]-primary_chief[coord])
        values[~fan.valid] = np.nan
        fixed_values[name] = _finite_list(values)
    object_curvature = None if curvature is None or scale is None else curvature/scale
    object_astigmatism = None if astigmatism is None or scale is None else astigmatism/scale
    fan_results = (*fans, *fixed_fans)
    failures = _trace_diagnostics(fan_results, (*fan_names, 'x', 'y'))
    if focus is None:
        failed = np.flatnonzero(~bundle.valid)
        first = failed[0]
        focus_failure = {'count': len(failed), 'surface': int(bundle.failed_surface[first])+1,
                         'reason': str(bundle.reason[first])}
    else:
        focus_failure = None
    return {
        'wavelength_um': float(system.wavelengths[wave]),
        'chief_xy_mm': chief.tolist(),
        'lateral_color_xy_um': (1000.*(chief-primary_chief)).tolist(),
        'lateral_color_radial_um': float(1000.*((chief-primary_chief) @ axes[0])),
        'geometric_focus_shifts_image_um': focus,
        'parabasal_focus_shifts_image_um': _parabasal_focus(system, wave, launch, axes),
        'finite_probe_parabasal_focus_shifts_image_um': _parabasal_focus(system, wave, launch, axes, one_sided=True),
        'field_curvature_image_um': curvature,
        'astigmatism_image_um': astigmatism,
        'field_curvature_object_um': object_curvature,
        'astigmatism_object_um': object_astigmatism,
        'paraxial_reference_distortion_percent': _distortion_percent(system, wave, matrix, launch, chief),
        'transverse_fans_image_um': fan_values,
        'transverse_fan_even_image_um': fan_even,
        'primary_chief_referenced_xy_fans_image_um': fixed_values,
        'failures': failures,
        'focus_failure': focus_failure,
        'negative_segment_rays': int(np.count_nonzero(bundle.negative_segment_count)
                                     + sum(np.count_nonzero(fan.negative_segment_count) for fan in fan_results)),
    }


def _max_abs(groups, key):
    return max((abs(group[key]) for group in groups if group[key] is not None), default=None)


def analyze_aberrations(system, *, pupil=None, weights=None, fan_samples=41, object_space=True):
    """Return image-space curves and finite-conjugate first-order object equivalents.

    All wavelengths trace the same physical field. For an actual-image-height
    field, its object-side launch is solved at the primary wavelength first.
    Field curvature uses the same-wavelength on-axis geometric RMS mean as
    baseline. The parabasal focus columns instead measure from the image plane.
    The project-equivalent object shifts use the near-axis chief magnification
    at the fixed image plane. The conjugate-plane scale is separately reported.
    """
    if not isinstance(fan_samples, int) or fan_samples < 3 or fan_samples % 2 != 1:
        raise ValueError('Fan sampling requires an odd number of at least three points')
    if pupil is None:
        pupil, weights = disk_quadrature(20, 64)
    weights = system.pupil_weights(pupil, weights, for_spot=True)
    fan_coordinates = np.linspace(-1., 1., fan_samples)
    radii = np.linspace(0., 1., fan_samples//2+1)
    wave_indices = range(len(system.wavelengths))
    axial = [_axial_wave(system, wave, pupil, weights, radii) for wave in wave_indices]
    fixed_image_scales = [_fixed_image_scale(system, wave, matrix) if object_space else None
                          for wave, (matrix, _, _) in enumerate(axial)]
    paraxial_key = 'paraxial_focus_shift_image_um'
    rms_key = 'geometric_rms_focus_shift_image_um'
    primary_axis = axial[system.primary][2]
    for _, _, row in axial:
        row['paraxial_axial_color_image_um'] = _relative(row[paraxial_key], primary_axis[paraxial_key])
        row['geometric_rms_axial_color_image_um'] = _relative(row[rms_key], primary_axis[rms_key])
        probe_key = 'finite_probe_axis_crossing_shift_image_um'
        row['finite_probe_axial_color_image_um'] = _relative(row[probe_key][0], primary_axis[probe_key][0])
    fields = []
    for field in system.fields:
        launch = system.launch_field(field)
        direction = launch[1] if system.infinite else launch[0]
        radial = direction/np.linalg.norm(direction) if np.any(direction) else np.array([0., 1.])
        axes = np.array([radial, [radial[1], -radial[0]]])
        chiefs = [_required(system._shoot(wave, np.zeros((1, 2)), *launch)).points[0, :2]
                  for wave in wave_indices]
        rows = []
        for wave, (matrix, _, reference) in enumerate(axial):
            row = _field_wave(system, wave, launch, axes, chiefs[wave], chiefs[system.primary],
                              pupil, weights, fan_coordinates, reference, matrix, fixed_image_scales[wave])
            rows.append(row)
        fields.append({'field': field.tolist(), 'waves': rows})
    groups = [row for field in fields for row in field['waves']]
    axial_rows = [row for _, _, row in axial]
    keys = ('field_curvature_image_um', 'astigmatism_image_um',
            'field_curvature_object_um', 'astigmatism_object_um')
    summary = {f'max_abs_{key}': _max_abs(groups, key) for key in keys}
    summary['focus_failures'] = sum(row['focus_failure'] is not None for row in groups)
    summary['fan_failures'] = sum(len(row['failures']) for row in groups+axial_rows)
    summary['signed_comparison_only'] = any(row['negative_segment_rays'] for row in groups+axial_rows)
    return {
        'field_type': system.model['field_type'],
        'primary_wavelength_um': float(system.wavelengths[system.primary]),
        'focus_pupil_sample_count': len(pupil),
        'fan_pupil_coordinate': fan_coordinates.tolist(),
        'fan_even_pupil_coordinate': fan_coordinates[len(fan_coordinates)//2:].tolist(),
        'axial_pupil_radius': radii.tolist(),
        'finite_probe_axial_pupil_radius': [ZEMAX_AXIAL_PROBE, *radii[1:].tolist()],
        'finite_probe_parabasal_pupil_step': ZEMAX_PARABASAL_PROBE,
        'axial': axial_rows,
        'object_to_image_longitudinal_magnification': fixed_image_scales,
        'paraxial_conjugate_longitudinal_magnification': [scale if object_space else None for _, scale, _ in axial],
        'fields': fields,
        'summary': summary,
    }
