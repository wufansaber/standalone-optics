"""Configured nominal solid checks for coaxial refractive models, in millimetres."""
import math
from copy import deepcopy

import numpy as np
from numpy.polynomial import Polynomial

from .standalone import Surface


_LIMITS = {
    'minimum_center_thickness_mm', 'minimum_radial_thickness_mm',
    'minimum_air_gap_mm', 'minimum_rim_gap_mm', 'maximum_diameter_mm',
    'minimum_clear_to_mechanical_radial_margin_mm', 'maximum_surface_slope',
    'center_thickness_budget_mm', 'radial_thickness_budget_mm',
    'air_gap_budget_mm', 'rim_gap_budget_mm',
}
_RADII = {'mechanical_radius_mm', 'clear_radius_mm'}


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0. or (positive and value == 0.):
        raise ValueError(f'{name} must be positive or nonnegative as specified')
    return float(value)


def _limits(value, name):
    if not isinstance(value, dict) or value.keys() - _LIMITS:
        raise ValueError(f'Unknown or malformed {name}')
    return {key: _number(v, f'{name}.{key}', key.startswith('maximum_')) for key, v in value.items()}


def _policy(value, count):
    if value is None:
        return {}, {}, None, False, 1., 1025
    if not isinstance(value, dict) or value.keys() - {
            'schema_version', 'defaults', 'surfaces', 'lenses', 'all_spherical',
            'ambient_index', 'radial_samples', 'policy_id', 'sources', 'notes'}:
        raise ValueError('Unknown or malformed manufacturing policy')
    if type(value.get('schema_version', 1)) is not int or value.get('schema_version', 1) != 1:
        raise ValueError('Unsupported manufacturing schema_version')
    spherical = value.get('all_spherical', False)
    if type(spherical) is not bool:
        raise ValueError('all_spherical must be a boolean')
    samples = value.get('radial_samples', 1025)
    if type(samples) is not int or not 33 <= samples <= 65537:
        raise ValueError('radial_samples must be an integer from 33 to 65537')
    defaults = _limits(value.get('defaults', {}), 'defaults')
    overrides = value.get('surfaces', {})
    if not isinstance(overrides, dict):
        raise ValueError('surfaces must map one-based surface numbers')
    radii = {}
    for key, row in overrides.items():
        if not isinstance(key, str) or not key.isdigit() or not 1 <= int(key) <= count:
            raise ValueError('Surface override keys must be one-based surface numbers')
        if not isinstance(row, dict) or row.keys() - _RADII:
            raise ValueError('Surface overrides accept only mechanical_radius_mm and clear_radius_mm')
        radii[int(key)] = {name: _number(v, f'S{key}.{name}', True) for name, v in row.items()}
    declarations = value.get('lenses')
    if declarations is not None:
        if not isinstance(declarations, list) or not declarations:
            raise ValueError('lenses must be a nonempty list')
        checked = []
        for lens in declarations:
            if not isinstance(lens, dict) or lens.keys() - (_LIMITS | {'front_surface', 'rear_surface'}):
                raise ValueError('Unknown or malformed lens declaration')
            a, b = lens.get('front_surface'), lens.get('rear_surface')
            if type(a) is not int or type(b) is not int or not 1 <= a < b <= count:
                raise ValueError('Lens boundaries require 1 <= front_surface < rear_surface <= surface count')
            if checked and a < checked[-1]['rear_surface']:
                raise ValueError('Lens declarations overlap or are out of order')
            checked.append(dict(front_surface=a, rear_surface=b,
                                **_limits({k: v for k, v in lens.items() if k in _LIMITS}, 'lens')))
        declarations = checked
    ambient = _number(value.get('ambient_index', 1.), 'ambient_index', True)
    return defaults, radii, declarations, spherical, ambient, samples


def _model(model):
    vertices = np.asarray(model['vertices_mm'], dtype=float)
    indices = np.asarray(model['indices_by_wave'], dtype=float)
    rows = model['surface_rows']
    if (vertices.ndim != 1 or not len(vertices) or not np.isfinite(vertices).all()
            or np.any(np.diff(vertices) < 0) or len(rows) != len(vertices)):
        raise ValueError('Use finite sequential vertices and aligned surface rows')
    if (indices.ndim != 2 or indices.shape[1] != len(vertices)+1 or not len(indices)
            or not np.isfinite(indices).all() or np.any(indices <= 0)):
        raise ValueError('Use finite positive refractive indices aligned with media')
    kinds = model.get('surface_types', ['STANDARD']*len(rows))
    if len(kinds) != len(rows) or any(k not in ('STANDARD', 'EVENASPH') for k in kinds):
        raise ValueError('Manufacturing audit supports STANDARD and EVENASPH surfaces')
    surfaces = []
    for z, row in zip(vertices, rows, strict=True):
        surfaces.append(Surface(float(z), 1., row.get('curvature_mm_inv', 0.), row.get('conic', 0.),
                                tuple(row.get('even_coefficients', ()))))
    return vertices, indices, rows, surfaces


def _infer(model, indices, ambient):
    names = model.get('material_names')
    if names is not None and (not isinstance(names, list) or len(names) != indices.shape[1]):
        raise ValueError('material_names must align with index media')
    result, start, previous = [], None, None
    for medium in range(1, indices.shape[1]):
        name = names[medium] if names is not None else None
        active = bool(name) if names is not None else not np.allclose(indices[:, medium], ambient, rtol=0., atol=1e-12)
        identity = name if names is not None else tuple(indices[:, medium])
        if start is not None and (not active or identity != previous):
            result.append(dict(front_surface=start, rear_surface=medium))
            start = None
        if active and start is None:
            start = medium
        previous = identity if active else None
    return result, start is not None


def _check_declarations(model, indices, declarations, ambient):
    names = model.get('material_names')
    if names is not None and (not isinstance(names, list) or len(names) != indices.shape[1]):
        raise ValueError('material_names must align with index media')
    covered = set()
    for lens in declarations:
        segment = range(lens['front_surface'], lens['rear_surface'])
        identities = []
        for i in segment:
            if names is not None:
                if not names[i]:
                    raise ValueError('Declared lens spans an ambient material gap')
                identities.append(names[i])
            else:
                identities.append(tuple(indices[:, i]))
            covered.add(i)
        if len(set(identities)) != 1:
            raise ValueError('Declared lens spans multiple materials; declare each cemented piece separately')
    for i in range(1, indices.shape[1]-1):
        active = bool(names[i]) if names is not None else not np.allclose(indices[:, i], ambient, rtol=0., atol=1e-12)
        if active and i not in covered:
            raise ValueError(f'Declared lenses omit nonambient material interval after S{i}')


def _sag(surface, radius):
    return surface.sag_and_gradient(np.asarray(radius)**2)[0]


def _stationary(left, right, radius):
    """Enumerate sag-difference stationary candidates by removing conic radicals."""
    square = radius*radius
    def term(s):
        c = s.curvature_mm_inv
        domain = Polynomial([1., -(1.+s.conic)*c*c*square])
        poly = Polynomial([(j+1)*a*square**j for j, a in enumerate(s.even_coefficients)] or [0.])
        return c/2., domain, poly
    a, da, pa = term(left)
    b, db, pb = term(right)
    p = pb-pa
    if a == 0. and b == 0.:
        equation = p
    elif a == 0.:
        equation = p*p*db-b*b
    elif b == 0.:
        equation = p*p*da-a*a
    else:
        equation = (p*p*da*db-a*a*db-b*b*da)**2-4.*a*a*b*b*da*db
    coefficients = np.asarray(equation.coef)
    if not np.isfinite(coefficients).all() or not np.any(coefficients):
        return np.empty(0)
    roots = Polynomial(coefficients/np.max(abs(coefficients))).trim(tol=1e-14).roots()
    good = (abs(roots.imag) < 1e-6) & (roots.real > 0.) & (roots.real < 1.)
    return radius*np.sqrt(roots.real[good])


def _golden(function, lo, hi):
    ratio = (math.sqrt(5.)-1.)/2.
    x, y = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fx, fy = function(x), function(y)
    for _ in range(48):
        if fx < fy:
            hi, y, fy = y, x, fx
            x = hi-ratio*(hi-lo)
            fx = function(x)
        else:
            lo, x, fx = x, y, fy
            y = lo+ratio*(hi-lo)
            fy = function(y)
    return x if fx < fy else y


def _minimum(left, right, radius, offset, samples):
    radii = np.linspace(0., radius, samples)
    function = lambda r: offset+_sag(right, r)-_sag(left, r)
    values = function(radii)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite solid-interval sag')
    candidates = [0., radius, *_stationary(left, right, radius)]
    local = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))+1
    candidates.extend(_golden(function, radii[i-1], radii[i+1]) for i in local)
    evaluated = function(np.asarray(candidates))
    if not np.isfinite(evaluated).all():
        raise ValueError('Nonfinite solid extrema')
    best = int(np.argmin(evaluated))
    return float(evaluated[best]), float(candidates[best])


def _audit_manufacturing(model, policy=None):
    """Return invalid, incomplete, or passed configured nominal solid geometry.

    Default missing envelopes/limits are incomplete. Known physical or configured
    violations are ineligible. Explicit lens boundaries identify cemented pieces,
    including index-matched glass. Shared surfaces are bonded interfaces.
    """
    report = dict(status='incomplete', eligible=True, passed=False, violations=[], missing=[],
                  surfaces=[], lenses=[], gaps=[], assumptions=[],
                  scope=('configured nominal coaxial solids and supplied scalar budgets; ray-envelope coverage '
                         'requires audit_ray_envelope; mounting, insertion, stress, tooling, coating and manufacturing yield not evaluated'))
    def issue(collection, code, detail, **extra):
        report[collection].append(dict(code=code, detail=detail, **extra))
    try:
        z, indices, rows, surfaces = _model(model)
        defaults, overrides, declarations, spherical, ambient, samples = _policy(policy, len(surfaces))
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
        issue('violations', 'invalid_input', str(error))
        report.update(status='invalid', eligible=False)
        return report
    report['method'] = dict(radial_samples=samples, extrema='polynomial candidates plus sampled local refinement',
                            interval_certified=False)
    if policy is None:
        issue('missing', 'manufacturing_policy_missing', 'Supply project/supplier dimensions and tolerance budgets')
    if not all(key in defaults for key in ('center_thickness_budget_mm', 'radial_thickness_budget_mm',
                                           'air_gap_budget_mm', 'rim_gap_budget_mm')):
        report['assumptions'].append('Unspecified scalar tolerance budgets are zero; a nominal pass does not establish toleranced clearance')
    if declarations is not None:
        try:
            _check_declarations(model, indices, declarations, ambient)
        except (ValueError, TypeError) as error:
            issue('violations', 'invalid_topology', str(error))
            report.update(status='invalid', eligible=False)
            return report
    if declarations is None:
        try:
            declarations, unclosed = _infer(model, indices, ambient)
        except (ValueError, TypeError) as error:
            issue('violations', 'invalid_topology', str(error))
            report.update(status='invalid', eligible=False)
            return report
        report['assumptions'].append('Lens topology inferred from named material or refractive-index runs')
        if model.get('material_names') is None:
            issue('missing', 'explicit_lens_topology_missing', 'Index-matched or cemented pieces require explicit lens declarations')
        if unclosed:
            issue('missing', 'unclosed_material_run', 'A material run reaches the last plane; declare closed physical solids')
        if not declarations:
            issue('missing', 'no_closed_solids', 'No closed physical lens identified')
    faces = {lens[key] for lens in declarations for key in ('front_surface', 'rear_surface')}
    lens_limits = [dict(defaults, **{k: v for k, v in lens.items() if k in _LIMITS})
                   for lens in declarations]
    report['limit_precedence'] = ('Lens entries override defaults for that lens. Shared faces use the '
                                  'tightest effective slope/margin; air and rim requirements apply to '
                                  'both adjacent ambient interfaces and use the tightest complete '
                                  '(minimum + budget) requirement from either bordering lens. '
                                  'Bonded shared faces have no air/rim gap requirement.')
    def face_limit(sn, key):
        owners = [limits[key] for lens, limits in zip(declarations, lens_limits, strict=True)
                  if sn in (lens['front_surface'], lens['rear_surface']) and key in limits]
        values = owners or ([defaults[key]] if key in defaults else [])
        return (min(values) if key.startswith('maximum_') else max(values)) if values else None

    clear, mechanical, usable = {}, {}, {}
    for sn, (row, surface) in enumerate(zip(rows, surfaces, strict=True), 1):
        override = overrides.get(sn, {})
        c = override.get('clear_radius_mm', row.get('clear_radius_mm', row.get('aperture_radius_mm')))
        m = override.get('mechanical_radius_mm', row.get('mechanical_radius_mm'))
        try:
            if c is not None:
                c = _number(c, f'S{sn}.clear_radius_mm', True)
            if m is not None:
                m = _number(m, f'S{sn}.mechanical_radius_mm', True)
        except ValueError as error:
            issue('violations', 'invalid_radius', str(error), surface=sn)
            continue
        clear[sn], mechanical[sn] = c, m
        if sn in faces:
            for value, code, detail in ((c, 'clear_radius_missing', 'Declare full clear aperture'),
                                        (m, 'mechanical_radius_missing', 'Declare full physical surface radius')):
                if value is None:
                    issue('missing', code, detail, surface=sn)
        radius = m if m is not None else c
        if spherical and (surface.conic != 0. or any(surface.even_coefficients)):
            issue('violations', 'asphere_not_allowed', 'Configured design permits only planes and spheres', surface=sn)
        if radius is None:
            continue
        diagnostic = dict(surface=sn, clear_radius_mm=c, mechanical_radius_mm=m)
        if c is not None and m is not None:
            margin = m-c
            diagnostic['clear_to_mechanical_radial_margin_mm'] = margin
            if margin < 0.:
                issue('violations', 'clear_outside_mechanical', 'Clear aperture exceeds physical radius',
                      surface=sn, value=margin, limit=0.)
            required = face_limit(sn, 'minimum_clear_to_mechanical_radial_margin_mm')
            diagnostic['effective_minimum_clear_to_mechanical_radial_margin_mm'] = required
            if required is None and sn in faces:
                issue('missing', 'radial_margin_limit_missing', 'Configure bevel and mounting radial allowance', surface=sn)
            elif required is not None and margin < required:
                issue('violations', 'radial_margin_below_limit', 'Insufficient radial allowance',
                      surface=sn, value=margin, limit=required)
        domain = 1.-(1.+surface.conic)*surface.curvature_mm_inv**2*radius**2
        if not math.isfinite(domain) or domain <= 0.:
            issue('violations', 'surface_domain_or_singular_slope', 'Undefined surface or singular slope over the declared radius',
                  surface=sn, radius_mm=radius)
            report['surfaces'].append(diagnostic)
            continue
        radial = np.linspace(0., radius, samples)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            sag, gradient = surface.sag_and_gradient(radial**2)
            slope = abs(2.*radial*gradient)
        if not np.isfinite(sag).all() or not np.isfinite(slope).all():
            issue('violations', 'nonfinite_surface', 'Nonfinite sag or slope on the declared surface', surface=sn)
            report['surfaces'].append(diagnostic)
            continue
        peak = int(np.argmax(slope))
        maximum_slope = float(slope[peak])
        if 0 < peak < samples-1:
            location = _golden(lambda r: -abs(2.*r*surface.sag_and_gradient(r*r)[1]), radial[peak-1], radial[peak+1])
            maximum_slope = max(maximum_slope, float(abs(2.*location*surface.sag_and_gradient(location*location)[1])))
        diagnostic['maximum_sampled_surface_slope'] = maximum_slope
        limit = face_limit(sn, 'maximum_surface_slope')
        diagnostic['effective_maximum_surface_slope'] = limit
        if limit is not None and maximum_slope > limit:
            issue('violations', 'surface_slope_above_limit', 'Slope exceeds configured process/metrology capability',
                  surface=sn, value=maximum_slope, limit=limit)
        usable[sn] = radius
        report['surfaces'].append(diagnostic)
    for number, declaration in enumerate(declarations, 1):
        a, b = declaration['front_surface'], declaration['rear_surface']
        limits = lens_limits[number-1]
        row = dict(lens=number, front_surface=a, rear_surface=b, effective_limits=limits)
        report['lenses'].append(row)
        center = float(z[b-1]-z[a-1])
        row['center_thickness_mm'] = center
        if center <= 0.:
            issue('violations', 'nonpositive_center_thickness', 'Physical lens center thickness must be positive',
                  lens=number, value=center, limit=0.)
        required = limits.get('minimum_center_thickness_mm')
        budget = limits.get('center_thickness_budget_mm', 0.)
        if required is None:
            issue('missing', 'center_thickness_limit_missing', 'Configure minimum center thickness', lens=number)
        elif center-budget < required:
            issue('violations', 'center_thickness_below_limit', 'Center thickness less budget misses limit',
                  lens=number, value=center-budget, limit=required)
        if a not in usable or b not in usable:
            continue
        ra, rb = mechanical.get(a), mechanical.get(b)
        radius = min(usable[a], usable[b])
        if ra is not None and rb is not None:
            row['mechanical_diameter_mm'] = 2.*max(ra, rb)
            if ra != rb:
                issue('missing', 'unequal_lens_face_radii', 'Unequal face radii require an explicit shoulder/edge solid model', lens=number)
            limit = limits.get('maximum_diameter_mm')
            if limit is not None and row['mechanical_diameter_mm'] > limit:
                issue('violations', 'diameter_above_limit', 'Diameter exceeds configured capability',
                      lens=number, value=row['mechanical_diameter_mm'], limit=limit)
        minimum, at = _minimum(surfaces[a-1], surfaces[b-1], radius, center, samples)
        row.update(common_radius_mm=radius, minimum_radial_thickness_mm=minimum, minimum_at_radius_mm=at,
                   edge_thickness_mm=float(center+_sag(surfaces[b-1], radius)-_sag(surfaces[a-1], radius)))
        if minimum <= 0.:
            issue('violations', 'nonpositive_radial_thickness', 'Lens surfaces meet or intersect on the common radius',
                  lens=number, value=minimum, radius_mm=at, limit=0.)
        required = limits.get('minimum_radial_thickness_mm')
        budget = limits.get('radial_thickness_budget_mm', 0.)
        if required is None:
            issue('missing', 'radial_thickness_limit_missing', 'Configure minimum full-radius material thickness', lens=number)
        elif minimum-budget < required:
            issue('violations', 'radial_thickness_below_limit', 'Material thickness less budget misses limit',
                  lens=number, value=minimum-budget, radius_mm=at, limit=required)
    for position, left in enumerate(report['lenses']):
        for right in report['lenses'][position+1:]:
            a, b = left['rear_surface'], right['front_surface']
            if a == b:
                report['gaps'].append(dict(left_lens=left['lens'], right_lens=right['lens'],
                                           kind='shared_cemented_interface', left_rear_surface=a, right_front_surface=b))
                continue
            if a not in usable or b not in usable:
                continue
            ra, rb = mechanical.get(a), mechanical.get(b)
            radius = min(usable[a], usable[b])
            gap = float(z[b-1]-z[a-1])
            minimum, at = _minimum(surfaces[a-1], surfaces[b-1], radius, gap, samples)
            row = dict(left_lens=left['lens'], right_lens=right['lens'], kind='separated_solids',
                       left_rear_surface=a, right_front_surface=b, minimum_gap_mm=minimum,
                       minimum_at_radius_mm=at, common_radius_mm=radius)
            report['gaps'].append(row)
            if minimum < 0.:
                issue('violations', 'solid_intersection', 'Solids intersect within their common physical radius',
                      left_lens=left['lens'], right_lens=right['lens'], value=minimum, radius_mm=at, limit=0.)
            if right['lens'] != left['lens']+1:
                continue
            intermediate = indices[:, a:b]
            if intermediate.size and not np.allclose(intermediate, ambient, rtol=0., atol=1e-12):
                issue('missing', 'nonambient_interface', 'Declare adhesive or immersion requirements for this interface',
                      left_lens=left['lens'], right_lens=right['lens'])
                row['kind'] = 'nonambient_interface'
                continue
            def interface_limit(minimum_key, budget_key):
                candidates = [(limits[minimum_key]+limits.get(budget_key, 0.), limits[minimum_key],
                               limits.get(budget_key, 0.), number)
                              for number, limits in ((left['lens'], lens_limits[left['lens']-1]),
                                                     (right['lens'], lens_limits[right['lens']-1]))
                              if minimum_key in limits]
                if not candidates:
                    return None, 0., None
                _, required, budget, owner = max(candidates)
                return required, budget, owner

            required, budget, owner = interface_limit('minimum_air_gap_mm', 'air_gap_budget_mm')
            row.update(effective_minimum_air_gap_mm=required, effective_air_gap_budget_mm=budget,
                       air_requirement_lens=owner)
            if required is None:
                issue('missing', 'air_gap_limit_missing', 'Configure minimum physical air separation',
                      left_lens=left['lens'], right_lens=right['lens'])
            elif minimum-budget < required:
                issue('violations', 'air_gap_below_limit', 'Air separation less budget misses limit',
                      left_lens=left['lens'], right_lens=right['lens'], value=minimum-budget, limit=required)
            if ra is None or rb is None:
                continue
            rim = float(gap+_sag(surfaces[b-1], rb)-_sag(surfaces[a-1], ra))
            row['separate_rim_axial_gap_mm'] = rim
            required, budget, owner = interface_limit('minimum_rim_gap_mm', 'rim_gap_budget_mm')
            row.update(effective_minimum_rim_gap_mm=required, effective_rim_gap_budget_mm=budget,
                       rim_requirement_lens=owner)
            if required is None:
                issue('missing', 'rim_gap_limit_missing', 'Configure mechanical-rim axial clearance',
                      left_lens=left['lens'], right_lens=right['lens'])
            elif rim-budget < required:
                issue('violations', 'rim_gap_below_limit', 'Separate mechanical-rim axial clearance misses limit',
                      left_lens=left['lens'], right_lens=right['lens'], value=rim-budget, limit=required)
    if report['violations']:
        report.update(status='invalid', eligible=False)
    elif not report['missing']:
        report.update(status='passed', passed=True)
    return report


def audit_manufacturing(model, policy=None):
    """Audit supported nominal geometry, returning structured numerical failures."""
    try:
        with np.errstate(over='raise', invalid='raise'):
            return _audit_manufacturing(model, policy)
    except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
        return dict(status='invalid', eligible=False, passed=False,
                    violations=[dict(code='numerical_geometry_failure', detail=str(error))],
                    missing=[], surfaces=[], lenses=[], gaps=[], assumptions=[],
                    scope='numerical geometry check could not complete')


def audit_ray_envelope(system, geometry, pupil, *, rim_samples=64, use_primary_field=False):
    """Join fresh full-path ray hits to the audited CA and physical radii.

    Check every declared field and wavelength, even when optical screening selects
    only a subset. Include chief and exact rim rays in addition to the supplied
    interior pupil nodes. This is a sampled check, not a continuous-domain proof.
    Optical apertures retain their original meaning; configured CA/OD never alter
    ray launching or clip the bundle to manufacture a passing result.
    """
    report = deepcopy(geometry)
    if not geometry['eligible']:
        return report
    pupil = np.asarray(pupil, dtype=float)
    if (pupil.ndim != 2 or pupil.shape[1] != 2 or not len(pupil)
            or not np.isfinite(pupil).all() or type(rim_samples) is not int or rim_samples < 8):
        raise ValueError('Envelope checking requires finite pupil XY nodes and at least eight rim samples')
    azimuth = 2.*np.pi*np.arange(rim_samples)/rim_samples
    nodes = np.vstack((np.zeros((1, 2)), pupil, np.column_stack((np.cos(azimuth), np.sin(azimuth)))))
    bounded = {row['surface']: row for row in geometry['surfaces']
               if row['clear_radius_mm'] is not None or row['mechanical_radius_mm'] is not None}
    envelope = dict(status='passed', backend='cpu_full_path', field_indices=list(range(len(system.fields))),
                    wave_indices=list(range(len(system.wavelengths))),
                    samples_per_group=len(nodes), rim_samples=rim_samples, chief_included=True,
                    field_reference='primary' if use_primary_field else 'measurement',
                    numerical_radius_tolerance_mm=1e-9, continuous_domain_certified=False,
                    surfaces=[], failed_groups=[])
    report['ray_envelope'] = envelope
    peaks = {sn: (-1., None, None, None) for sn in bounded}
    for fi, field in enumerate(system.fields):
        for wi in range(len(system.wavelengths)):
            try:
                origin, slopes = system.launch_field(field, None if use_primary_field else wi)
                traced = system._shoot(wi, nodes, origin, slopes, record_path=True)
                if not traced.valid.all() or np.any(traced.negative_segment_count):
                    bad = np.flatnonzero(~traced.valid)
                    detail = ('Negative propagation segment in physical envelope check' if not len(bad)
                              else f'Ray {bad[0]} failed at S{traced.failed_surface[bad[0]]+1}: {traced.reason[bad[0]]}')
                    envelope['failed_groups'].append(dict(field_index=fi, wave_index=wi, detail=detail))
                for sn in bounded:
                    xy = traced.surface_points[sn-1, :, :2]
                    radius = np.linalg.norm(xy, axis=-1)
                    finite = np.flatnonzero(np.isfinite(radius))
                    if not len(finite):
                        continue
                    ray = int(finite[np.argmax(radius[finite])])
                    if radius[ray] > peaks[sn][0]:
                        peaks[sn] = (float(radius[ray]), fi, wi, ray)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                envelope['failed_groups'].append(dict(field_index=fi, wave_index=wi, detail=str(error)))
    for sn, dimensions in bounded.items():
        radius, fi, wi, ray = peaks[sn]
        item = dict(surface=sn, maximum_hit_radius_mm=None if radius < 0. else radius,
                    field_index=fi, wave_index=wi, pupil_index=ray)
        for key, code in (('clear_radius_mm', 'ray_outside_clear_aperture'),
                          ('mechanical_radius_mm', 'ray_outside_mechanical_radius')):
            limit = dimensions[key]
            item[key] = limit
            if limit is not None and radius >= 0.:
                item[key.replace('radius_mm', 'ray_margin_mm')] = limit-radius
                if radius > limit+envelope['numerical_radius_tolerance_mm']:
                    report['violations'].append(dict(code=code, detail='Fresh traced ray exceeds configured surface radius',
                                                      surface=sn, value=radius, limit=limit,
                                                      field_index=fi, wave_index=wi, pupil_index=ray))
        envelope['surfaces'].append(item)
    if envelope['failed_groups']:
        report['violations'].append(dict(code='ray_envelope_trace_failed',
                                          detail='At least one full-field/full-spectrum envelope bundle failed; '
                                                 'see ray_envelope.failed_groups'))
    if report['violations']:
        envelope['status'] = 'invalid'
        report.update(status='invalid', eligible=False, passed=False)
    report['scope'] = ('configured nominal coaxial solids, scalar budgets and sampled fresh ray-envelope coverage; '
                       'mounting, insertion, stress, tooling, coating and manufacturing yield not evaluated')
    return report
