"""Analytic forward tangents for the supported coaxial refractive ray tracer.

These are derivatives of *existing rays on their current smooth branch*.
They do not differentiate the population/weights of rays at a moving aperture,
re-aim a launch automatically, or certify manufacturing constraints. Use dense
independent tracing for candidate acceptance. All lengths are in millimetres.
"""

from dataclasses import dataclass

import numpy as np

from .standalone import TraceResult, trace


@dataclass(frozen=True)
class RayParameter:
    """A tangent direction in a surface attribute; surface_index is zero-based.

    ``gap_before_mm`` translates this and every subsequent surface. In contrast,
    ``vertex_mm`` moves just the named surface. ``even_coefficient`` uses the
    zero-based A2/A4/... coefficient_index. ``scale`` maps a dimensionless design
    parameter to a physical attribute. Repeated parameters are independent
    columns, and directions may be linearly combined by the caller.
    """

    surface_index: int
    attribute: str
    coefficient_index: int | None = None
    scale: float = 1.0


@dataclass
class TraceDerivativeResult:
    trace: TraceResult
    points_jacobian: np.ndarray
    directions_jacobian: np.ndarray
    opl_jacobian: np.ndarray
    differentiable: np.ndarray
    derivative_reason: np.ndarray
    parameters: tuple[RayParameter, ...]


@dataclass
class WavefrontDerivativeResult:
    rms_waves: float
    residual_waves: np.ndarray
    rms_jacobian: np.ndarray
    residual_jacobian: np.ndarray
    rms_differentiable: bool


def _parameter_table(surfaces, parameters):
    parameters = tuple(parameters)
    count, width = len(surfaces), len(parameters)
    table = {key: np.zeros((count, width)) for key in
             ('vertex_mm', 'curvature_mm_inv', 'conic', 'index_after')}
    coefficients = [np.zeros((len(sf.even_coefficients), width)) for sf in surfaces]
    for column, parameter in enumerate(parameters):
        if not isinstance(parameter, RayParameter):
            raise TypeError('Use RayParameter objects')
        i, key = parameter.surface_index, parameter.attribute
        if (isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < count
                or not np.isfinite(parameter.scale)):
            raise ValueError('Invalid parameter surface index or scale')
        if key == 'gap_before_mm':
            table['vertex_mm'][i:, column] = parameter.scale
        elif key == 'even_coefficient':
            k = parameter.coefficient_index
            if (isinstance(k, bool) or not isinstance(k, (int, np.integer))
                    or not 0 <= k < len(coefficients[i])):
                raise ValueError('Coefficient index must identify an existing A2/A4/... entry')
            coefficients[i][k, column] = parameter.scale
        elif key in table:
            if parameter.coefficient_index is not None:
                raise ValueError('coefficient_index applies only to even_coefficient')
            table[key][i, column] = parameter.scale
        else:
            raise ValueError(f'Unsupported differentiated attribute: {key}')
    return parameters, table, coefficients


def _tangent(value, shape, name):
    if value is None:
        return np.zeros(shape)
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return result.copy()


def _sag_terms(surface, rho, dc, dk, da):
    """Return sag rho derivatives and explicit attribute partials."""
    c, k = surface.curvature_mm_inv, surface.conic
    q = np.sqrt(1. - (1. + k)*c*c*rho)
    sr = c/(2.*q)
    srr = (1. + k)*c**3/(4.*q**3)
    sp = rho[:, None]/(q*(1.+q))[:, None]*dc
    sp += (c**3*rho*rho/(2.*q*(1.+q)**2))[:, None]*dk
    srp = (.5/q + (1.+k)*c*c*rho/(2.*q**3))[:, None]*dc
    srp += (c**3*rho/(4.*q**3))[:, None]*dk
    for j, a in enumerate(surface.even_coefficients):
        power = j + 1
        sr += power*a*rho**(power-1)
        if power > 1:
            srr += power*(power-1)*a*rho**(power-2)
        sp += rho[:, None]**power*da[j]
        srp += power*rho[:, None]**(power-1)*da[j]
    return sr, srr, sp, srp, q


def trace_with_jacobian(surfaces, origins, directions, parameters, object_index=1., *,
                        origin_jacobian=None, direction_jacobian=None,
                        object_index_jacobian=None, boundary_tolerance_mm=1e-7,
                        angular_tolerance=1e-10, record_path=False):
    """Trace fixed launched rays and propagate analytic parameter tangents.

    Jacobian axes are (ray, XYZ, parameter), or (ray, parameter) for OPL.
    Input direction tangents are for the raw directions before normalization.
    A valid primal ray can have undefined derivatives at an aperture edge,
    grazing intersection, zero-length segment, TIR boundary or singular conic.
    Such columns return NaN and an explicit differentiable=False ray mask.
    Negative/signed propagation is deliberately not supported here.
    """
    surfaces = tuple(surfaces)
    parameters, table, coefficients = _parameter_table(surfaces, parameters)
    if (not np.isfinite(boundary_tolerance_mm) or boundary_tolerance_mm < 0
            or not np.isfinite(angular_tolerance) or angular_tolerance <= 0):
        raise ValueError('Use nonnegative distance and positive angular boundary tolerances')
    primal = trace(surfaces, origins, directions, object_index, record_path=True)
    raw = np.asarray(directions, dtype=float)
    count, width = len(raw), len(parameters)
    length = np.linalg.norm(raw, axis=1)
    unit = raw/length[:, None]
    dd = _tangent(direction_jacobian, (count, 3, width), 'direction_jacobian')
    dd = (dd-unit[..., None]*np.einsum('ni,nip->np', unit, dd)[:, None, :])/length[:, None, None]
    dp = _tangent(origin_jacobian, (count, 3, width), 'origin_jacobian')
    dn_before = _tangent(object_index_jacobian, (width,), 'object_index_jacobian')
    origins = np.broadcast_to(np.asarray(origins, dtype=float), raw.shape).copy()
    points, outgoing = origins, unit
    dopl = np.zeros((count, width))
    good = primal.valid.copy()
    reasons = np.where(good, '', 'invalid_primal').astype('U32')
    n_before = float(object_index)

    def mark(mask, reason):
        newly = good & mask
        reasons[newly] = reason
        good[newly] = False

    # Trace only globally valid rays in the tangent pass. This never changes
    # the independent primal failure classification or its ray population.
    active = np.flatnonzero(primal.valid)
    for i, sf in enumerate(surfaces):
        if not len(active):
            break
        p, d, pd, dtan = points[active], outgoing[active], dp[active], dd[active]
        hit = primal.surface_points[i, active]
        # The primal trace already selected the branch. Unit d makes this the
        # signed physical distance; avoid solving Newton a second time.
        distance = np.sum((hit-p)*d, axis=1)
        rho = np.sum(hit[:, :2]**2, axis=1)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            sr, srr, sp, srp, q = _sag_terms(sf, rho, table['curvature_mm_inv'][i],
                                            table['conic'][i], coefficients[i])
            denominator = d[:, 2]-2.*sr*np.sum(hit[:, :2]*d[:, :2], axis=1)
            free = pd+distance[:, None, None]*dtan
            ds = (table['vertex_mm'][i]+sp+2.*sr[:, None]*np.einsum(
                'ni,nip->np', hit[:, :2], free[:, :2])-free[:, 2])/denominator[:, None]
            hd = free+d[..., None]*ds[:, None, :]
            drho = 2.*np.einsum('ni,nip->np', hit[:, :2], hd[:, :2])
            dsr = srr[:, None]*drho+srp
            normal = np.column_stack((-2.*hit[:, :2]*sr[:, None], np.ones(len(active))))
            nd = np.zeros_like(hd)
            nd[:, :2] = -2.*(hd[:, :2]*sr[:, None, None]+hit[:, :2, None]*dsr[:, None, :])
            norm = np.linalg.norm(normal, axis=1)
            normal /= norm[:, None]
            nd = (nd-normal[..., None]*np.einsum('ni,nip->np', normal, nd)[:, None, :])/norm[:, None, None]
            cosine = np.sum(d*normal, axis=1)
            sign = np.where(cosine < 0, -1., 1.)
            normal *= sign[:, None]
            nd *= sign[:, None, None]
            cosine = abs(cosine)
            cd = np.einsum('nip,ni->np', dtan, normal)+np.einsum('ni,nip->np', d, nd)
            eta = n_before/sf.index_after
            dn_after = table['index_after'][i]
            deta = dn_before/sf.index_after-n_before/sf.index_after**2*dn_after
            tangent = eta*(d-cosine[:, None]*normal)
            td = deta[None, None, :]*(d-cosine[:, None]*normal)[..., None]+eta*(
                dtan-cd[:, None, :]*normal[..., None]-cosine[:, None, None]*nd)
            root = np.sqrt(np.maximum(1.-np.sum(tangent*tangent, axis=1), 0.))
            root_d = -np.einsum('ni,nip->np', tangent, td)/root[:, None]
            new_d = tangent+root[:, None]*normal
            new_dd = td+root_d[:, None, :]*normal[..., None]+root[:, None, None]*nd
        small_segment = np.zeros(count, dtype=bool)
        small_segment[active] = abs(distance) <= boundary_tolerance_mm
        mark(small_segment, 'segment_boundary')
        singular = np.zeros(count, dtype=bool)
        singular[active] = ((abs(denominator) <= angular_tolerance) | (q <= angular_tolerance)
                            | (root <= angular_tolerance) | ~np.isfinite(hd).all(axis=(1, 2))
                            | ~np.isfinite(new_dd).all(axis=(1, 2)))
        mark(singular, 'singular_optical_branch')
        if np.isfinite(sf.aperture_radius_mm):
            edge = np.zeros(count, dtype=bool)
            edge[active] = abs(np.sqrt(rho)-sf.aperture_radius_mm) <= boundary_tolerance_mm
            mark(edge, 'aperture_boundary')
        dopl[active] += n_before*ds+distance[:, None]*dn_before
        points[active], dp[active] = hit, hd
        outgoing[active], dd[active] = new_d, new_dd
        n_before, dn_before = sf.index_after, dn_after
    dp[~good], dd[~good], dopl[~good] = np.nan, np.nan, np.nan
    if not record_path:
        primal.surface_points = None
    return TraceDerivativeResult(primal, dp, dd, dopl, good, reasons, parameters)


def paraxial_transfer_with_jacobian(surfaces, origin_z_mm, parameters, object_index=1., *,
                                    origin_z_jacobian=None, object_index_jacobian=None):
    """First-order transfer and exact tangents, including gap/index derivatives."""
    surfaces = tuple(surfaces)
    parameters, table, coefficients = _parameter_table(surfaces, parameters)
    width = len(parameters)
    matrix, jac = np.eye(2), np.zeros((2, 2, width))
    position, n = float(origin_z_mm), float(object_index)
    dz = _tangent(origin_z_jacobian, (width,), 'origin_z_jacobian')
    dn = _tangent(object_index_jacobian, (width,), 'object_index_jacobian')
    if not np.isfinite(position) or not np.isfinite(n) or n <= 0:
        raise ValueError('Use a finite origin and positive object index')
    for i, sf in enumerate(surfaces):
        gap = sf.vertex_mm-position
        if gap < 0:
            raise ValueError('Paraxial surfaces must follow the starting plane in order')
        dg = table['vertex_mm'][i]-dz
        drift = np.array([[1., gap/n], [0., 1.]])
        dj = np.zeros((2, 2, width))
        dj[0, 1] = dg/n-gap/n**2*dn
        jac = np.einsum('abp,bc->acp', dj, matrix)+np.einsum('ab,bcp->acp', drift, jac)
        matrix = drift@matrix
        c = sf.curvature_mm_inv+(2.*sf.even_coefficients[0] if len(sf.even_coefficients) else 0.)
        dc = table['curvature_mm_inv'][i]+(2.*coefficients[i][0] if len(coefficients[i]) else 0.)
        dnext = table['index_after'][i]
        power = (sf.index_after-n)*c
        dpower = (dnext-dn)*c+(sf.index_after-n)*dc
        bend = np.array([[1., 0.], [-power, 1.]])
        bj = np.zeros((2, 2, width))
        bj[1, 0] = -dpower
        jac = np.einsum('abp,bc->acp', bj, matrix)+np.einsum('ab,bcp->acp', bend, jac)
        matrix = bend@matrix
        position, dz = sf.vertex_mm, table['vertex_mm'][i]
        n, dn = sf.index_after, dnext
    return matrix, jac


def reference_opd_with_jacobian(bundle, chief, pupil_distance_mm, image_index=1., *,
                                 pupil_distance_jacobian=None, image_index_jacobian=None,
                                 center=None, center_jacobian=None):
    """Chief-referenced sphere OPD and exact tangents, including moving center.

    ``pupil_distance_jacobian`` must include any paraxial reference movement.
    Its omission explicitly means a fixed reference, never automatic re-aiming.
    """
    if bundle.parameters != chief.parameters or chief.trace.valid.shape != (1,):
        raise ValueError('Bundle and one chief must share parameter columns')
    if not bundle.differentiable.all() or not chief.differentiable.all():
        raise ValueError('Wavefront derivatives require differentiable valid rays')
    width = len(bundle.parameters)
    signed = float(pupil_distance_mm)
    if not np.isfinite(signed) or signed == 0 or not np.isfinite(image_index) or image_index <= 0:
        raise ValueError('Use a nonzero finite reference distance and positive image index')
    radius, sign = abs(signed), np.sign(signed)
    dr = sign*_tangent(pupil_distance_jacobian, (width,), 'pupil_distance_jacobian')
    dn = _tangent(image_index_jacobian, (width,), 'image_index_jacobian')
    if center is None:
        if center_jacobian is not None:
            raise ValueError('Specify an explicit center when overriding its Jacobian')
        center, dc = chief.trace.points[0], chief.points_jacobian[0]
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Center must contain finite XYZ coordinates')
        dc = _tangent(center_jacobian, (3, width), 'center_jacobian')

    def correction(value):
        points, directions = value.trace.points-center, value.trace.directions
        dp = value.points_jacobian-dc
        dd = value.directions_jacobian
        proj = np.sum(points*directions, axis=1)
        dproj = np.einsum('nip,ni->np', dp, directions)+np.einsum('ni,nip->np', points, dd)
        perp = points-proj[:, None]*directions
        dperp = dp-dproj[:, None, :]*directions[..., None]-proj[:, None, None]*dd
        square = np.sum(perp*perp, axis=1)
        dsquare = 2.*np.einsum('ni,nip->np', perp, dperp)
        domain = radius*radius-square
        if np.any(domain <= 1e-20):
            raise ValueError('Reference sphere derivative is singular or has no intersection')
        root = np.sqrt(domain)
        droot = (radius*dr-.5*dsquare)/root[:, None]
        denom = radius+root
        corr = -proj+sign*square/denom
        deriv = -dproj+sign*(dsquare/denom[:, None]-square[:, None]*(dr+droot)/denom[:, None]**2)
        return corr, deriv
    a, da = correction(bundle)
    b, db = correction(chief)
    diff, ddiff = a-b[0], da-db[0]
    opd = bundle.trace.opl_mm-chief.trace.opl_mm[0]+image_index*diff
    jac = bundle.opl_jacobian-chief.opl_jacobian[0]+image_index*ddiff+diff[:, None]*dn
    return opd, jac


def wavefront_rms_with_jacobian(opd_mm, opd_jacobian, weights, wavelength_um,
                               tilt_coordinates=None, *, tilt_jacobian=None,
                               wavelength_jacobian=None):
    """Differentiate weighted piston/tilt least squares and RMS in waves."""
    opd, jac, weights = (np.asarray(v, dtype=float) for v in (opd_mm, opd_jacobian, weights))
    if (opd.ndim != 1 or jac.ndim != 2 or jac.shape[0] != len(opd) or weights.shape != opd.shape
            or not all(np.isfinite(v).all() for v in (opd, jac, weights))
            or np.any(weights < 0) or weights.sum() <= 0
            or not np.isfinite(wavelength_um) or wavelength_um <= 0):
        raise ValueError('Use finite OPD/Jacobian, nonnegative weights and positive wavelength')
    weights = weights/weights.sum()
    width = jac.shape[1]
    basis = np.ones((len(opd), 1))
    db = np.zeros((len(opd), 1, width))
    if tilt_coordinates is not None:
        coords = np.asarray(tilt_coordinates, dtype=float)
        if coords.shape != (len(opd), 2) or not np.isfinite(coords).all():
            raise ValueError('Tilt coordinates must contain finite XY pairs')
        basis = np.column_stack((basis, coords))
        db = np.concatenate((db, _tangent(tilt_jacobian, (len(opd), 2, width), 'tilt_jacobian')), axis=1)
    elif tilt_jacobian is not None:
        raise ValueError('Tilt coordinate Jacobian requires tilt coordinates')
    root_weight = np.sqrt(weights)
    fit, _, rank, _ = np.linalg.lstsq(basis*root_weight[:, None], opd*root_weight, rcond=None)
    if rank != basis.shape[1]:
        raise ValueError('Pupil does not identify requested reference terms')
    # Differentiate normal equations with the residual term retained; simply
    # projecting dopd would omit the derivative of a moving direction basis.
    residual = opd-basis@fit
    free = jac-np.einsum('nip,i->np', db, fit)
    gram = basis.T@(weights[:, None]*basis)
    rhs = np.einsum('nip,n,n->ip', db, weights, residual)+basis.T@(weights[:, None]*free)
    dfit = np.linalg.solve(gram, rhs)
    deriv = free-basis@dfit
    scale = wavelength_um*1e-3
    dw = _tangent(wavelength_jacobian, (width,), 'wavelength_jacobian')*1e-3
    residual_waves = residual/scale
    dres = deriv/scale-residual[:, None]*dw/scale**2
    rms = float(np.sqrt(np.sum(weights*residual_waves**2)))
    regular = rms > 1e-14
    drms = np.einsum('n,n,np->p', weights, residual_waves, dres)/rms if regular else np.full(width, np.nan)
    return WavefrontDerivativeResult(rms, residual_waves, drms, dres, regular)
