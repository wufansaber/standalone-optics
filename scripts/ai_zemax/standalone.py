"""Sequential, coaxial real-ray optics using NumPy. All geometry is in mm."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Surface:
    vertex_mm: float
    index_after: float = 1.0
    curvature_mm_inv: float = 0.0
    conic: float = 0.0
    even_coefficients: tuple[float, ...] = ()  # A2, A4, A6, ...
    aperture_radius_mm: float = np.inf

    def __post_init__(self):
        values = (self.vertex_mm, self.index_after, self.curvature_mm_inv,
                  self.conic, *self.even_coefficients)
        if not np.isfinite(values).all() or self.index_after <= 0 or not self.aperture_radius_mm > 0:
            raise ValueError("Invalid surface geometry, index, or aperture")

    def sag_and_gradient(self, radius_squared):
        """Return sag and its derivative with respect to radius squared."""
        c = self.curvature_mm_inv
        domain = 1.0 - (1.0 + self.conic) * c * c * radius_squared
        with np.errstate(invalid="ignore", divide="ignore"):
            root = np.sqrt(domain)
            sag = c * radius_squared / (1.0 + root)
            derivative = c / (2.0 * root)
        for power, coefficient in enumerate(self.even_coefficients, 1):
            sag = sag + coefficient * radius_squared ** power
            derivative = derivative + power * coefficient * radius_squared ** (power - 1)
        return sag, derivative


@dataclass
class TraceResult:
    points: np.ndarray
    directions: np.ndarray
    opl_mm: np.ndarray
    valid: np.ndarray
    failed_surface: np.ndarray
    reason: np.ndarray
    surface_points: np.ndarray | None = None
    negative_segment_count: np.ndarray | None = None
    shortest_segment_mm: np.ndarray | None = None
    shortest_segment_surface: np.ndarray | None = None


def sellmeier_index(wavelength_um, strengths, resonance_um_squared):
    """Evaluate n² = 1 + sum(B·λ²/(λ²−C)) with coefficients supplied by the caller."""
    strengths = np.asarray(strengths, dtype=float)
    resonances = np.asarray(resonance_um_squared, dtype=float)
    if (strengths.ndim != 1 or not len(strengths) or strengths.shape != resonances.shape
            or not np.isfinite(strengths).all() or not np.isfinite(resonances).all()
            or not np.isfinite(wavelength_um) or wavelength_um <= 0):
        raise ValueError("Sellmeier coefficients and a positive wavelength are required")
    wavelength_squared = wavelength_um ** 2
    denominators = wavelength_squared - resonances
    if np.any(denominators == 0):
        raise ValueError("Wavelength lies at a Sellmeier resonance")
    index_squared = 1.0 + np.sum(strengths * wavelength_squared / denominators)
    if not np.isfinite(index_squared) or index_squared <= 0:
        raise ValueError("Sellmeier equation did not produce a physical refractive index")
    return float(np.sqrt(index_squared))


def _conic_start(surface, point, direction, allow_signed_segments=False):
    """Choose the forward vertex-branch intersection of the base conic."""
    c = surface.curvature_mm_inv
    plane = -point[:, 2] / direction[:, 2]
    if c == 0:
        return plane
    scale = np.array([1.0, 1.0, 1.0 + surface.conic])
    a = c * np.sum(direction * direction * scale, axis=1)
    b = c * np.sum(point * direction * scale, axis=1) - direction[:, 2]
    f = c * np.sum(point * point * scale, axis=1) - 2.0 * point[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = -b - np.copysign(np.sqrt(b * b - a * f), b)
        roots = np.stack((q / a, f / q))
        roots[0] = np.where(abs(a) < 1e-15, -f / (2.0 * b), roots[0])
    z = point[:, 2] + roots * direction[:, 2]
    admissible = (c * (1.0 + surface.conic) * z <= 1.0 + 1e-12) & np.isfinite(roots)
    if allow_signed_segments:
        score = np.where(admissible,abs(roots-plane),np.inf)
        nearest_index = np.argmin(score,axis=0)
        nearest = roots[nearest_index,np.arange(len(plane))]
        return np.where(np.any(admissible,axis=0),nearest,plane)
    admissible &= roots >= -1e-9
    nearest = np.min(np.where(admissible, roots, np.inf), axis=0)
    return np.where(np.isfinite(nearest), nearest, plane)


def _intersect(surface, point, direction, allow_signed_segments=False):
    local = point.copy()
    local[:, 2] -= surface.vertex_mm
    distance = _conic_start(surface, local, direction, allow_signed_segments)
    # Newton solves the actual sag, including even aspheric terms. A root is
    # accepted only on the forward sag branch and after a residual check.
    for _ in range(24):
        hit = local + distance[:, None] * direction
        radius_squared = np.sum(hit[:, :2] ** 2, axis=1)
        sag, derivative = surface.sag_and_gradient(radius_squared)
        residual = hit[:, 2] - sag
        converged = np.isfinite(residual) & (abs(residual) <= 2e-12)
        slope = direction[:, 2] - 2.0 * derivative * np.sum(hit[:, :2] * direction[:, :2], axis=1)
        active = ~converged & np.isfinite(slope) & (abs(slope) > 1e-13)
        if not np.any(active):
            break
        distance[active] -= residual[active] / slope[active]
    hit = local + distance[:, None] * direction
    sag, derivative = surface.sag_and_gradient(np.sum(hit[:, :2] ** 2, axis=1))
    normal = np.column_stack((-2.0 * hit[:, :2] * derivative[:, None], np.ones(len(hit))))
    good = (abs(hit[:, 2] - sag) <= 1e-10) & np.isfinite(normal).all(axis=1)
    hit[:, 2] += surface.vertex_mm
    return hit, normal, distance, good


def refract(direction, normal, index_before, index_after):
    """Apply vector Snell refraction to unit directions and flag total internal reflection."""
    direction = np.asarray(direction, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    cosine = np.sum(direction * normal, axis=-1)
    normal = np.where((cosine < 0)[..., None], -normal, normal)
    tangential = (index_before / index_after) * (direction - abs(cosine)[..., None] * normal)
    normal_squared = 1.0 - np.sum(tangential * tangential, axis=-1)
    outgoing = tangential + np.sqrt(np.maximum(normal_squared, 0.0))[..., None] * normal
    return outgoing, normal_squared >= -1e-14


def trace(surfaces, origins, directions, object_index=1.0, *, record_path=False,
          allow_signed_segments=False):
    """Trace ordered transmissive surfaces. Signed segments require explicit opt-in."""
    surfaces = tuple(surfaces)
    if not surfaces or any(b.vertex_mm < a.vertex_mm for a, b in zip(surfaces, surfaces[1:])):
        raise ValueError("Provide surfaces in sequential order")
    if not np.isfinite(object_index) or object_index <= 0:
        raise ValueError("Object index must be positive and finite")
    points = np.array(origins, dtype=float, copy=True)
    directions = np.array(directions, dtype=float, copy=True)
    if directions.ndim != 2 or directions.shape[1] != 3 or not len(directions):
        raise ValueError("Directions must have shape (N, 3)")
    if points.shape == (3,):
        points = np.broadcast_to(points, directions.shape).copy()
    if points.shape != directions.shape or not np.isfinite(points).all() or not np.isfinite(directions).all():
        raise ValueError("Origins and directions must contain finite XYZ coordinates")
    if np.any(directions[:, 2] <= 0):
        raise ValueError("This sequential tracer requires forward launch directions")
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    count = len(points)
    optical_path = np.zeros(count)
    failed_surface = np.full(count, -1, dtype=int)
    reason = np.full(count, "", dtype="U16")
    history = np.full((len(surfaces), count, 3), np.nan) if record_path else None
    negative_count = np.zeros(count,dtype=int)
    shortest_segment = np.full(count,np.inf)
    shortest_surface = np.full(count,-1,dtype=int)
    index_before = object_index
    for number, surface in enumerate(surfaces):
        active = np.flatnonzero(failed_surface == -1)
        if not len(active):
            break
        incident = directions[active]
        hit, normal, distance, good = _intersect(surface, points[active], incident, allow_signed_segments)
        negative = good & (distance < -1e-9)
        if not allow_signed_segments:
            good &= ~negative
        clipped = good & (np.linalg.norm(hit[:, :2], axis=1) > surface.aperture_radius_mm + 1e-9)
        with np.errstate(invalid="ignore"):
            outgoing, transmits = refract(incident, normal, index_before, surface.index_after)
        valid = good & ~clipped & transmits & (outgoing[:, 2] > 0)
        reason[active[~good]] = "no_intersection"
        if not allow_signed_segments:
            reason[active[negative]] = "negative_segment"
        reason[active[clipped]] = "aperture"
        reason[active[good & ~clipped & ~transmits]] = "tir"
        reason[active[good & ~clipped & transmits & ~valid]] = "backward"
        failed_surface[active[~valid]] = number
        accepted = active[valid]
        lengths = distance[valid]
        negative_count[accepted] += lengths < -1e-9
        shortest_surface[accepted[lengths < shortest_segment[accepted]]] = number
        shortest_segment[accepted] = np.minimum(shortest_segment[accepted],lengths)
        if not allow_signed_segments:
            lengths = np.maximum(lengths,0.)
        optical_path[accepted] += index_before * lengths
        points[accepted] = hit[valid]
        directions[accepted] = outgoing[valid]
        if history is not None:
            history[number, accepted] = hit[valid]
        index_before = surface.index_after
    valid = failed_surface == -1
    points[~valid] = np.nan
    directions[~valid] = np.nan
    optical_path[~valid] = np.nan
    return TraceResult(points, directions, optical_path, valid, failed_surface, reason, history,
                       negative_count, shortest_segment, shortest_surface)


def disk_quadrature(radial_order=12, azimuths=32):
    """Unit-disk quadrature with weights normalized by pupil area."""
    if radial_order < 1 or azimuths < 3:
        raise ValueError("Use at least one radial node and three azimuths")
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    angle = np.arange(azimuths) * (2.0 * np.pi / azimuths)
    radius = np.sqrt((nodes + 1.0) / 2.0)
    pupil = (radius[:, None, None] * np.column_stack((np.cos(angle), np.sin(angle)))).reshape(-1, 2)
    return pupil, np.repeat(weights / (2.0 * azimuths), azimuths)


def directions_from_slopes(slopes):
    slopes = np.asarray(slopes, dtype=float)
    directions = np.column_stack((slopes, np.ones(len(slopes))))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def paraxial_transfer(surfaces, origin_z_mm, object_index=1.0):
    """First-order matrix from the starting plane to just after the last surface."""
    matrix = np.eye(2)
    position = origin_z_mm
    index = object_index
    for surface in surfaces:
        gap = surface.vertex_mm - position
        if gap < 0:
            raise ValueError("Paraxial surfaces must follow the starting plane in order")
        matrix = np.array([[1.0, gap / index], [0.0, 1.0]]) @ matrix
        curvature = surface.curvature_mm_inv
        if surface.even_coefficients:
            curvature += 2.0 * surface.even_coefficients[0]
        power = (surface.index_after - index) * curvature
        matrix = np.array([[1.0, 0.0], [-power, 1.0]]) @ matrix
        position, index = surface.vertex_mm, surface.index_after
    return matrix


def aim_at_stop(surfaces_to_stop, origin, target_xy, initial_slopes, object_index=1.0):
    """Solve launch slopes for prescribed XY positions on a real stop."""
    surfaces_to_stop = tuple(surfaces_to_stop)
    target_xy = np.asarray(target_xy, dtype=float)
    slopes = np.array(initial_slopes, dtype=float, copy=True)
    if target_xy.shape != slopes.shape or slopes.ndim != 2 or slopes.shape[1] != 2:
        raise ValueError("Stop targets and initial slopes must have shape (N, 2)")
    # Trial rays may cross a clear-aperture edge while solving. The final full
    # trace applies every aperture and retains per-ray failure information.
    trial_surfaces = tuple(replace(surface, aperture_radius_mm=np.inf) for surface in surfaces_to_stop)

    def intercept(estimate):
        result = trace(trial_surfaces, origin, directions_from_slopes(estimate), object_index)
        return result.points[:, :2]

    epsilon = 2e-6
    for _ in range(18):
        positions = intercept(slopes)
        error = positions - target_xy
        if not np.isfinite(error).all():
            raise RuntimeError("Stop aiming trial did not reach the stop")
        if np.max(np.linalg.norm(error, axis=1)) < 1e-9:
            return directions_from_slopes(slopes)
        jacobian = np.stack([(intercept(slopes + delta) - positions) / epsilon
                             for delta in (np.array([epsilon, 0.0]), np.array([0.0, epsilon]))], axis=-1)
        try:
            step = np.linalg.solve(jacobian, error[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Singular stop aiming Jacobian") from exc
        # Bound each Newton change. A failed step is reported rather than
        # silently substituting a paraxial launch for an extreme field.
        scale = np.maximum(1.0, np.linalg.norm(step, axis=1) / 0.25)
        slopes -= step / scale[:, None]
    raise RuntimeError("Stop aiming did not converge to 1e-9 mm")


def reference_opd_mm(result, chief, pupil_distance_mm, image_index=1.0, *, center=None):
    """Chief-referenced OPD at a sphere; positive distance selects its upstream branch."""
    if not np.all(result.valid) or chief.valid.shape != (1,) or not chief.valid[0]:
        raise ValueError("Reference wavefront needs valid rays across the requested pupil")
    radius = abs(float(pupil_distance_mm))
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(image_index) or image_index <= 0:
        raise ValueError("Use a finite, nonzero signed reference distance and positive index")
    center = np.asarray(chief.points[0] if center is None else center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Reference sphere center must contain three finite coordinates")

    def correction(rays):
        relative = rays.points - center
        projection = np.sum(relative * rays.directions, axis=1)
        perpendicular = relative - projection[:, None] * rays.directions
        perpendicular_squared = np.sum(perpendicular ** 2, axis=1)
        if np.any(perpendicular_squared > radius ** 2):
            raise ValueError("Some rays do not intersect the reference sphere")
        root = np.sqrt(radius ** 2 - perpendicular_squared)
        # Rationalize R - sqrt(R² - d²) to retain precision for a distant pupil.
        return -projection + np.sign(pupil_distance_mm) * perpendicular_squared / (radius + root)

    return (result.opl_mm - chief.opl_mm[0]) + image_index * (correction(result) - correction(chief)[0])


def wavefront_rms(opd_mm, weights, wavelength_um, tilt_coordinates=None):
    """Return weighted RMS and residual samples in waves. Always remove piston."""
    opd = np.asarray(opd_mm, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (opd.ndim != 1 or opd.shape != weights.shape or not np.isfinite(opd).all()
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0
            or not np.isfinite(wavelength_um) or wavelength_um <= 0):
        raise ValueError("Finite OPD samples, positive total weight, and positive wavelength required")
    weights = weights / weights.sum()
    basis = np.ones((len(opd), 1))
    if tilt_coordinates is not None:
        coords = np.asarray(tilt_coordinates, dtype=float)
        if coords.shape != (len(opd), 2) or not np.isfinite(coords).all():
            raise ValueError("Tilt coordinates must have shape (N, 2)")
        basis = np.column_stack((basis, coords))
    root_weight = np.sqrt(weights)
    fit, _, rank, _ = np.linalg.lstsq(basis * root_weight[:, None], opd * root_weight, rcond=None)
    if rank != basis.shape[1]:
        raise ValueError("Pupil samples do not identify every requested reference term")
    residual_waves = (opd - basis @ fit) / (wavelength_um * 1e-3)
    return float(np.sqrt(np.sum(weights * residual_waves ** 2))), residual_waves
