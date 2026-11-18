"""Thickness between two coaxial optical faces on the complete finished radius."""
import math

import numpy as np


def _sphere(surface):
    return surface.conic == 0 and not any(surface.even_coefficients)


def _slope_sign(surface, radius):
    """Enclose dz/d(r²) over the aperture, including the effective A2 term."""
    curvature = surface.curvature_mm_inv
    if _sphere(surface):
        return int(np.sign(curvature))
    if radius is None:
        return None
    domain = 1 - (1 + surface.conic)*curvature**2*radius**2
    if domain <= 0:
        return None
    ends = [curvature/2, curvature/(2*math.sqrt(domain))]
    lo, hi = min(ends), max(ends)
    for j, coefficient in enumerate(surface.even_coefficients):
        term = (j+1)*coefficient*radius**(2*j)
        lo += term if j == 0 else min(0., term)
        hi += term if j == 0 else max(0., term)
    if lo == hi == 0:
        return 0
    if lo >= 0:
        return 1
    if hi <= 0:
        return -1
    return None


def shape(front, rear, radius):
    signs = [_slope_sign(s, radius) for s in (front, rear)]
    if 0 in signs:
        return 'plano'
    if None in signs:
        return 'possible_aspheric_meniscus'
    return 'meniscus' if signs[0] == signs[1] else 'opposed'


def _point(s, r):
    return np.array([r, s.vertex_mm + float(s.sag_and_gradient(r*r)[0])])


def _project(p, surface, radius):
    candidates = [_point(surface, 0.), _point(surface, radius)]
    c = surface.curvature_mm_inv
    axial = 1-c*(p[1]-surface.vertex_mm)
    if axial > 0:
        rho = p[0]/math.hypot(c*p[0], axial)
        if 0 < rho < radius:
            candidates.append(_point(surface, rho))
    return min(candidates, key=lambda q: np.linalg.norm(q-p))


def _sphere_distance(front, rear, radius):
    # For disjoint spheres an interior common normal lies on the axis or the
    # spheres are concentric. Other minimizers contain a cap boundary.
    pairs = []
    for r in (0., radius):
        p, q = _point(front, r), _point(rear, r)
        pairs.extend([(p, _project(p, rear, radius)), (_project(q, front, radius), q)])
    p, q = min(pairs, key=lambda pair: np.linalg.norm(pair[0]-pair[1]))
    return float(np.linalg.norm(p-q)), p, q


def _curvature_bound(surface, radius):
    c = surface.curvature_mm_inv
    minimum_domain = min(1., 1-(1+surface.conic)*c*c*radius*radius)
    total = abs(c)/minimum_domain**1.5
    for k, coefficient in enumerate(surface.even_coefficients, start=1):
        degree = 2*k
        total += degree*(degree-1)*abs(coefficient)*radius**(degree-2)
    return total


def _polyline_distance(front, rear, radius, count):
    rr = np.linspace(0., radius, count+1)
    zz = [s.vertex_mm+s.sag_and_gradient(rr*rr)[0] for s in (front, rear)]
    # Same radial knots. Crossing segments therefore meet in an aligned interval.
    gap = zz[1]-zz[0]
    if np.min(gap) <= 0:
        return 0., None, None
    polylines = [np.column_stack((rr, z)) for z in zz]
    best = float(np.min(gap))
    i = int(np.argmin(gap))
    witness = [polylines[0][i], polylines[1][i]]
    for side in (0, 1):
        vertices, other = polylines[side], polylines[1-side]
        seg = np.diff(other, axis=0)
        squared = np.sum(seg*seg, axis=1)
        for p in vertices:
            # Only segments whose radial span lies within the current best.
            a = max(0, int(np.searchsorted(rr, p[0]-best))-1)
            b = min(count, int(np.searchsorted(rr, p[0]+best, side='right')))
            if a >= b:
                continue
            t = np.clip(np.sum((p-other[a:b])*seg[a:b], axis=1)/squared[a:b], 0., 1.)
            near = other[a:b]+seg[a:b]*t[:,None]
            dist = np.linalg.norm(near-p, axis=1)
            k = int(np.argmin(dist))
            if dist[k] < best:
                best = float(dist[k])
                witness = [p, near[k]] if side == 0 else [near[k], p]
    return best, *witness


def thickness_interval(front, rear, radius, required, axial_minimum, axial_at):
    """Bound minimum Euclidean face distance. A limit inside the bounds is unresolved."""
    if axial_minimum <= 0:
        # The signed axial intersection check separately rejects this lens.
        return {'method':'intersecting_or_reversed', 'lower_mm':0., 'upper_mm':0.}
    if _sphere(front) and _sphere(rear):
        distance, p, q = _sphere_distance(front, rear, radius)
        return dict(method='analytic_spherical_caps', lower_mm=distance, upper_mm=distance,
                    front_rz_mm=p.tolist(), rear_rz_mm=q.tolist())
    bound = sum(_curvature_bound(s,radius) for s in (front,rear))
    # Chord interpolation error on each face <= sup|z″| * step²/8.
    # A pair of actual surface points supplies the independent upper witness.
    best_p, best_q = _point(front, axial_at), _point(rear, axial_at)
    upper = float(np.linalg.norm(best_p-best_q))
    lower = 0.
    for segments in (16,32,64,128,256,512,1024):
        distance, p, q = _polyline_distance(front,rear,radius,segments)
        error = bound*(radius/segments)**2/8
        lower = max(lower, distance-error)
        if p is not None:
            p, q = _point(front,p[0]), _point(rear,q[0])
            witness = float(np.linalg.norm(p-q))
            if witness < upper:
                upper, best_p, best_q = witness,p,q
        if lower >= required or upper < required or upper-lower <= 1.e-6:
            break
    return dict(method='bounded_asphere_chords', lower_mm=lower, upper_mm=upper,
                segments=segments, front_rz_mm=best_p.tolist(), rear_rz_mm=best_q.tolist())
