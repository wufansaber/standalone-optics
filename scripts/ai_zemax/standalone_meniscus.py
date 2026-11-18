"""Thickness between two coaxial optical faces on the complete finished radius."""
import math
from functools import lru_cache

import numpy as np


def _sphere(surface):
    return surface.conic == 0 and not any(surface.even_coefficients)


@lru_cache(maxsize=512)
def _radial_polynomial_bounds(coefficients):
    """Enclose P(x) on 32 closed cells of x=(r/R)^2, with signed coefficients."""
    degree = len(coefficients)-1
    bernstein = np.array([math.fsum(coefficients[j]*math.comb(i,j)/math.comb(degree,j)
                                    for j in range(i+1)) for i in range(degree+1)])[None,:]
    # The convex hull of each Bernstein piece contains its curve. Subdivision
    # uses only convex combinations, so the initial absolute error is not amplified.
    for _ in range(5):
        levels=[bernstein]
        for i in range(degree):
            levels.append(.5*(levels[-1][:,1:]+levels[-1][:,:-1]))
        left=np.stack([a[:,0] for a in levels],axis=1)
        right=np.stack([a[:,-1] for a in reversed(levels)],axis=1)
        bernstein=np.stack((left,right),axis=1).reshape(-1,degree+1)
    padding=128*np.finfo(float).eps*(degree+1)**2*sum(abs(a) for a in coefficients)
    return bernstein.min(1)-padding,bernstein.max(1)+padding


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
    # Opposing coefficients can give an inconclusive single interval even
    # when every radius has the same sign. Accept a sign only if all closed
    # subintervals certify it, including a floating-point margin.
    radial_square = np.linspace(0.,radius**2,33)
    base = curvature/(2*np.sqrt(1.-(1.+surface.conic)*curvature**2*radial_square))
    coefficients=[(j+1)*v*radius**(2*j) for j,v in enumerate(surface.even_coefficients)]
    # Pure conics returned at the single-interval test above.
    lower,upper=_radial_polynomial_bounds(tuple(coefficients) or (0.,))
    lower=lower+np.minimum(base[:-1],base[1:])
    upper=upper+np.maximum(base[:-1],base[1:])
    error=64*np.finfo(float).eps*max(abs(base[0]),abs(base[-1]))/min(1.,domain)
    if np.min(lower) > error:
        return 1
    if np.max(upper) < -error:
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
    base = abs(c)/minimum_domain**1.5
    if not any(surface.even_coefficients):
        return base
    radial_square = np.linspace(0.,radius**2,33)
    conic = c/(1.-(1.+surface.conic)*c*c*radial_square)**1.5
    terms=[(2*k)*(2*k-1)*v*radius**(2*k-2) for k,v in enumerate(surface.even_coefficients,1)]
    lo,hi=_radial_polynomial_bounds(tuple(terms))
    lo=lo+np.minimum(conic[:-1],conic[1:])
    hi=hi+np.maximum(conic[:-1],conic[1:])
    padding=128*np.finfo(float).eps*base/minimum_domain
    return float(max(np.max(hi),-np.min(lo))+padding)


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


def thickness_interval(front, rear, radius, required, axial_minimum, axial_at, *, refine=False):
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
    grids = (16,32,64,128,256,512,1024)
    if refine:
        # The CT solver asks for a ten-nanometre bracket. Strong outer
        # aspheres can still have a larger chord error at 4096 segments.
        grids += (2048,4096,8192,16384)
    for segments in grids:
        distance, p, q = _polyline_distance(front,rear,radius,segments)
        error = bound*(radius/segments)**2/8
        lower = max(lower, distance-error)
        if p is not None:
            p, q = _point(front,p[0]), _point(rear,q[0])
            witness = float(np.linalg.norm(p-q))
            if witness < upper:
                upper, best_p, best_q = witness,p,q
        if lower >= required or upper < required or (not refine and upper-lower <= 1.e-6):
            break
    return dict(method='bounded_asphere_chords', lower_mm=lower, upper_mm=upper,
                segments=segments, front_rz_mm=best_p.tolist(), rear_rz_mm=best_q.tolist())

def center_thickness_bracket(front, rear, radius, row, rules):
    """Bracket the least CT of this isolated piece with both shapes and OD held fixed."""
    from dataclasses import replace
    ct = row['center_thickness_mm']
    limits = row['effective_limits']
    ct_loss = limits['center_thickness_budget_mm']
    # Translating the rear cap shifts every axial thickness by the same amount.
    lo = max(limits['minimum_center_thickness_mm']+ct_loss,
             2*radius*rules['minimum_center_to_diameter_ratio']+ct_loss,
             limits['minimum_radial_thickness_mm']+limits['radial_thickness_budget_mm']
             + ct-row['minimum_radial_thickness_mm'],
             rules['minimum_edge_thickness_mm']+rules['edge_thickness_budget_mm']
             + ct-row['edge_thickness_mm'])
    hi = rules['maximum_center_thickness_mm']-ct_loss
    required = row['face_distance']['limit_mm']+rules['face_distance_budget_mm']
    def evaluate(value, *, refine=False):
        moved = replace(rear, vertex_mm=front.vertex_mm+value)
        interval = thickness_interval(front,moved,radius,required,
                                      row['minimum_radial_thickness_mm']+value-ct,row['minimum_at_radius_mm'],refine=refine)
        if interval['lower_mm'] >= required:
            return 'feasible'
        if interval['upper_mm'] < required:
            return 'infeasible'
        return 'unresolved'
    if lo > hi:
        return dict(status='infeasible',axial_floor_mm=lo,maximum_ct_mm=hi)
    end = evaluate(hi)
    if end != 'feasible':
        return dict(status=end,maximum_ct_mm=hi)
    if evaluate(lo) == 'feasible':
        return dict(status='converged',lower_mm=lo,feasible_upper_mm=lo,width_mm=0.)
    # For separated single-valued graphs the distance is monotone under
    # outward axial translation (nested epigraphs). Retain a verified upper end.
    while hi-lo > 1.e-5:
        mid=(lo+hi)/2
        if mid in (lo,hi):
            break
        result=evaluate(mid)
        if result=='unresolved':
            # The midpoint can coincide with the exact boundary. Probe on
            # both sides, and retain only a certified change of endpoint.
            old_lo,old_hi=lo,hi
            for probe in ((lo+mid)/2,(mid+hi)/2):
                side=evaluate(probe,refine=True)
                if side=='infeasible':
                    lo=max(lo,probe)
                elif side=='feasible':
                    hi=min(hi,probe)
            if (lo,hi)==(old_lo,old_hi):
                break
            continue
        if result=='feasible':
            hi=mid
        else:
            lo=mid
    return dict(status='converged' if hi-lo <= 1.e-5 else 'unresolved',
                lower_mm=lo,feasible_upper_mm=hi,width_mm=hi-lo)
