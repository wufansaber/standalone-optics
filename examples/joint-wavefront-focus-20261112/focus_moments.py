"""Finite-aperture geometric focus from every ray, without paraxial angles.

This numerical example contains no lens, material catalog or private inputs.
Positions use millimetres. Directions are actual image-space direction cosines.
"""
import json
import numpy as np


def geometric_focus(points_xy, directions_lmn, weights, *, error=None, vignette=None):
    points = np.asarray(points_xy, dtype=float)
    directions = np.asarray(directions_lmn, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    if points.shape != (n, 2) or directions.shape != (n, 3) or weights.shape != (n,):
        raise ValueError("Every ray needs position, direction and intensity weight")
    for flags in (error, vignette):
        if flags is not None and (np.shape(flags) != (n,) or np.any(flags)):
            raise ValueError("Failed or vignetted ray: reject the whole bundle")
    if (not all(np.isfinite(x).all() for x in (points, directions, weights))
            or np.any(weights < 0) or sum(weights) <= 0
            or np.any(directions[:, 2] <= 0)):
        raise ValueError("Nonfinite, backwards or invalid ray population")
    w = weights / sum(weights)
    slopes = directions[:, :2] / directions[:, 2, None]
    x = points - w @ points
    t = slopes - w @ slopes
    cov = np.sum(w[:, None] * x * t, axis=0)
    var = np.sum(w[:, None] * t * t, axis=0)
    if np.any(var <= 0):
        raise ValueError("Both slope variances must be positive")
    axis = -cov / var
    return dict(S_image_mm=float(axis[0]), T_image_mm=float(axis[1]),
                mean_TS_image_mm=float(np.mean(axis)),
                full2D_image_mm=float(-sum(cov) / sum(var)))


def _golden_minimum(fun, low, high):
    """A derivative-free numerical oracle for the synthetic demonstration."""
    ratio = (np.sqrt(5.) - 1.) / 2.
    c, d = high - ratio * (high-low), low + ratio * (high-low)
    fc, fd = fun(c), fun(d)
    for _ in range(100):
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - ratio * (high-low); fc = fun(c)
        else:
            low, c, fc = c, d, fd
            d = low + ratio * (high-low); fd = fun(d)
    return float((low + high) / 2.)


def demonstration():
    rng = np.random.default_rng(7)
    pupil = rng.uniform(-1., 1., size=(8000, 2))
    pupil = pupil[np.sum(pupil*pupil, axis=1) <= 1.]
    slopes = pupil * [.16, .25]
    directions = np.c_[slopes, np.ones(len(slopes))]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = -slopes * [.05, .11] + .002 * pupil**3
    weights = np.exp(-.4 * np.sum(pupil*pupil, axis=1))
    weights /= sum(weights)
    result = geometric_focus(points, directions, weights)

    def spot_variance(distance):
        xy = points + distance * directions[:, :2] / directions[:, 2, None]
        centred = xy - weights @ xy
        return float(np.sum(weights * np.sum(centred**2, axis=1)))

    reference = _golden_minimum(spot_variance, -.2, .2)
    translation = geometric_focus(points + [1000., -700.], directions, weights)
    assert abs(reference - result['full2D_image_mm']) < 1e-8
    assert abs(translation['full2D_image_mm'] - result['full2D_image_mm']) < 1e-9
    assert spot_variance(result['full2D_image_mm']) <= spot_variance(reference) + 1e-16
    rejected = 0
    for kwargs in ({'error': np.ones(len(points))}, {'vignette': np.ones(len(points))}):
        try:
            geometric_focus(points, directions, weights, **kwargs)
        except ValueError:
            rejected += 1
    assert rejected == 2
    return dict(synthetic_only=True, lens_or_catalog_data=False, rays=len(points),
                covariance_result=result, direct_spot_minimum_mm=reference,
                direct_minimization_error_mm=abs(reference-result['full2D_image_mm']),
                translation_error_mm=abs(translation['full2D_image_mm']-result['full2D_image_mm']),
                invalid_populations_rejected=rejected,
                private_design_acceptance_not_tested_by_this_demo=True)


if __name__ == '__main__':
    print(json.dumps(demonstration(), indent=2, allow_nan=False))
