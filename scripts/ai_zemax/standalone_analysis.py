"""Geometric image-plane analyses using only traced intersections and directions."""

import numpy as np


def _intercepts(intercepts_xy):
    points = np.asarray(intercepts_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("Image intersections must be a nonempty N by 2 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("Image intersections must be finite")
    return points


def _weights(weights, count):
    if weights is None:
        return np.full(count, 1.0 / count)
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Weights must be finite, nonnegative and match the ray count")
    maximum = values.max()
    if maximum <= 0:
        raise ValueError("At least one ray must have positive weight")
    scaled = values / maximum
    return scaled / scaled.sum()


def _unit_axis(axis):
    vector = np.asarray(axis, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("Image-plane axis must be a finite 2-vector")
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError("Image-plane axis must be nonzero")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def spot_statistics(intercepts_xy, weights=None):
    """Return centroid, radial and per-axis RMS, and maximum radius about the centroid.

    Lengths use the same units as the image intersections. Pass valid rays only.
    Weights describe the pupil quadrature and default to equal sample weights.
    Zero-weight rays are excluded from the maximum radius.
    """
    points = _intercepts(intercepts_xy)
    normalized = _weights(weights, len(points))
    anchor = points[np.argmax(normalized)]
    offsets = points - anchor
    offset_centroid = normalized @ offsets
    centered = offsets - offset_centroid
    variance = normalized @ (centered * centered)
    support = normalized > 0
    return {
        "centroid_xy": anchor + offset_centroid,
        "rms_radius": float(np.sqrt(variance.sum())),
        "rms_x": float(np.sqrt(variance[0])),
        "rms_y": float(np.sqrt(variance[1])),
        "geometric_radius": float(np.linalg.norm(centered[support], axis=1).max()),
    }


def ray_fan(pupil_coordinate, intercepts_xy, chief_xy, transverse_axis):
    """Project chief-referenced transverse ray error along an image-plane axis.

    The result has one entry per normalized pupil coordinate in the input order.
    The caller selects a meridional or sagittal pupil cut when tracing the rays.
    """
    points = _intercepts(intercepts_xy)
    pupil = np.asarray(pupil_coordinate, dtype=float)
    if pupil.shape != (len(points),) or not np.all(np.isfinite(pupil)):
        raise ValueError("Pupil coordinates must be finite and match the ray count")
    chief = np.asarray(chief_xy, dtype=float)
    if chief.shape != (2,) or not np.all(np.isfinite(chief)):
        raise ValueError("Chief-ray image intersection must be a finite 2-vector")
    return (points - chief) @ _unit_axis(transverse_axis)


def best_focus(intercepts_xy, directions_xyz, weights=None, *, tangential_axis=(0.0, 1.0)):
    """Find finite-aperture geometric RMS focus along parallel image planes.

    Shifts are signed displacements along +z from the supplied image plane.
    The tangential axis is given in image x/y coordinates. The sagittal axis
    is perpendicular to it. Astigmatic separation is tangential minus sagittal.
    Each shift minimizes the corresponding centroid-referenced spot variance.
    A shift is None when the corresponding variance does not vary with focus.
    The rays must propagate in a homogeneous image space over the focus range.
    """
    points = _intercepts(intercepts_xy)
    directions = np.asarray(directions_xyz, dtype=float)
    if directions.shape != (len(points), 3) or not np.all(np.isfinite(directions)):
        raise ValueError("Ray directions must be a finite N by 3 array")
    normalized = _weights(weights, len(points))
    support = normalized > 0
    directions = directions[support]
    points = points[support]
    normalized = normalized[support]
    if np.any(directions[:, 2] == 0):
        raise ValueError("Weighted rays must cross planes normal to z")
    slopes = directions[:, :2] / directions[:, 2, None]
    if not np.all(np.isfinite(slopes)):
        raise ValueError("Weighted rays must have finite image-space slopes")

    tangential = _unit_axis(tangential_axis)
    sagittal = np.array([tangential[1], -tangential[0]])
    basis = np.column_stack((tangential, sagittal))
    # Work relative to one ray so identical slopes remain exactly constant.
    positions = (points - points[0]) @ basis
    projected_slopes = (slopes - slopes[0]) @ basis
    centered_positions = positions - normalized @ positions
    centered_slopes = projected_slopes - normalized @ projected_slopes
    covariance = normalized @ (centered_positions * centered_slopes)
    slope_variance = normalized @ (centered_slopes * centered_slopes)

    total_slope_variance = slope_variance.sum()
    # Rotation can leave roundoff in a direction with no true slope variation.
    variance_floor = total_slope_variance * (8 * np.finfo(float).eps) ** 2
    tangential_shift = float(-covariance[0] / slope_variance[0]) if slope_variance[0] > variance_floor else None
    sagittal_shift = float(-covariance[1] / slope_variance[1]) if slope_variance[1] > variance_floor else None
    radial_shift = float(-covariance.sum() / total_slope_variance) if total_slope_variance > 0 else None
    both_defined = tangential_shift is not None and sagittal_shift is not None
    return {
        "tangential_shift": tangential_shift,
        "sagittal_shift": sagittal_shift,
        "mean_shift": (tangential_shift + sagittal_shift) / 2 if both_defined else None,
        "astigmatic_separation": tangential_shift - sagittal_shift if both_defined else None,
        "radial_rms_shift": radial_shift,
    }
