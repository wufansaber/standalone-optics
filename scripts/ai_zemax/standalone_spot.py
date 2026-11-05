"""Fixed-image spot diagrams with one spatial reference for all wavelengths."""

import numpy as np

from .standalone_analysis import spot_statistics
from .standalone_system import rectangular_pupil


def _spot_wave(system, wave, pupil, weights, launch, origin):
    rays = system._shoot(wave, pupil, *launch)
    valid = rays.valid
    clipped = ~valid & (rays.reason == "aperture")
    failed = np.flatnonzero(~valid & ~clipped)
    signed_rays = int(np.count_nonzero(rays.negative_segment_count[valid]))
    row = {
        "wave_index": wave + 1,
        "wavelength_um": float(system.wavelengths[wave]),
        "valid_rays": int(np.count_nonzero(valid)),
        "aperture_clipped_rays": int(np.count_nonzero(clipped)),
        "failed_rays": len(failed),
        "rays_with_negative_segments": signed_rays,
    }
    if len(failed):
        first = failed[0]
        surface = int(rays.failed_surface[first]) + 1
        reason = str(rays.reason[first])
        return dict(row, status="failed", error=f"{len(failed)} rays failed. "
                    f"First failure at surface {surface}: {reason}",
                    first_failure={"surface": surface, "reason": reason})
    if not np.any(valid):
        return dict(row, status="failed", error="No rays reached the image after aperture clipping")

    points_um = 1000. * (rays.points[valid, :2] - origin)
    stats = spot_statistics(points_um, weights[valid])
    valid_ids = np.flatnonzero(valid)
    shortest = valid_ids[np.argmin(rays.shortest_segment_mm[valid])]
    return dict(
        row, status="signed_comparison_only" if signed_rays else "calculated",
        intercepts_xy_um=points_um.tolist(), centroid_xy_um=stats["centroid_xy"].tolist(),
        rms_radius_um=stats["rms_radius"], rms_x_um=stats["rms_x"], rms_y_um=stats["rms_y"],
        geometric_radius_um=stats["geometric_radius"],
        shortest_signed_segment_mm=float(rays.shortest_segment_mm[shortest]),
        shortest_segment_target_surface=int(rays.shortest_segment_surface[shortest]) + 1,
    )


def spot_diagram(system, field, *, samples=32):
    """Trace a RECT pupil grid at one fixed image plane for each wavelength.

    The primary wavelength determines the object-side launch for every color.
    Spot coordinates share its chief intercept so lateral color is visible.
    Each RMS radius is intensity-weighted about that wavelength's centroid.
    Aperture clipping is counted. A non-aperture trace failure invalidates that
    wavelength rather than presenting its surviving rays as a complete spot.
    """
    pupil, weights = rectangular_pupil(samples)
    field = np.asarray(field, dtype=float)
    launch = system.launch_field(field, system.primary)
    chief = system._shoot(system.primary, np.zeros((1, 2)), *launch)
    if not chief.valid[0]:
        raise ValueError(f"Spot reference chief failed at surface {chief.failed_surface[0] + 1}: "
                         f"{chief.reason[0]}")
    origin = chief.points[0, :2]
    weights = system.pupil_weights(pupil, weights, for_spot=True)
    waves = [_spot_wave(system, wave, pupil, weights, launch, origin)
             for wave in range(len(system.wavelengths))]
    signed_chief = bool(chief.negative_segment_count[0])
    statuses = {row["status"] for row in waves}
    if "failed" in statuses:
        status = "incomplete"
    elif signed_chief or "signed_comparison_only" in statuses:
        status = "signed_comparison_only"
    else:
        status = "calculated"
    return {
        "status": status,
        "field_xy": field.tolist(),
        "image_z_mm": float(system.image_z_mm),
        "reference_wave_index": system.primary + 1,
        "reference_wavelength_um": float(system.wavelengths[system.primary]),
        "reference_chief_xy_mm": origin.tolist(),
        "reference_chief_has_negative_segment": signed_chief,
        "sampling": f"rect{samples}",
        "pupil_grid_size_per_axis": 2 * samples,
        "pupil_ray_count": len(pupil),
        "waves": waves,
    }
