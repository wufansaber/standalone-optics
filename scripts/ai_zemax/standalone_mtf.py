"""Scalar FFT PSF and MTF for the supported sequential refractive prescriptions."""

import numpy as np

from .standalone import TraceResult, reference_opd_mm
from .standalone_analysis import _unit_axis


def _fft_pupil(pupil, padding, include_psf):
    """Return complex sagittal/tangential OTF cuts and optional centered PSF."""
    size = len(pupil) * padding
    amplitude = np.fft.fft2(pupil, s=(size, size))
    psf = np.abs(amplitude) ** 2
    psf /= psf.sum()
    # An axial OTF cut is the 1D transform of the corresponding PSF marginal.
    marginals = np.stack((psf.sum(axis=0), psf.sum(axis=1)))
    curves = np.fft.ifft(marginals, axis=1)[:, :len(pupil)].copy()
    curves /= curves[:, :1].real.copy()
    return np.fft.fftshift(psf) if include_psf else None, curves


def _working_na(system, wave, launch, axes, chief_direction):
    """Return meridional working NA and its image-plane projection."""
    samples = np.array([1.0, 0.99, 0.975, 0.95, 0.9, 0.8])
    pupil = samples[:, None, None] * np.concatenate((axes.T, -axes.T))[None, :, :]
    rays = system._shoot(wave, pupil.reshape(-1, 2), *launch)
    directions = rays.directions.reshape(-1, 4, 3)
    for radius, valid, direction in zip(samples, rays.valid.reshape(-1, 4), directions):
        if np.all(valid):
            relative_sine = np.linalg.norm(np.cross(direction, chief_direction), axis=1)
            pair_na = (relative_sine[:2] + relative_sine[2:]) / 2.0
            na = np.minimum(system.indices[wave, -1] * pair_na / radius, system.indices[wave, -1])
            if np.all(na > 0):
                # FFT MTF reports frequency in the fixed image plane, whose
                # tangential footprint is enlarged by an oblique chief ray.
                chief_projection = chief_direction[:2] @ axes
                projection = np.sqrt(1.0 - np.clip(chief_projection**2, 0.0, 1.0))
                if np.any(projection <= 0.0):
                    raise ValueError("MTF cannot project a grazing chief ray onto the image plane")
                return na, na * projection, float(radius)
    raise ValueError("MTF could not determine the working NA from marginal rays")


def _spectrum(system, wave, launch, axes, grid, support, center, padding, include_psf):
    samples = len(support)
    rays = system._shoot(wave, grid, *launch)
    chief = system._shoot(wave, np.zeros((1, 2)), *launch)
    if not chief.valid[0]:
        raise ValueError(f"MTF chief ray failed at surface {chief.failed_surface[0] + 1}: {chief.reason[0]}")
    failed = np.flatnonzero(~rays.valid & (rays.reason != "aperture"))
    if len(failed):
        first = failed[0]
        raise ValueError(f"MTF ray failed at surface {rays.failed_surface[first] + 1}: {rays.reason[first]}")
    if not rays.valid.any():
        raise ValueError("MTF pupil has no transmitted rays")
    valid = rays.valid
    transmitted = TraceResult(rays.points[valid], rays.directions[valid], rays.opl_mm[valid],
                              valid[valid], rays.failed_surface[valid], rays.reason[valid])
    image_index = system.indices[wave, -1]
    reference_distance = system.reference_distance_mm / chief.directions[0, 2]
    wavelength_mm = system.wavelengths[wave] * 1e-3
    opd = reference_opd_mm(transmitted, chief, reference_distance, image_index, center=center)
    phase_waves = opd / wavelength_mm
    amplitude = system.pupil_weights(grid[valid], None, for_spot=False)
    pupil = np.zeros((samples, samples), dtype=complex)
    pupil.flat[np.flatnonzero(support)[valid]] = amplitude * np.exp(2j * np.pi * phase_waves)
    psf, curves = _fft_pupil(pupil, padding, include_psf)
    na, projected_na, marginal_radius = _working_na(system, wave, launch, axes, chief.directions[0])
    cutoff = 2.0 * projected_na / wavelength_mm

    phase_grid = np.full_like(support, np.nan, dtype=float)
    phase_grid.flat[np.flatnonzero(support)[valid]] = phase_waves
    adjacent = np.concatenate([np.diff(phase_grid, axis=ax).ravel() for ax in range(2)])
    adjacent = np.abs(adjacent[np.isfinite(adjacent)])
    chief_direction = chief.directions[0]
    chief_angle_deg = float(np.rad2deg(np.arccos(np.clip(chief_direction[2], -1., 1.))))
    transverse_axes = np.vstack((axes, np.zeros(2)))
    transverse_axes -= chief_direction[:, None] * (chief_direction @ transverse_axes)
    transverse_axes /= np.linalg.norm(transverse_axes, axis=0)
    direction_cosines = transmitted.directions @ transverse_axes
    affine_pupil = np.column_stack((np.ones(len(transmitted.points)), grid[valid] @ axes))
    mapping = np.linalg.lstsq(affine_pupil, direction_cosines, rcond=None)[0]
    mapping_residual = (direction_cosines - affine_pupil @ mapping) / (na / image_index)
    transverse_mm = np.linalg.norm(transmitted.points[:, :2] - chief.points[0, :2], axis=1)
    metadata = {
        "wave_index": int(wave),
        "wavelength_um": float(system.wavelengths[wave]),
        "working_f_number_sagittal_tangential": (0.5 / na).tolist(),
        "cutoff_lpmm_sagittal_tangential": cutoff.tolist(),
        "marginal_na_sample_radius": marginal_radius,
        "chief_incidence_angle_deg": chief_angle_deg,
        "pupil_mapping_rms_fraction": float(np.sqrt(np.mean(mapping_residual**2))),
        "pupil_ray_count": len(grid),
        "aperture_clipped_rays": int(np.count_nonzero(~valid)),
        "negative_segment_rays": int(np.count_nonzero(rays.negative_segment_count[valid])),
        "opd_peak_to_valley_waves": float(np.ptp(phase_waves)),
        "max_transverse_ray_aberration_wavelengths": float(transverse_mm.max() / wavelength_mm),
        "maximum_neighbor_opd_step_waves": float(adjacent.max()) if len(adjacent) else None,
    }
    return psf, curves, cutoff, metadata


def _interpolate_otf(values, frequency, cutoff, samples):
    bins = np.arange(samples) * (cutoff / samples)
    # The full pupil diameter does not overlap after a shift by the cutoff.
    bins = np.r_[bins, cutoff]
    values = np.r_[values, 0j]
    return (np.interp(frequency, bins, values.real, right=0.0)
            + 1j * np.interp(frequency, bins, values.imag, right=0.0))


def _common_psf(spectra, weights, padding):
    """Resample unit-energy spectral PSFs on common image-space axes."""
    size = len(spectra[0][0])
    spacings = np.array([1.0 / (padding * entry[2]) for entry in spectra])
    spacing = spacings.min(axis=0)
    half = np.ceil((size // 2) * spacings.max(axis=0) / spacing).astype(int)
    if np.any(half > 1024):
        raise ValueError("The common PSF grid would exceed 2049 pixels per axis")
    x, y = [np.arange(-n, n + 1) * pitch for n, pitch in zip(half, spacing)]
    combined = np.zeros((len(y), len(x)))
    pixels = np.arange(size) - size // 2
    for (psf, _, _, _), weight, (dx, dy) in zip(spectra, weights, spacings):
        # Convert pixel energy to physical intensity before interpolation.
        horizontally = np.array([np.interp(x, pixels * dx, row, left=0., right=0.) for row in psf])
        plane = np.array([np.interp(y, pixels * dy, col, left=0., right=0.)
                          for col in horizontally.T]).T
        combined += weight * plane / (dx * dy)
    combined /= combined.sum()
    return {"sagittal_position_um": (1000.0 * x).tolist(),
            "tangential_position_um": (1000.0 * y).tolist(),
            "normalized_pixel_energy": combined.tolist()}


def diffraction_mtf(system, field_xy, *, samples=128, padding=2, frequency_lpmm=None,
                    wave=None, include_psf=False, axis_convention="fixed_xy"):
    """Calculate scalar diffraction curves in image-space cycles/mm.

    The Fourier grid is uniform in normalized entrance-pupil coordinates and
    its two scales use real marginal-ray working NAs. Its accuracy assumes
    modest exit pupil aberration and chief incidence angle. The default axes
    follow the OpticStudio FFT convention (Y tangential, X sagittal); use
    axis_convention="field" for radial tangential/orthogonal sagittal cuts.
    Monochromatic
    calculations re-resolve image-height fields at that wavelength. A
    polychromatic calculation shares the primary wavelength's field launch
    and image reference, then sums complex OTFs using spectral weights before
    taking the magnitude. No polarization or coating transmission is applied.
    """
    if not isinstance(samples, int) or samples < 16 or samples > 1024 or samples % 2:
        raise ValueError("Use an even MTF pupil sample count from 16 to 1024")
    if not isinstance(padding, int) or not 2 <= padding <= 4:
        raise ValueError("FFT padding must be an integer from 2 to 4")
    if samples * padding > 2048:
        raise ValueError("FFT grid must not exceed 2048 pixels per axis")
    field = np.asarray(field_xy, dtype=float)
    if field.shape != (2,) or not np.isfinite(field).all():
        raise ValueError("Field must contain two finite coordinates")
    if wave is not None and (not isinstance(wave, (int, np.integer))
                             or not 0 <= wave < len(system.wavelengths)):
        raise ValueError("Wavelength index is outside the model")
    if wave is None:
        weights = np.asarray(system.model.get("wavelength_weights", np.ones(len(system.wavelengths))),
                             dtype=float)
        if (weights.shape != system.wavelengths.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError("MTF wavelength weights must be nonnegative and nonzero in total")
        waves = np.flatnonzero(weights > 0)
        weights = weights[waves] / weights.max()
        weights /= weights.sum()
        reference_wave = system.primary
    else:
        waves, weights, reference_wave = [wave], np.ones(1), wave
    launch = system.launch_field(field, reference_wave)
    chief = system._shoot(reference_wave, np.zeros((1, 2)), *launch)
    if not chief.valid[0]:
        raise ValueError(f"MTF reference chief failed at surface {chief.failed_surface[0] + 1}: {chief.reason[0]}")

    if axis_convention not in ("fixed_xy", "field"):
        raise ValueError("MTF axis convention must be fixed_xy or field")
    tangential = (np.array([0., 1.]) if axis_convention == "fixed_xy" or not np.any(field)
                  else _unit_axis(field))
    sagittal = np.array([tangential[1], -tangential[0]])
    axes = np.column_stack((sagittal, tangential))
    axis = (2.0 * (np.arange(samples) + 0.5) / samples) - 1.0
    x, y = np.meshgrid(axis, axis)
    support = x*x + y*y <= 1.0
    grid = np.column_stack((x[support], y[support])) @ axes.T
    if frequency_lpmm is not None:
        frequency = np.asarray(frequency_lpmm, dtype=float)
        if (frequency.ndim != 1 or not len(frequency) or not np.isfinite(frequency).all()
                or np.any(frequency < 0)):
            raise ValueError("Supply a nonempty vector of finite nonnegative image frequencies")
    spectra = [_spectrum(system, w, launch, axes, grid, support, chief.points[0], padding, include_psf)
               for w in waves]
    if frequency_lpmm is None:
        frequency = np.linspace(0., max(entry[2].max() for entry in spectra), 101)
    curves = {}
    for axis_index, label in enumerate(("sagittal", "tangential")):
        spectral = [_interpolate_otf(entry[1][axis_index], frequency, entry[2][axis_index], samples)
                    for entry in spectra]
        curves[label] = np.clip(np.abs(weights @ np.asarray(spectral)), 0.0, 1.0).tolist()
    metadata = [entry[3] for entry in spectra]
    warnings = []
    if any(row["chief_incidence_angle_deg"] > 20.0 for row in metadata):
        warnings.append("chief_incidence_exceeds_20_deg")
    if any(min(row["working_f_number_sagittal_tangential"]) < 1.5 for row in metadata):
        warnings.append("fast_image_cone_scalar_model")
    if any(row["marginal_na_sample_radius"] < 1.0 for row in metadata):
        warnings.append("working_na_extrapolated_from_subaperture")
    if any(row["max_transverse_ray_aberration_wavelengths"] > 300.0 for row in metadata):
        warnings.append("large_transverse_ray_aberration_scalar_fft")
    if any(row["opd_peak_to_valley_waves"] > 10.0 for row in metadata):
        warnings.append("large_opd_span_scalar_fft")
    if any((row["maximum_neighbor_opd_step_waves"] or 0.0) >= 0.5 for row in metadata):
        warnings.append("phase_grid_may_alias_increase_sampling")
    if any(row["pupil_mapping_rms_fraction"] > 0.05 for row in metadata):
        warnings.append("exit_pupil_mapping_depends_non_linearly_on_pupil_position")
    if any(row["negative_segment_rays"] for row in metadata):
        warnings.append("signed_comparison_only")
    result = {"method": "scalar_fft_normalized_pupil", "field_xy": field.tolist(),
              "axis_convention": axis_convention,
              "pupil_samples_per_axis": samples, "fft_padding": padding,
              "frequency_lpmm": frequency.tolist(), **curves,
              "wavelength_weights": weights.tolist(), "spectra": metadata,
              "warnings": warnings}
    if include_psf:
        result["psf"] = _common_psf(spectra, weights, padding)
    return result
