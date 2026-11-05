"""Image-space MTF computed from the irradiance of traced geometric rays."""

import numpy as np

from .standalone_analysis import _unit_axis
from .standalone_mtf import _working_na


def _ray_otf(coordinates, weights, frequency):
    """Fourier transform the discrete ray energy at the requested frequencies."""
    output = np.empty(len(frequency), dtype=complex)
    block_size = max(1, 2**20 // len(coordinates))
    for start in range(0, len(frequency), block_size):
        end = start + block_size
        phase = (2.0 * np.pi * frequency[start:end, None]) * coordinates
        real = np.einsum("ij,j->i", np.cos(phase), weights)
        imaginary = np.einsum("ij,j->i", np.sin(phase), weights)
        output[start:end] = real + 1j * imaginary
    return output


def _diffraction_limit(frequency, cutoff):
    ratio = np.clip(frequency / cutoff, 0., 1.)
    return (2. / np.pi) * (np.arccos(ratio) - ratio * np.sqrt(1. - ratio**2))


def _spectrum(system, wave, launch, axes, grid, support, center):
    rays = system._shoot(wave, grid, *launch)
    chief = system._shoot(wave, np.zeros((1, 2)), *launch)
    if not chief.valid[0]:
        raise ValueError(f"Geometric MTF chief ray failed at surface {chief.failed_surface[0] + 1}: {chief.reason[0]}")
    valid = rays.valid
    failed = np.flatnonzero(~valid & (rays.reason != "aperture"))
    if len(failed):
        first = failed[0]
        raise ValueError(f"Geometric MTF ray failed at surface {rays.failed_surface[first] + 1}: {rays.reason[first]}")
    weights = system.pupil_weights(grid, None, for_spot=True)
    all_energy = weights.sum()
    weights = weights[valid]
    transmitted_energy = weights.sum()
    if not np.isfinite(all_energy) or not np.isfinite(transmitted_energy) or transmitted_energy <= 0.:
        raise ValueError("Geometric MTF pupil has no finite positive transmitted ray weight")
    weights /= transmitted_energy
    points = (rays.points[valid, :2] - center) @ axes
    centroid = weights @ points
    sigma = np.sqrt(weights @ (points - centroid)**2)

    # Adjacent pupil samples bound the highest reliably sampled phase frequency
    # in the Fourier sum over image intercepts.
    positions = np.full((*support.shape, 2), np.nan)
    positions.reshape(-1, 2)[np.flatnonzero(support)[valid]] = points
    largest_step = np.zeros(2)
    for axis in (0, 1):
        delta = np.abs(np.diff(positions, axis=axis))
        largest_step = np.maximum(largest_step, np.max(delta, axis=(0, 1), where=np.isfinite(delta), initial=0.))
    safe_frequency = np.divide(.5, largest_step, out=np.full(2, np.inf), where=largest_step > 0.)

    raw_na, image_na, marginal_radius = _working_na(system, wave, launch, axes, chief.directions[0])
    cutoff = 2. * image_na / (system.wavelengths[wave] * 1.e-3)
    spectrum = {
        "wave_index": int(wave),
        "wavelength_um": float(system.wavelengths[wave]),
        "pupil_ray_count": len(grid),
        "aperture_clipped_rays": int(np.count_nonzero(~valid)),
        "negative_segment_rays": int(np.count_nonzero(rays.negative_segment_count[valid])),
        "transmitted_energy_fraction": float(transmitted_energy / all_energy),
        "working_f_number_sagittal_tangential": (0.5 / raw_na).tolist(),
        "marginal_na_sample_radius": marginal_radius,
        "diffraction_cutoff_lpmm_sagittal_tangential": cutoff.tolist(),
        "centroid_from_reference_chief_mm_sagittal_tangential": centroid.tolist(),
        "rms_intercept_mm_sagittal_tangential": sigma.tolist(),
        "ray_grid_nyquist_lpmm_sagittal_tangential": [
            float(value) if np.isfinite(value) else None for value in safe_frequency],
    }
    return points, weights, cutoff, spectrum, safe_frequency, sigma


def geometric_mtf(system, field_xy, *, samples=128, frequency_lpmm=None, wave=None,
                  multiply_by_diffraction_limit=False, axis_convention="fixed_xy", include_otf=False):
    """Return T/S MTF from a uniform pupil grid and requested image frequencies.

    Each complex OTF is the Fourier transform of the weighted image intercepts.
    Wavelengths share the primary launch and a common image origin in multi-
    wavelength mode, so lateral color affects the combined modulation. Optional
    Airy scaling applies each wavelength's circular diffraction-limit OTF before
    the sum. The geometric model does not trace diffraction, scatter, polarization
    or coating transmission. Default frequency sampling resolves both the
    monochromatic blur and the polychromatic centroid spread. Check ray grid
    sampling by repeating at a denser ray grid.
    """
    if not isinstance(samples, int) or samples < 16 or samples > 1024 or samples % 2:
        raise ValueError("Use an even geometric MTF pupil sample count from 16 to 1024")
    field = np.asarray(field_xy, dtype=float)
    if field.shape != (2,) or not np.isfinite(field).all():
        raise ValueError("Field must contain two finite coordinates")
    if wave is None:
        weights = np.asarray(system.model.get("wavelength_weights", np.ones(len(system.wavelengths))), dtype=float)
        if (weights.shape != system.wavelengths.shape or not np.isfinite(weights).all()
                or np.any(weights < 0.) or not np.any(weights > 0.)):
            raise ValueError("MTF wavelength weights must be nonnegative and nonzero in total")
        waves = np.flatnonzero(weights > 0.)
        spectral_weights = weights[waves] / weights.max()
        reference_wave = system.primary
    else:
        if not isinstance(wave, (int, np.integer)) or not 0 <= wave < len(system.wavelengths):
            raise ValueError("Wavelength index is outside the model")
        waves, spectral_weights, reference_wave = [wave], np.ones(1), wave
    if axis_convention not in ("fixed_xy", "field"):
        raise ValueError("MTF axis convention must be fixed_xy or field")
    tangential = (np.array([0., 1.]) if axis_convention == "fixed_xy" or not np.any(field)
                  else _unit_axis(field))
    sagittal = np.array([tangential[1], -tangential[0]])
    axes = np.column_stack((sagittal, tangential))
    if frequency_lpmm is not None:
        frequency = np.asarray(frequency_lpmm, dtype=float)
        if (frequency.ndim != 1 or not len(frequency) or not np.isfinite(frequency).all()
                or np.any(frequency < 0.)):
            raise ValueError("Supply a nonempty vector of finite nonnegative image frequencies")
    launch = system.launch_field(field, reference_wave)
    chief = system._shoot(reference_wave, np.zeros((1, 2)), *launch)
    if not chief.valid[0]:
        raise ValueError(f"Geometric MTF reference chief failed at surface {chief.failed_surface[0] + 1}: {chief.reason[0]}")
    x = (2. * (np.arange(samples) + .5) / samples) - 1.
    px, py = np.meshgrid(x, x)
    support = px*px + py*py <= 1.
    grid = np.column_stack((px[support], py[support])) @ axes.T
    spectra = [_spectrum(system, w, launch, axes, grid, support, chief.points[0, :2]) for w in waves]
    metadata = [row[3] for row in spectra]
    spectral_weights *= [row["transmitted_energy_fraction"] for row in metadata]
    spectral_weights /= spectral_weights.sum()
    nyquist = np.min([row[4] for row in spectra], axis=0)
    if frequency_lpmm is None:
        centroids = np.asarray([row["centroid_from_reference_chief_mm_sagittal_tangential"] for row in metadata])
        centroids -= spectral_weights @ centroids
        spectral_sigma = np.asarray([row[5] for row in spectra])
        blur = np.sqrt(spectral_weights @ (spectral_sigma**2 + centroids**2))
        blur_limit = np.divide(2., blur, out=np.full(2, np.inf), where=blur > 0.)
        # Keep frequencies useful for visual judgment of highly blurred images.
        cutoff = np.max([row[2] for row in spectra], axis=0)
        band = np.minimum(np.minimum(cutoff, blur_limit), nyquist.min())
        if band.max() > 2. * band.min():
            frequency = np.union1d(np.linspace(0., band.min(), 201), np.linspace(0., band.max(), 101))
        else:
            frequency = np.linspace(0., band.max(), 101)
    combined = np.zeros((len(frequency), 2), dtype=complex)
    for weight, (points, weights, cutoff, _, _, _) in zip(spectral_weights, spectra):
        for axis in range(2):
            otf = _ray_otf(points[:, axis], weights, frequency)
            if multiply_by_diffraction_limit:
                otf *= _diffraction_limit(frequency, cutoff[axis])
            combined[:, axis] += weight * otf
    modulation = np.clip(np.abs(combined), 0., 1.)
    modulation[frequency == 0.] = 1.
    warnings = []
    if np.any(frequency.max() > nyquist):
        warnings.append("geometric_mtf_ray_grid_may_alias_increase_sampling")
    if any(row["negative_segment_rays"] for row in metadata):
        warnings.append("signed_comparison_only")
    result = {
        "method": "geometric_ray_intercept",
        "multiply_by_diffraction_limit": bool(multiply_by_diffraction_limit),
        "field_xy": field.tolist(),
        "axis_convention": axis_convention,
        "pupil_samples_per_axis": samples,
        "frequency_lpmm": frequency.tolist(),
        "sagittal": modulation[:, 0].tolist(),
        "tangential": modulation[:, 1].tolist(),
        "wavelength_weights": spectral_weights.tolist(),
        "spectra": metadata,
        "warnings": warnings,
    }

    if include_otf:
        result["otf_real_sagittal_tangential"] = combined.real.tolist()
        result["otf_imag_sagittal_tangential"] = combined.imag.tolist()
    return result