"""Analytic reference checks for the standalone diffraction implementation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ai_zemax.standalone import TraceResult
from ai_zemax.standalone_mtf import diffraction_mtf


class IdealOptics:
    """Uniform circular pupils with identical wavelength and shifted image centers."""

    primary = 0
    reference_distance_mm = 100.0
    model = {"wavelength_weights": [1.0, 1.0]}
    wavelengths = np.array([0.55, 0.55])
    indices = np.ones((2, 1))

    def launch_field(self, field, wave):
        return field, None

    def _shoot(self, wave, pupil, *_):
        count = len(pupil)
        xy = -0.2 * pupil
        return TraceResult(
            np.tile([wave * 0.003, 0.0, 100.0], (count, 1)),
            np.c_[xy, np.sqrt(1 - np.sum(xy**2, axis=1))],
            np.full(count, 100.0),
            np.ones(count, dtype=bool),
            np.zeros(count, dtype=int),
            np.full(count, ""),
            negative_segment_count=np.zeros(count, dtype=int),
        )

    def pupil_weights(self, pupil, weights, *, for_spot):
        return np.ones(len(pupil))


class DiffractionTests(unittest.TestCase):
    def setUp(self):
        self.optics = IdealOptics()
        fraction = np.linspace(0.0, 1.0, 101)
        self.frequency = fraction * (400.0 / 0.55)
        self.circular_mtf = 2 / np.pi * (np.arccos(fraction) - fraction * np.sqrt(1 - fraction**2))

    def test_uniform_circular_pupil_matches_analytic_mtf(self):
        result = diffraction_mtf(self.optics, [0, 0], wave=0, frequency_lpmm=self.frequency)
        for axis in ("tangential", "sagittal"):
            np.testing.assert_allclose(result[axis], self.circular_mtf, atol=0.001)
        self.assertEqual(result["warnings"], [])

    def test_polychromatic_mtf_preserves_lateral_displacement(self):
        result = diffraction_mtf(self.optics, [0, 0], frequency_lpmm=self.frequency)
        separation_mm = 0.003
        expected = self.circular_mtf * np.abs(np.cos(np.pi * self.frequency * separation_mm))
        np.testing.assert_allclose(result["sagittal"], expected, atol=0.001)
        np.testing.assert_allclose(result["tangential"], self.circular_mtf, atol=0.001)

    def test_oblique_chief_projects_tangential_cutoff(self):
        tilt = np.deg2rad(19.0)
        cosine, sine = np.cos(tilt), np.sin(tilt)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])

        class TiltedOptics(IdealOptics):
            def _shoot(self, wave, pupil, *launch):
                rays = super()._shoot(wave, pupil, *launch)
                rays.directions = rays.directions @ rotation.T
                return rays

        result = diffraction_mtf(TiltedOptics(), [0, 0], wave=0, frequency_lpmm=self.frequency)
        fraction = np.clip(self.frequency / (self.frequency[-1] * cosine), 0.0, 1.0)
        projected = 2 / np.pi * (np.arccos(fraction) - fraction * np.sqrt(1 - fraction**2))
        np.testing.assert_allclose(result["tangential"], projected, atol=0.001)
        np.testing.assert_allclose(result["sagittal"], self.circular_mtf, atol=0.001)

    def test_common_psf_energy_and_position(self):
        result = diffraction_mtf(self.optics, [0, 0], samples=64, include_psf=True)
        psf = result["psf"]
        energy = np.asarray(psf["normalized_pixel_energy"])
        self.assertAlmostEqual(energy.sum(), 1.0, places=12)
        self.assertAlmostEqual(energy.sum(axis=0) @ psf["sagittal_position_um"], 1.5, delta=0.025)


if __name__ == "__main__":
    unittest.main()
