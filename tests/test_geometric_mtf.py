"""Analytic references for geometric image-plane MTF."""
import unittest

import numpy as np

from test_standalone_mtf import IdealOptics
from ai_zemax.standalone_mtf_geometric import geometric_mtf


class GeometricMtfTests(unittest.TestCase):
    def test_fixed_image_axes_and_lateral_color(self):
        frequency = np.linspace(0., 500., 41)
        measured = geometric_mtf(IdealOptics(), [0., 0.], samples=16, frequency_lpmm=frequency)
        np.testing.assert_allclose(measured['sagittal'], np.abs(np.cos(np.pi*frequency*.003)), atol=1e-12)
        np.testing.assert_allclose(measured['tangential'], 1., atol=1e-12)
        self.assertEqual(measured['warnings'], [])

    def test_optional_diffraction_limit_multiplies_geometric_otf(self):
        fraction = np.linspace(0., 1.2, 31)
        frequency = fraction * (400. / .55)
        measured = geometric_mtf(IdealOptics(), [0., 0.], samples=16, frequency_lpmm=frequency,
                                 wave=0, multiply_by_diffraction_limit=True)
        rho = np.clip(fraction, 0., 1.)
        airy = (2. / np.pi) * (np.arccos(rho) - rho*np.sqrt(1.-rho**2))
        np.testing.assert_allclose(measured['sagittal'], airy, atol=1e-12)
        np.testing.assert_allclose(measured['tangential'], airy, atol=1e-12)

    def test_default_curve_resolves_polychromatic_centroid_spread(self):
        class DispersedOptics(IdealOptics):
            def _shoot(self, wave, pupil, *launch):
                rays = super()._shoot(wave, pupil, *launch)
                rays.points[:, 0] += wave * 4.
                return rays

        measured = geometric_mtf(DispersedOptics(), [0., 0.], samples=16)
        frequency = np.asarray(measured['frequency_lpmm'])
        near_first_null = np.argmin(np.abs(frequency - 1./(2.*4.003)))
        self.assertLess(measured['sagittal'][near_first_null], .02)
        self.assertGreater(frequency.max(), 600.)  # Preserve the sharper axis's band.

    def test_default_curve_resolves_a_heavily_blurred_spot(self):
        class BlurredOptics(IdealOptics):
            def _shoot(self, wave, pupil, *launch):
                rays = super()._shoot(wave, pupil, *launch)
                rays.points[:, :2] += 400.*pupil
                return rays

        optics = BlurredOptics()
        measured = geometric_mtf(optics, [0., 0.], samples=128)
        self.assertLess(measured['frequency_lpmm'][-1], .011)
        self.assertLess(min(measured['sagittal']), .02)
        unresolved = geometric_mtf(optics, [0., 0.], samples=32, frequency_lpmm=[0., 1.])
        self.assertIn('geometric_mtf_ray_grid_may_alias_increase_sampling', unresolved['warnings'])


if __name__ == '__main__':
    unittest.main()
