"""Checks of the thick singlet seed model and its mechanical gates."""

import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone_seed_realize import realize_seed
from scripts.ai_zemax.standalone_system import OpticalSystem


SEED = {
    'gaps_mm': [50., 50., 50., 50.],
    'powers_mm_inv': [.024, -.01, .024],
    'matrix': [[-1., 0.], [.01, -1.]],
    'stop_group': 2,
}


class RealizeSeedTests(unittest.TestCase):
    def test_recovers_fixed_conjugates_and_traces_color(self):
        model = realize_seed(SEED, refractive_indices=[[1.54, 1.55, 1.53], [1.52, 1.53, 1.51]],
                             wavelengths_um=[.486, .5876], primary_wave=1)
        json.dumps(model, allow_nan=False)
        info = model['seed_realization']
        np.testing.assert_allclose(info['primary_transfer'][0], [-1., 0.], rtol=0., atol=1e-10)
        np.testing.assert_array_equal(info['group_centers_mm'], [50., 100., 150.])
        np.testing.assert_array_equal(np.sign(info['realized_powers_mm_inv']), [1., -1., 1.])
        self.assertEqual(info['track_mm'], 200.)
        self.assertEqual(model['stop_surface'], 3)
        self.assertAlmostEqual(info['realized_powers_mm_inv'][1], -.01)
        system = OpticalSystem(model)
        for field in system.fields:
            for wave in range(2):
                row, _, _ = system.evaluate(field, wave)
                self.assertEqual(row['rays_with_negative_segments'], 0)

    def test_preserves_stop_choice_and_reports_aperture(self):
        model = realize_seed(dict(SEED, stop_group=3))
        self.assertEqual(model['stop_surface'], 5)
        radii = model['seed_realization']['clear_radii_mm']
        np.testing.assert_array_equal([r['aperture_radius_mm'] for r in model['surface_rows'][:-1]],
                                      np.repeat(radii, 2))

    def test_rejects_invalid_thick_geometry(self):
        with self.assertRaisesRegex(ValueError, 'vertex air gaps'):
            realize_seed(SEED, center_thicknesses_mm=[101., 3., 3.])
        with self.assertRaisesRegex(ValueError, 'edge thickness'):
            realize_seed(SEED, minimum_edge_thickness_mm=2.95)
        with self.assertRaisesRegex(ValueError, 'maximum group radius'):
            realize_seed(SEED, max_group_radius_mm=2.)
        with self.assertRaisesRegex(ValueError, 'does not image'):
            realize_seed(dict(SEED, matrix=[[-1., 1.], [.01, -1.]]))


if __name__ == '__main__':
    unittest.main()
