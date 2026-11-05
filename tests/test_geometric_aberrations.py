"""Regression checks for independent geometric aberration conventions."""
import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone import disk_quadrature
from scripts.ai_zemax.standalone_aberrations import analyze_aberrations, axis_crossings_image_um
from scripts.ai_zemax.standalone_system import OpticalSystem


def refracting_surface(infinite=False):
    first, image = (0., 90.) if infinite else (100., 325.)
    return {
        'vertices_mm': [first, image],
        'surface_rows': [{'curvature_mm_inv': 1./30.}, {}],
        'wavelengths_um': [.486, .55, .656],
        'indices_by_wave': [[1., index, index] for index in (1.51, 1.5, 1.49)],
        'primary_wave': 1,
        'stop_surface': 1,
        'object_at_infinity': infinite,
        'aperture_type': 'enpd' if infinite else 'obna',
        'aperture_value': 3. if infinite else .015,
        'field_type': 'real_image_height' if infinite else 'object_height',
        'fields': [[0., 0.], [.5, 1.]],
    }


class GeometricAberrationTests(unittest.TestCase):
    def evaluate(self, infinite=False):
        pupil, weights = disk_quadrature(8, 24)
        system = OpticalSystem(refracting_surface(infinite))
        report = analyze_aberrations(system, pupil=pupil, weights=weights)
        json.dumps(report, allow_nan=False)
        return report

    def test_finite_conjugate_longitudinal_reference_and_units(self):
        report = self.evaluate()
        primary_axis = report['axial'][1]
        self.assertAlmostEqual(primary_axis['paraxial_focus_shift_image_um'], 0.)
        self.assertEqual(primary_axis['paraxial_axial_color_image_um'], 0.)
        self.assertLess(report['axial'][0]['paraxial_axial_color_image_um'], 0.)
        self.assertGreater(report['axial'][2]['paraxial_axial_color_image_um'], 0.)
        self.assertAlmostEqual(report['object_to_image_longitudinal_magnification'][1], 3.375)
        on_axis = report['fields'][0]['waves'][1]
        off_axis = report['fields'][1]['waves'][1]
        self.assertAlmostEqual(on_axis['field_curvature_image_um'], 0.)
        self.assertAlmostEqual(on_axis['astigmatism_image_um'], 0.)
        self.assertAlmostEqual(off_axis['field_curvature_image_um'], 3.375*off_axis['field_curvature_object_um'])
        self.assertAlmostEqual(off_axis['astigmatism_image_um'], 3.375*off_axis['astigmatism_object_um'])
        self.assertEqual(primary_axis['longitudinal_spherical_image_um'][0], 0.)
        self.assertEqual(primary_axis['finite_probe_longitudinal_spherical_image_um'][0], 0.)
        self.assertGreater(abs(primary_axis['finite_probe_axis_crossing_shift_image_um'][0]), 1e-6)
        self.assertGreater(abs(primary_axis['longitudinal_spherical_image_um'][-1]), 1.)

    def test_fixed_image_scale_matches_near_axis_chief_trace(self):
        report = self.evaluate()
        system = OpticalSystem(refracting_surface())
        launch = system.launch_field([0., 2e-5])
        for wave, scale in enumerate(report['object_to_image_longitudinal_magnification']):
            chief = system._shoot(wave, np.zeros((1, 2)), *launch)
            magnification = chief.points[0, 1]/2e-5
            self.assertAlmostEqual(scale, system.indices[wave, -1]*magnification**2, places=10)
        self.assertNotAlmostEqual(report['object_to_image_longitudinal_magnification'][0],
                                  report['paraxial_conjugate_longitudinal_magnification'][0])

    def test_fixed_physical_field_and_ray_fan_reference(self):
        report = self.evaluate(infinite=True)
        short, primary, long = report['fields'][1]['waves']
        self.assertIsNone(short['field_curvature_object_um'])
        self.assertIsNone(long['astigmatism_object_um'])
        np.testing.assert_allclose(primary['chief_xy_mm'], [.5, 1.], atol=1e-10)
        np.testing.assert_allclose(short['lateral_color_xy_um'],
                                   1000.*(np.array(short['chief_xy_mm'])-primary['chief_xy_mm']))
        self.assertGreater(abs(long['lateral_color_radial_um']), 1.)
        self.assertAlmostEqual(short['primary_chief_referenced_xy_fans_image_um']['y'][20],
                               short['lateral_color_xy_um'][1])
        self.assertAlmostEqual(short['transverse_fans_image_um']['tangential'][20], 0.)

    def test_axis_crossing_uses_signed_vergence(self):
        points = [[1., 0., 30.], [1., 0., 30.], [0., 0., 30.]]
        directions = [[-.25, 0., 1.], [.25, 0., 1.], [0., 0., 1.]]
        np.testing.assert_allclose(axis_crossings_image_um(points, directions), [4000., -4000., np.nan])


if __name__ == '__main__':
    unittest.main()
