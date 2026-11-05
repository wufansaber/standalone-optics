"""Fixed-image spot plotting data tests."""

import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone_spot import spot_diagram
from scripts.ai_zemax.standalone_system import OpticalSystem, rectangular_pupil


def model():
    return {
        "vertices_mm": [0., 90.],
        "surface_rows": [{"curvature_mm_inv": 1./30.}, {}],
        "wavelengths_um": [.486, .55, .656],
        "indices_by_wave": [[1., n, n] for n in (1.51, 1.50, 1.49)],
        "primary_wave": 1,
        "stop_surface": 1,
        "object_at_infinity": True,
        "aperture_type": "enpd",
        "aperture_value": 3.,
        "field_type": "real_image_height",
        "fields": [[.5, 1.]],
    }


class SpotDiagramTests(unittest.TestCase):
    def test_primary_chief_reference_and_same_weighted_rms(self):
        prescription = model()
        prescription["gaussian_apodization"] = 0.7
        system = OpticalSystem(prescription)
        field = system.fields[0]
        pupil, weights = rectangular_pupil(12)
        report = spot_diagram(system, field, samples=12)
        self.assertEqual(report["status"], "calculated")
        self.assertEqual(report["pupil_ray_count"], len(pupil))
        np.testing.assert_allclose(report["reference_chief_xy_mm"], field, atol=1.e-10)
        for wave, row in enumerate(report["waves"]):
            metrics, rays, _ = system.evaluate(field, wave, pupil, weights, use_primary_field=True)
            expected = 1000.*(rays.points[:, :2] - report["reference_chief_xy_mm"])
            np.testing.assert_allclose(row["intercepts_xy_um"], expected, atol=1.e-10)
            self.assertAlmostEqual(row["rms_radius_um"], metrics["spot_rms_um"], places=10)
        self.assertGreater(abs(report["waves"][0]["centroid_xy_um"][1] -
                               report["waves"][2]["centroid_xy_um"][1]), 1.)
        json.dumps(report, allow_nan=False)

    def test_aperture_clipping_keeps_transmitted_spot(self):
        prescription = model()
        prescription["surface_rows"][0]["aperture_radius_mm"] = 0.75
        result = spot_diagram(OpticalSystem(prescription), [0., 0.], samples=12)
        self.assertEqual(result["status"], "calculated")
        row = result["waves"][0]
        self.assertGreater(row["aperture_clipped_rays"], 0)
        self.assertEqual(len(row["intercepts_xy_um"]) + row["aperture_clipped_rays"],
                         result["pupil_ray_count"])

    def test_negative_segment_strict_and_signed_replay(self):
        prescription = model()
        prescription.update(
            vertices_mm=[0., .02, 2.],
            surface_rows=[{"curvature_mm_inv": .2}, {"curvature_mm_inv": -.2}, {}],
            indices_by_wave=[[1.]*4 for _ in range(3)],
            aperture_value=2.,
            field_type="angle",
        )
        strict = spot_diagram(OpticalSystem(prescription), [0., 0.], samples=12)
        self.assertEqual(strict["status"], "incomplete")
        self.assertEqual(strict["waves"][0]["first_failure"]["reason"], "negative_segment")
        self.assertNotIn("intercepts_xy_um", strict["waves"][0])
        signed = spot_diagram(OpticalSystem(prescription, allow_signed_segments=True),
                              [0., 0.], samples=12)
        self.assertEqual(signed["status"], "signed_comparison_only")
        self.assertGreater(signed["waves"][0]["rays_with_negative_segments"], 0)
        self.assertLess(signed["waves"][0]["shortest_signed_segment_mm"], 0.)


if __name__ == "__main__":
    unittest.main()
