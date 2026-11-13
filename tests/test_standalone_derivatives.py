"""Central differences independently check smooth optical tangents and boundaries."""

from dataclasses import replace
import unittest

import numpy as np

from scripts.ai_zemax.standalone import (Surface, disk_quadrature, paraxial_transfer,
                                        reference_opd_mm, trace, wavefront_rms)
from scripts.ai_zemax.standalone_derivatives import (
    RayParameter, paraxial_transfer_with_jacobian, reference_opd_with_jacobian,
    trace_with_jacobian, wavefront_rms_with_jacobian)


def perturb(surfaces, parameter, delta):
    rows = list(surfaces)
    i, key = parameter.surface_index, parameter.attribute
    change = delta*parameter.scale
    if key == 'gap_before_mm':
        for j in range(i, len(rows)):
            rows[j] = replace(rows[j], vertex_mm=rows[j].vertex_mm+change)
    elif key == 'even_coefficient':
        coefficients = list(rows[i].even_coefficients)
        coefficients[parameter.coefficient_index] += change
        rows[i] = replace(rows[i], even_coefficients=tuple(coefficients))
    else:
        rows[i] = replace(rows[i], **{key: getattr(rows[i], key)+change})
    return tuple(rows)


class DerivativeTests(unittest.TestCase):
    def setUp(self):
        self.surfaces = (Surface(10., 1.52, .024, -.6, (0., 1e-6), 8.),
                         Surface(13., 1., -.018, .2, (1e-4, -3e-7), 8.),
                         Surface(80., 1., aperture_radius_mm=20.))
        self.parameters = (
            RayParameter(0, 'curvature_mm_inv', scale=.001),
            RayParameter(1, 'curvature_mm_inv', scale=.001),
            RayParameter(0, 'conic', scale=.1),
            RayParameter(0, 'even_coefficient', 0, 1e-4),
            RayParameter(0, 'even_coefficient', 1, 1e-6),
            RayParameter(1, 'vertex_mm', scale=.1),
            RayParameter(1, 'gap_before_mm', scale=.1),
            RayParameter(0, 'index_after', scale=.01),
        )
        self.pupil, self.weights = disk_quadrature(4, 12)
        self.directions = np.column_stack((.055*self.pupil, np.ones(len(self.pupil))))

    def test_endpoints_directions_and_opl_match_central_differences(self):
        eps = 1e-4
        for index in (1.49, 1.52, 1.56):
            surfaces = (replace(self.surfaces[0], index_after=index), *self.surfaces[1:])
            for field in (0., .7, 1.4):
                origin = [0., field, 0.]
                answer = trace_with_jacobian(surfaces, origin, self.directions, self.parameters)
                self.assertTrue(answer.differentiable.all())
                base = trace(surfaces, origin, self.directions)
                np.testing.assert_array_equal(answer.trace.valid, base.valid)
                np.testing.assert_array_equal(answer.trace.points, base.points)
                for j, parameter in enumerate(self.parameters):
                    plus = trace(perturb(surfaces, parameter, eps), origin, self.directions)
                    minus = trace(perturb(surfaces, parameter, -eps), origin, self.directions)
                    for key, derivative in (('points', answer.points_jacobian),
                                            ('directions', answer.directions_jacobian),
                                            ('opl_mm', answer.opl_jacobian)):
                        expected = (getattr(plus, key)-getattr(minus, key))/(2*eps)
                        np.testing.assert_allclose(derivative[..., j], expected, atol=7e-9, rtol=2e-6,
                                                   err_msg=f'{field=} {index=} {parameter=} {key=}')

    def test_paraxial_transfer_matches_finite_differences(self):
        matrix, jac = paraxial_transfer_with_jacobian(self.surfaces, 0., self.parameters)
        np.testing.assert_allclose(matrix, paraxial_transfer(self.surfaces, 0.), atol=0., rtol=0.)
        eps = 1e-4
        for j, parameter in enumerate(self.parameters):
            expected = (paraxial_transfer(perturb(self.surfaces, parameter, eps), 0.)-
                        paraxial_transfer(perturb(self.surfaces, parameter, -eps), 0.))/(2*eps)
            np.testing.assert_allclose(jac[..., j], expected, atol=4e-9, rtol=1e-6)

    def test_wavefront_sphere_piston_and_moving_tilt_basis(self):
        width, count = len(self.parameters), len(self.pupil)
        raw = self.directions.copy()
        raw[:, 1] -= .02
        chief_raw = np.array([[0., -.02, 1.]])
        origin = np.array([0., .7, 0.])
        od = np.zeros((count, 3, width))
        od[:, 1, 1] = .02
        dd = np.zeros((count, 3, width))
        dd[:, 1, 2] = .001
        bundle = trace_with_jacobian(self.surfaces, origin, raw, self.parameters,
                                     origin_jacobian=od, direction_jacobian=dd)
        chief = trace_with_jacobian(self.surfaces, origin, chief_raw, self.parameters,
                                    origin_jacobian=od[:1], direction_jacobian=dd[:1])
        ref, wavelength = 100., .55
        dr = np.arange(width)*.07
        dw = np.arange(width)*.0001
        opd, jac = reference_opd_with_jacobian(bundle, chief, ref, pupil_distance_jacobian=dr)
        actual = wavefront_rms_with_jacobian(opd, jac, self.weights, wavelength,
                    bundle.trace.directions[:, :2], tilt_jacobian=bundle.directions_jacobian[:, :2],
                    wavelength_jacobian=dw)
        expected = wavefront_rms(opd, self.weights, wavelength, bundle.trace.directions[:, :2])
        self.assertAlmostEqual(actual.rms_waves, expected[0], places=12)
        eps = 1e-4
        for j, parameter in enumerate(self.parameters):
            outcomes = []
            for delta in (eps, -eps):
                surfaces = perturb(self.surfaces, parameter, delta)
                trace_bundle = trace(surfaces, origin+delta*od[:, :, j], raw+delta*dd[:, :, j])
                trace_chief = trace(surfaces, origin+delta*od[:1, :, j], chief_raw+delta*dd[:1, :, j])
                wave = reference_opd_mm(trace_bundle, trace_chief, ref+delta*dr[j])
                outcomes.append((wave, *wavefront_rms(wave, self.weights, wavelength+delta*dw[j],
                                                       trace_bundle.directions[:, :2])))
            expected_opd = (outcomes[0][0]-outcomes[1][0])/(2*eps)
            expected_rms = (outcomes[0][1]-outcomes[1][1])/(2*eps)
            expected_residual = (outcomes[0][2]-outcomes[1][2])/(2*eps)
            np.testing.assert_allclose(jac[:, j], expected_opd, atol=1e-9, rtol=2e-5)
            np.testing.assert_allclose(actual.rms_jacobian[j], expected_rms, atol=3e-7, rtol=2e-5)
            np.testing.assert_allclose(actual.residual_jacobian[:, j], expected_residual,
                                       atol=4e-7, rtol=4e-5)

    def test_plane_axis_and_launch_normalization(self):
        rows = (Surface(10., 1.5), Surface(15., 1.))
        parameters = (RayParameter(0, 'curvature_mm_inv'), RayParameter(1, 'gap_before_mm'))
        raw = np.array([[0., 0., 2.], [.01, .02, 2.]])
        dd = np.zeros((2, 3, 2))
        dd[:, 0, 0] = .01
        measured = trace_with_jacobian(rows, [0., 0., 0.], raw, parameters, direction_jacobian=dd)
        self.assertTrue(measured.differentiable.all())
        eps = 1e-5
        for j, parameter in enumerate(parameters):
            a = trace(perturb(rows, parameter, eps), [0., 0., 0.], raw+eps*dd[..., j])
            b = trace(perturb(rows, parameter, -eps), [0., 0., 0.], raw-eps*dd[..., j])
            np.testing.assert_allclose(measured.points_jacobian[..., j], (a.points-b.points)/(2*eps),
                                       atol=1e-8, rtol=1e-5)

    def test_invalid_and_boundary_rays_never_have_finite_gradients(self):
        rows = (Surface(10., 1.5, aperture_radius_mm=1.), Surface(15.))
        parameters = (RayParameter(0, 'curvature_mm_inv'),)
        raw = np.array([[0., 0., 1.], [.1, 0., 1.], [.2, 0., 1.]])
        result = trace_with_jacobian(rows, [0., 0., 0.], raw, parameters)
        self.assertEqual(result.trace.valid.tolist(), [True, True, False])
        self.assertEqual(result.differentiable.tolist(), [True, False, False])
        self.assertEqual(result.derivative_reason[1], 'aperture_boundary')
        self.assertEqual(result.derivative_reason[2], 'invalid_primal')
        self.assertTrue(np.isnan(result.points_jacobian[1:]).all())
        # A zero-length first propagation is on the forward/signed boundary.
        zero = trace_with_jacobian((Surface(0.), Surface(2.)), [0., 0., 0.], [[0., 0., 1.]], parameters)
        self.assertTrue(zero.trace.valid[0])
        self.assertFalse(zero.differentiable[0])

    def test_invalid_parameters_and_zero_wavefront_rms(self):
        with self.assertRaises(ValueError):
            trace_with_jacobian(self.surfaces, [0., 0., 0.], self.directions,
                                [RayParameter(0, 'aperture_radius_mm')])
        with self.assertRaises(ValueError):
            trace_with_jacobian(self.surfaces, [0., 0., 0.], self.directions,
                                [RayParameter(0, 'even_coefficient', 9)])
        wave = wavefront_rms_with_jacobian(np.zeros(4), np.ones((4, 1)), np.ones(4), .55)
        self.assertFalse(wave.rms_differentiable)
        self.assertTrue(np.isnan(wave.rms_jacobian).all())
        np.testing.assert_allclose(wave.residual_jacobian, 0., atol=1e-12)

    def test_tir_and_conic_domain_fail_closed(self):
        parameters = (RayParameter(0, 'curvature_mm_inv'),)
        tir = trace_with_jacobian((Surface(10., 1.), Surface(15.)),
                                  [0., 0., 0.], [[1., 0., 1.]], parameters, object_index=1.5)
        self.assertFalse(tir.trace.valid[0])
        self.assertEqual(tir.trace.reason[0], 'tir')
        self.assertFalse(tir.differentiable[0])
        self.assertTrue(np.isnan(tir.opl_jacobian).all())
        outside = trace_with_jacobian((Surface(10., 1.5, .5), Surface(15.)),
                                      [3., 0., 0.], [[0., 0., 1.]], parameters)
        self.assertFalse(outside.trace.valid[0])
        self.assertFalse(outside.differentiable[0])
        self.assertTrue(np.isnan(outside.points_jacobian).all())


if __name__ == '__main__':
    unittest.main()
