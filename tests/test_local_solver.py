"""Behavioral checks for convergence, cache reuse and constrained margin search."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin


class Problem(EfficientSolveMixin):
    bounds = np.array([[-2., 2.]])

    def __init__(self, function):
        self.function = function

    def batch(self, qs):
        return [self.function(q) for q in qs]

    def slacks(self, row):
        return row

    def model(self, q):
        return dict(parameters=np.asarray(q).tolist())


class SolverTests(unittest.TestCase):
    def solve(self, problem, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            result = problem.solve(Path(directory), 12, .5, .2, .0001, False, False,
                                   verbose=False, **kwargs)
            self.assertTrue((Path(directory)/'BEST.json').is_file())
            self.assertTrue((Path(directory)/'PROGRESS.json').is_file())
            return result

    def test_numerical_stop_does_not_claim_strict_acceptance(self):
        result = self.solve(Problem(lambda q: {'geometry': np.array([-2e-9])}))
        self.assertEqual(result['status'], 'numerical_feasibility')
        self.assertTrue(result['numerical_converged'])
        self.assertFalse(result['strict_feasible'])
        self.assertEqual(result['counters']['linearizations'], 0)
        self.assertEqual(result['counters']['traced_candidates'], 1)
        self.assertIsNone(result['final_acceptance']['passed'])

    def test_rejected_steps_reuse_linearization(self):
        result = self.solve(Problem(lambda q: {'aperture': np.array([-1.])}),
                            jacobian_provider=lambda q, row, keys: np.ones((1, 1)),
                            stagnation_retries=4)
        self.assertEqual(result['status'], 'stagnation')
        self.assertEqual(result['counters']['linearizations'], 1)
        self.assertEqual(result['counters']['derivative_cache_hits'], 3)
        self.assertEqual(result['counters']['rejected_steps'], 4)

    def test_finite_difference_restores_feasibility_and_validator_is_independent(self):
        result = self.solve(Problem(lambda q: {'focus': np.array([q[0]-.2])}),
                            strict_validator=lambda model: dict(passed=False, reason='independent gate'))
        self.assertTrue(result['numerical_converged'])
        self.assertGreaterEqual(result['q'][0], .2-1e-8)
        self.assertFalse(result['final_acceptance']['passed'])
        self.assertEqual(result['final_acceptance']['reason'], 'independent gate')
        self.assertIsNotNone(result['history'][0]['rho'])

    def test_margin_phase_preserves_all_gates_and_excludes_order_objective(self):
        result = self.solve(Problem(lambda q: {'clearance': np.array([q[0], 1.-q[0]]),
                                             'quality_order': np.array([.01])}),
                            phase2_iterations=8, margin_scales={'clearance': .1})
        self.assertEqual(result['margin_keys'], ['clearance'])
        self.assertAlmostEqual(result['q'][0], .5, places=5)
        self.assertGreater(result['margin'], 4.999)
        self.assertTrue(result['strict_feasible'])

    def test_nonfinite_subproblem_cannot_reach_tracer(self):
        fake = SimpleNamespace(x=np.full(3, np.nan), success=True, status=0, message='bad', nit=1)
        with patch('scripts.ai_zemax.standalone_local_solver.minimize', return_value=fake):
            result = self.solve(Problem(lambda q: {'focus': np.array([-1.])}), stagnation_retries=2)
        self.assertEqual(result['counters']['accepted_steps'], 0)
        # The only evaluations are the baseline and one derivative probe.
        self.assertEqual(result['counters']['traced_candidates'], 2)
        self.assertFalse(result['history'][0]['subproblem']['usable'])

    def test_inexact_finite_feasible_subproblem_is_used(self):
        # One variable, one relaxed constraint, one epigraph scalar.
        fake = SimpleNamespace(x=np.array([.3, 0., 0.]), success=False, status=9, message='iteration limit', nit=85)
        with patch('scripts.ai_zemax.standalone_local_solver.minimize', return_value=fake):
            result = self.solve(Problem(lambda q: {'focus': np.array([q[0]-.2])}))
        self.assertTrue(result['numerical_converged'])
        self.assertEqual(result['counters']['accepted_steps'], 1)
        self.assertFalse(result['history'][0]['subproblem']['success'])
        self.assertTrue(result['history'][0]['subproblem']['usable'])

    def test_fixed_parameter_and_narrow_bounds(self):
        p = Problem(lambda q: {'position': np.array([q[0]+q[1]-1e-5])})
        p.bounds = np.array([[0., 0.], [0., 2e-5]])
        result = self.solve(p)
        self.assertTrue(result['numerical_converged'])
        self.assertEqual(result['q'][0], 0.)
        self.assertLessEqual(result['q'][1], 2e-5)

    def test_unknown_margin_or_tolerance_names_are_rejected(self):
        p = Problem(lambda q: {'geometry': np.array([1.])})
        with self.assertRaises(ValueError):
            self.solve(p, margin_keys=['misspelled'])
        with self.assertRaises(ValueError):
            self.solve(p, tolerances={'misspelled': 10.})

    def test_signed_vector_cones_with_svd_preserve_exact_affine_norms(self):
        class WaveProblem(Problem):
            wfe_target = axis_target = .2
            condition_scale = .5

            def __init__(self):
                super().__init__(lambda q: {'wave_vector': np.array([[[q[0]-.6, .1]]])})

            def slacks(self, row):
                wave = np.linalg.norm(row['wave_vector'], axis=-1)
                return {'wfe': 1.-wave/self.wfe_target, 'axis': 1.-wave[:, 0]/self.axis_target}

        with tempfile.TemporaryDirectory() as directory:
            result = WaveProblem().solve(directory, 10, 1., .2, .0001, True, False, verbose=False)
        self.assertTrue(result['numerical_converged'])
        self.assertLess(np.linalg.norm([result['q'][0]-.6, .1]), .2+2e-9)
        self.assertEqual(result['counters']['qr_builds'], result['counters']['linearizations'])
        self.assertEqual(result['counters']['svd_builds'], result['counters']['linearizations'])

    def test_margin_phase_cannot_spend_numerical_tolerance_on_positive_gate(self):
        p = Problem(lambda q: {'working_margin': np.array([q[0]]),
                               'physical_gate': np.array([1e-10-q[0]])})
        result = self.solve(p, phase2_iterations=5, margin_keys=['working_margin'], tolerances=1e-3)
        self.assertGreaterEqual(result['gates']['physical_gate'], 0.)

    def test_underdetermined_residual_svd_keeps_nullspace_and_fixed_parameter(self):
        class UnderdeterminedWave(Problem):
            bounds = np.array([[-2., 2.], [-2., 2.], [-2., 2.], [0., 0.]])
            wfe_target = axis_target = .2
            condition_scale = .5

            def __init__(self):
                super().__init__(lambda q: {'wave_vector': np.array([[[q[0]-.6, q[1]+.1]]]),
                                             'null_margin': np.array([q[2]-.1])})

            def slacks(self, row):
                wave = np.linalg.norm(row['wave_vector'], axis=-1)
                return {'wfe': 1.-wave/.2, 'axis': 1.-wave[:, 0]/.2,
                        'clearance': row['null_margin']}

        with tempfile.TemporaryDirectory() as directory:
            result = UnderdeterminedWave().solve(directory, 10, 1., .2, .0001,
                                                 True, False, verbose=False)
        self.assertTrue(result['numerical_converged'])
        self.assertGreaterEqual(result['q'][2], .1-1e-8)
        self.assertEqual(result['q'][3], 0.)
        self.assertLessEqual(np.linalg.norm([result['q'][0]-.6, result['q'][1]+.1]), .2+2e-9)
        self.assertGreater(result['counters']['svd_builds'], 0)

    def test_external_jacobian_avoids_parameter_probes(self):
        result = self.solve(Problem(lambda q: {'focus': np.array([q[0]-.2])}),
                            jacobian_provider=lambda q, row, keys: np.ones((1, 1)))
        self.assertTrue(result['numerical_converged'])
        self.assertEqual(result['derivative_backend'], 'external')
        self.assertEqual(result['counters']['traced_candidates'], 1+6*result['counters']['proposals'])

    def test_acceptable_smaller_trial_survives_best_score_with_poor_prediction(self):
        p = Problem(lambda q: {'gate': np.array([-1.+(0. if q[0] == 0. else .01 if q[0] >= .1 else .004)])})
        with tempfile.TemporaryDirectory() as directory, patch.object(
                p, '_local_proposal', return_value=(np.array([.5]), {'usable': True})):
            result = p.solve(directory, 1, .5, .2, .0001, False, False, verbose=False,
                             jacobian_provider=lambda q, row, keys: np.ones((1, 1)))
        self.assertEqual(result['counters']['accepted_steps'], 1)
        self.assertAlmostEqual(result['q'][0], .14)
        self.assertGreaterEqual(result['history'][0]['rho'], .05)

    def test_norm_spec_validation_uses_current_base_after_multiple_steps(self):
        class WeightedWave(Problem):
            wfe_target = axis_target = .2
            local_norm_specs = {
                'wfe': dict(vector='wave_vector', gain=10., offset=2.),
                'axis': dict(vector='wave_vector', gain=15., offset=3., axis=True)}

            def __init__(self):
                super().__init__(lambda q: {'wave_vector': np.array([[[q[0]-.6, .1]]])})

            def slacks(self, row):
                wave = np.linalg.norm(row['wave_vector'], axis=-1)
                return {'wfe': 2.*(1.-wave/.2), 'axis': 3.*(1.-wave[:, 0]/.2)}

        with tempfile.TemporaryDirectory() as directory:
            result = WeightedWave().solve(directory, 12, .1, .2, .0001, True, False,
                                          max_trust=.1, verbose=False)
        self.assertTrue(result['numerical_converged'])
        self.assertGreater(result['counters']['linearizations'], 1)
        self.assertEqual(result['counters']['cone_linear_fallbacks'], 0)

    def test_explicit_weighted_cones_match_adapter_constraints(self):
        class WeightedWave(Problem):
            wfe_target = axis_target = .2
            condition_scale = .4
            local_norm_specs = {
                'wfe': dict(vector='wave_vector', gain=10., offset=2.),
                'axis': dict(vector='wave_vector', gain=15., offset=3., axis=True)}

            def __init__(self):
                super().__init__(lambda q: {'wave_vector': np.array([[[q[0]-.6, .1]]])})

            def slacks(self, row):
                wave = np.linalg.norm(row['wave_vector'], axis=-1)
                return {'wfe': 2.*(1.-wave/.2), 'axis': 3.*(1.-wave[:, 0]/.2)}

        p = WeightedWave()
        with tempfile.TemporaryDirectory() as directory:
            result = p.solve(directory, 10, 1., .2, .0001, True, False, verbose=False)
        self.assertTrue(result['numerical_converged'])
        self.assertEqual(result['counters']['cone_linear_fallbacks'], 0)
        self.assertLessEqual(np.linalg.norm([result['q'][0]-.6, .1]), .2+1e-9)

        # Without an explicit matching model, preserve measured weighted slacks
        # and their Jacobian rather than silently replacing them by unit cones.
        p.local_norm_specs = {}
        with tempfile.TemporaryDirectory() as directory:
            fallback = p.solve(directory, 10, 1., .2, .0001, True, False, verbose=False)
        self.assertTrue(fallback['numerical_converged'])
        self.assertGreater(fallback['counters']['cone_linear_fallbacks'], 0)


if __name__ == '__main__':
    unittest.main()
