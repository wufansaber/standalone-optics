"""Cross-backend, true-norm, hard-gate and dependency behavior regressions."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ai_zemax.standalone_conic import available
from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin


class AffineWave(EfficientSolveMixin):
    bounds = np.array([[-2., 2.], [-2., 2.], [0., 0.]])
    condition_scale = .5
    local_norm_specs = {'quality': dict(vector='wave_vector', gain=np.array([2., 4.]), offset=1.)}

    def batch(self, qs):
        return [dict(wave_vector=np.array([[q[0]-.7, .1], [q[1]+.2, .08]]),
                     clearance=np.array([q[0]+.1, 1.-q[0]])) for q in qs]

    def slacks(self, row):
        return dict(quality=1.-np.array([2., 4.])*np.linalg.norm(row['wave_vector'], axis=-1),
                    clearance=row['clearance'])

    def model(self, q):
        return dict(parameters=q.tolist())

    def jacobian(self, q, base, keys):
        vj = np.zeros((2, 2, 3)); vj[0, 0, 0] = vj[1, 0, 1] = 1.
        v = base['wave_vector']; norm = np.linalg.norm(v, axis=-1)
        j = -np.array([2., 4.])[:, None]*np.einsum('ri,rip->rp', v, vj)/norm[:, None]
        return dict(slacks=np.r_[j, [[1., 0., 0.], [-1., 0., 0.]]], vectors={'wave_vector': vj})


@unittest.skipUnless(available(), 'Optional Clarabel backend not installed')
class ConicTests(unittest.TestCase):
    def run_problem(self, backend):
        p = AffineWave()
        with tempfile.TemporaryDirectory() as directory:
            return p.solve(directory, 15, .5, .2, 1e-4, phase2_iterations=5,
                margin_keys=['quality'], hard_gate_keys=['clearance'],
                jacobian_provider=p.jacobian, subproblem_backend=backend, verbose=False)

    def test_weighted_norms_conditioning_fixed_columns_and_margin_agree(self):
        a, b = self.run_problem('slsqp'), self.run_problem('clarabel')
        for result in (a, b):
            self.assertTrue(result['strict_feasible'])
            self.assertIsNone(result['final_acceptance']['passed'])
            self.assertEqual(result['q'][2], 0.)
            self.assertGreater(result['margin'], .679)
        self.assertLess(abs(a['margin']-b['margin']), 2e-4)
        self.assertTrue(all(h['subproblem']['backend'] == 'clarabel' for h in b['history']))

    def test_actual_hard_gate_cannot_be_sacrificed_for_quality(self):
        class Impossible(EfficientSolveMixin):
            bounds = np.array([[-1., 1.]])
            def batch(self, qs):
                return [dict(goal=np.array([q[0]-.8]), physical=np.array([.4-q[0]])) for q in qs]
            def slacks(self, row):
                return row
            def model(self, q):
                return dict(q=q.tolist())
        for backend in ('clarabel', 'slsqp'):
            with tempfile.TemporaryDirectory() as directory:
                result = Impossible().solve(directory, 10, .5, .2, 0., hard_gate_keys=['physical'],
                    subproblem_backend=backend, verbose=False)
            self.assertGreaterEqual(result['gates']['physical'], 0.)
            self.assertFalse(result['strict_feasible'])
            self.assertLessEqual(result['q'][0], .4)

    def test_infeasible_initial_hard_gate_fails_before_solver(self):
        p = AffineWave()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'Initial hard'):
            p.solve(directory, 2, .5, .2, 0., hard_gate_keys=['clearance'], initial_q=[-.2, 0., 0.])


class DependencyTests(unittest.TestCase):
    def test_auto_records_missing_backend_and_explicit_request_fails(self):
        with patch('scripts.ai_zemax.standalone_conic.available', return_value=False):
            p = AffineWave()
            with tempfile.TemporaryDirectory() as directory:
                result = p.solve(directory, 5, .5, .2, .0001, subproblem_backend='auto', verbose=False)
            self.assertEqual(result['history'][0]['subproblem']['fallback_reason'], 'clarabel_not_installed')
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ImportError):
                p.solve(directory, 2, .5, .2, 0., subproblem_backend='clarabel', verbose=False)

    def test_bad_backend_and_negative_regularizer_are_rejected(self):
        for settings in [dict(subproblem_backend='unknown'), dict(damping=-1.)]:
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                kwargs = dict(damping=0.); kwargs.update(settings)
                AffineWave().solve(directory, 2, .5, .2, verbose=False, **kwargs)
