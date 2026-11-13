"""Bounded physical controls, whole-population gates and lifted derivatives."""
import tempfile
import unittest

import numpy as np

from scripts.ai_zemax.standalone_scenarios import ScenarioProblem


def evaluate(q, payload, controls):
    shift = payload['shift']
    focus = controls[0] if len(controls) else 0.
    wave = np.array([q[0]+shift+focus, .02])
    return dict(gates=dict(wfe=np.array([.1-np.linalg.norm(wave)]),
                           clearance=np.array([1.-q[0]**2])), vectors={'wave_vector': wave[None]})


def jacobian(q, payload, controls, row):
    columns = 1+len(controls)
    vj = np.zeros((1, 2, columns)); vj[0, 0, :] = 1.
    v = row['vectors']['wave_vector'][0]
    return dict(gates=dict(wfe=(-v@vj[0]/np.linalg.norm(v))[None],
                           clearance=np.r_[-2*q[0], np.zeros(columns-1)][None]),
                vectors=dict(wave_vector=vj))


def problem(compensated=False):
    return ScenarioProblem([[-.5, .5]], {'nominal': {'shift': 0.}, 'positive': {'shift': .22},
        'negative': {'shift': -.22}}, evaluate, lambda q: dict(q=q.tolist()),
        compensator_bounds={'nominal': [], 'positive': [[-.25, .25]], 'negative': [[-.25, .25]]}
            if compensated else None, local_jacobian=jacobian,
        local_norm_specs={'wfe': dict(vector='wave_vector', gain=1., offset=.1)})


class ScenarioTests(unittest.TestCase):
    def test_bounded_controls_restore_all_scenarios_and_keep_nominal_uncompensated(self):
        p = problem(True)
        with tempfile.TemporaryDirectory() as directory:
            result = p.solve(directory, 15, .25, .2, .0001, subproblem_backend='auto',
                hard_gate_keys=['clearance'], jacobian_provider=p.jacobian,
                phase2_iterations=3, margin_keys=['wfe'], verbose=False)
        self.assertTrue(result['strict_feasible'])
        saved = p.model(np.array(result['q']))
        self.assertEqual(saved['scenarios'][0]['controls'], [])
        self.assertTrue(all(abs(c['controls'][0]) <= .25 for c in saved['scenarios'][1:]))
        self.assertFalse(saved['statistical_yield_estimated'])
        self.assertIsNone(result['final_acceptance']['passed'])

    def test_no_compensator_cannot_claim_these_conflicting_scenarios_pass(self):
        p = problem()
        with tempfile.TemporaryDirectory() as directory:
            result = p.solve(directory, 5, .25, .2, .0001, jacobian_provider=p.jacobian, verbose=False)
        self.assertFalse(result['strict_feasible'])
        self.assertEqual(len(result['q']), 1)

    def test_lifted_jacobian_and_signed_vectors_match_independent_differences(self):
        p = problem(True); q = np.array([.03, -.04, .05]); base = p.batch([q])[0]
        keys = tuple(base['gates']); actual = p.jacobian(q, base, keys); eps = 1e-6
        for i in range(3):
            a, b = p.batch([q+eps*np.eye(3)[i], q-eps*np.eye(3)[i]])
            expected = np.concatenate([(a['gates'][k]-b['gates'][k]).ravel()/(2*eps) for k in keys])
            np.testing.assert_allclose(actual['slacks'][:, i], expected, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(actual['vectors']['wave_vector'][..., i],
                (a['wave_vector']-b['wave_vector'])/(2*eps), atol=1e-9)
        # An adjustment to the positive scenario cannot repair a negative-scenario gate.
        self.assertEqual(actual['slacks'][2, 1], 0.)

    def test_failed_or_nonfinite_scenario_invalidates_whole_candidate(self):
        for row in [None, {'valid': False}, {'gates': {'wave': [np.nan]}}]:
            p = ScenarioProblem([[-1., 1.]], {'nominal': {}, 'bad': {}},
                lambda q, s, c: row, lambda q: {})
            self.assertIsNone(p.batch([[0.]])[0])

    def test_undeclared_controls_and_changing_layout_fail_closed(self):
        with self.assertRaises(ValueError):
            ScenarioProblem([[-1., 1.]], {'a': {}, 'b': {}}, evaluate, lambda q: {},
                            compensator_bounds={'a': [[-1., 1.]]})
        p = ScenarioProblem([[-1., 1.]], {'a': {'count': 1}, 'b': {'count': 2}},
            lambda q, s, c: dict(gates={'wave': np.ones(s['count'])}), lambda q: {})
        with self.assertRaises(ValueError):
            p.batch([[0.]])
