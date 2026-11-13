import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin
from scripts.ai_zemax.standalone_retraction import project_feasible


class RetractionTests(unittest.TestCase):
    def test_projection_uses_actual_nonlinear_gates_inside_original_box(self):
        # A circle boundary cannot be enforced by a first-order step alone.
        q = project_feasible(np.array([1.2, .3]), np.array([.5, -.5]), np.array([1.5, .5]),
            lambda q: np.array([1.-q@q]), lambda q: -2*q[None])
        self.assertIsNotNone(q)
        self.assertGreaterEqual(1.-q@q, 0.)
        self.assertGreaterEqual(q[0], .5)

    def test_infeasible_projection_returns_no_candidate(self):
        q = project_feasible(np.array([0.]), np.array([-.1]), np.array([.1]),
            lambda q: np.array([q[0]-.2]), lambda q: np.ones((1, 1)))
        self.assertIsNone(q)

    def test_corrected_trial_is_retraced_and_cannot_spend_real_hard_slack(self):
        class P(EfficientSolveMixin):
            bounds = np.array([[-1., 1.]])
            def batch(self, qs):
                return [dict(goal=np.array([q[0]-.2]), clearance=np.array([.3-q[0]])) for q in qs]
            def slacks(self, row):
                return row
            def model(self, q):
                return dict(q=q.tolist())
        # A plausible-looking correction is physically invalid. Re-tracing must reject it.
        with tempfile.TemporaryDirectory() as d, patch.object(P, '_local_proposal',
                return_value=(np.array([.5]), {'usable': True})):
            result = P().solve(d, 1, .5, .2, 0., hard_gate_keys=['clearance'],
                trial_retractor=lambda trial, low, high: np.array([.4]), verbose=False)
        self.assertEqual(result['q'], [0.])
        self.assertEqual(result['counters']['accepted_steps'], 0)
        self.assertGreater(result['counters']['retraction_calls'], 0)

    def test_retractor_outside_trust_box_is_rejected(self):
        class P(EfficientSolveMixin):
            bounds = np.array([[-1., 1.]])
            batch = lambda self, qs: [dict(g=np.array([q[0]-.2])) for q in qs]
            slacks = lambda self, row: row
            model = lambda self, q: dict(q=q.tolist())
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, 'trust box'):
            P().solve(d, 2, .5, .2, 0., trial_retractor=lambda q, low, high: np.array([1.]), verbose=False)
