"""Independent-audit failure handling and working-set enrichment."""
import unittest

import numpy as np

from scripts.ai_zemax.standalone_constraint_exchange import run_constraint_exchange


class ExchangeTests(unittest.TestCase):
    def test_finds_interior_failure_and_promotes_complete_quadrature(self):
        seen = []
        def solve(q, samples, iteration):
            seen.append(samples)
            return ([1.] if any(s['field'] == .37 for s in samples) else [0.]), {'status': 'done'}
        def audit(q):
            return [dict(sample=dict(field=f, pupil_rule='rect32'),
                         gates=dict(wfe=1. if f != .37 or q[0] == 1. else -.2))
                    for f in [0., .37, 1.]]
        result = run_constraint_exchange(solve, audit, [dict(field=0., pupil_rule='gq8')], [0.])
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['history']), 2)
        self.assertIn(dict(field=.37, pupil_rule='rect32'), seen[1])
        self.assertFalse(result['continuous_domain_certified'])

    def test_no_false_pass_on_nonfinite_or_already_active_failure(self):
        sample = dict(field=0., pupil_rule='gq8')
        for value in [float('nan'), -.01]:
            result = run_constraint_exchange(lambda q, s, i: (q, {}),
                         lambda q: [dict(sample=sample, gates={'wfe': value})], [sample], [0.])
            self.assertFalse(result['passed'])
            self.assertEqual(result['termination'], 'active_set_stalled')

    def test_fixed_audit_domain_and_finite_solver_output(self):
        count = [0]
        def audit(q):
            count[0] += 1
            return [dict(sample=dict(field=float(count[0])), gates={'wfe': -.2})]
        with self.assertRaisesRegex(ValueError, 'domain changed'):
            run_constraint_exchange(lambda q, s, i: (q, {}), audit, [dict(field=0.)], [0.])
        with self.assertRaisesRegex(ValueError, 'invalid parameter'):
            run_constraint_exchange(lambda q, s, i: ([np.nan], {}), audit, [dict(field=0.)], [0.])

    def test_audit_cannot_silently_drop_a_gate_after_enrichment(self):
        count = [0]
        def audit(q):
            count[0] += 1
            gates = dict(wfe=-.1, physical_gap=.2) if count[0] == 1 else dict(wfe=.1)
            return [dict(sample=dict(field=1.), gates=gates)]
        with self.assertRaisesRegex(ValueError, 'gate layout changed'):
            run_constraint_exchange(lambda q, s, i: (q, {}), audit, [dict(field=0.)], [0.])

    def test_invalid_flag_must_be_explicit_boolean(self):
        sample = dict(field=0.)
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            run_constraint_exchange(lambda q, s, i: (q, {}),
                lambda q: [dict(sample=sample, gates=dict(wfe=1.), valid='false')], [sample], [0.])


if __name__ == '__main__':
    unittest.main()
