"""MF context, dependencies, weights, budgets and spherical bound proofs."""
from copy import deepcopy
import math
from pathlib import Path
import random
import tempfile
import unittest

from scripts.ai_zemax.standalone_merit_graph import audit, compact, evaluate, main, normalize, write_native
from scripts.ai_zemax.standalone_merit_sphere import spherical_support


class Graph:
    def __init__(self): self.rows = []
    def add(self, op, *p, target=0., weight=0.):
        self.rows.append(dict(operand=op, params=list(p), target=target, weight=weight))
        return len(self.rows)
    def bound(self, value, lower, upper):
        roots = []
        if lower is not None: roots.append(self.add('OPGT', value, target=lower, weight=1.))
        if upper is not None: roots.append(self.add('OPLT', value, target=upper, weight=1.))
        return roots


class MeritGraphTests(unittest.TestCase):
    def test_context_and_weighted_roots_preserved_for_multiple_designs(self):
        g = Graph(); g.add('CONF', 1); g.add('PRIM', 2)
        a = g.add('REAY', 5, 1); b = g.add('REAY', 5, 1)
        g.add('REAY', 5, 1, weight=2.); g.add('REAY', 5, 1, weight=3.)
        g.add('PRIM', 3); g.add('REAY', 5, 1)
        g.add('CONF', 2); g.add('PRIM', 2); g.add('REAY', 5, 1)
        g.add('CONF', 1); g.add('PRIM', 2); g.add('REAY', 5, 1)
        g.add('SUMM', a, b, weight=4.)
        result, report = compact(g.rows)
        self.assertEqual(report['merged_rows'], 2)
        self.assertTrue(report['full_objective_preserved'])
        for design in (-.1, 0., 1.):
            measure = lambda row, ctx: design + 7*ctx[0][0] + ctx[1][0]
            old, new = evaluate(g.rows, measure), evaluate(result, measure)
            for i, j in enumerate(report['row_map_1based'][1:], 1):
                self.assertEqual(old[i], new[j])
        self.assertEqual(sum(r['weight'] > 0 for r in result), 3)

    def test_zero_weight_dependencies_and_diagnostic_default(self):
        g = Graph(); a = g.add('CONS', target=2.); b = g.add('CONS', target=3.)
        s = g.add('PROD', a, b); root = g.bound(s, 5., None)[0]
        diag = g.add('CONS', target=123.)
        full, report = compact(g.rows)
        self.assertIsNotNone(report['row_map_1based'][diag])
        work, report = compact(g.rows, active_rows=[root], audit_identity='frozen-full-audit-sha256')
        self.assertIsNone(report['row_map_1based'][diag])
        self.assertEqual(evaluate(work, lambda *_: 0.)[-1], 5.)
        self.assertEqual(report['omitted_rows'], 1)
        self.assertTrue(report['requires_external_full_audit'])
        self.assertFalse(report['final_acceptance'])

    def test_unknown_primary_state_not_reused_across_configuration_entries(self):
        g = Graph(); g.add('CONF',1); a = g.add('REAY',5,1)
        g.add('PRIM',2); g.add('CONF',1); b = g.add('REAY',5,1)
        result,report=compact(g.rows)
        self.assertNotEqual(report['row_map_1based'][a],report['row_map_1based'][b])

    def test_weighted_dependency_is_deactivated_unless_protected(self):
        g = Graph(); a = g.add('CONS', target=2., weight=7.)
        root = g.add('PROB', a, 1, 3., target=6., weight=4.)
        result, report = compact(g.rows, active_rows=[root], audit_identity='a')
        self.assertEqual(result[0]['weight'], 0.)
        self.assertEqual(report['deactivated_weighted_rows'], [a])
        result, report = compact(g.rows, active_rows=[root], protected_rows=[a], audit_identity='a')
        self.assertEqual(result[0]['weight'], 7.)

    def test_invalid_schema_and_references_fail_closed(self):
        for rows in [[dict(operand='ZPLM')], [dict(operand='DIFF', params=[1, 1])],
                     [dict(operand='CONS', target=math.nan)], [dict(operand='CONS', weight=-1.)],
                     [dict(operand='CONF', params=[1], weight=1.)]]:
            with self.assertRaises(ValueError): normalize(rows)
        g = Graph(); a = g.add('CONS'); g.add('PROB', a, 2, 1.)
        with self.assertRaises(ValueError): normalize(g.rows)
        g = Graph(); a = g.add('CONS', weight=1.)
        with self.assertRaises(ValueError): compact(g.rows, active_rows=[a])
        with self.assertRaises(ValueError): compact(g.rows, active_rows=[True], audit_identity='a')

    def test_budget_and_native_roundtrip_format(self):
        g = Graph(); g.add('RWCE', 20, 1, 0., 1.)
        report = audit(g.rows, {'rows': 0}, evaluation_seconds=.1)
        self.assertFalse(report['budget_passed'])
        self.assertEqual(report['counts']['wavefront_queries'], 1)
        with self.assertRaises(ValueError): audit(g.rows, {'unknown': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.MF'; write_native(path, g.rows)
            lines = path.read_text(encoding='utf-16').splitlines()
            self.assertEqual(lines[0], 'VERS 221026')
            self.assertEqual(len(lines[1].split()), 12)

    def test_arithmetic_limit_semantics_and_invalid_domain(self):
        g = Graph(); a = g.add('CONS', target=2.); g.add('CONS', target=11.)
        b = g.add('CONS', target=3.); g.add('SUMM', a, b); g.add('PROD', a, b)
        g.add('DIVI', a, b); g.add('PROB', a, 1, .25); g.bound(a, 3., 1.)
        self.assertEqual(evaluate(g.rows, lambda *_: 0.)[4:], [5., 6., 2/3, .5, 2., 2.])
        bad = Graph(); a = bad.add('CONS', target=-1.); bad.add('SQRT', a)
        with self.assertRaises(ValueError): evaluate(bad.rows, lambda *_: 0.)

    def test_cli_over_budget_reports_without_emitting_native_mf(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); graph=p/'graph.json'; out=p/'compiled'
            graph.write_text(json.dumps([dict(operand='CONS',target=1.,weight=1.)]))
            with self.assertRaises(SystemExit): main(['--graph',str(graph),'--out',str(out),'--max-rows','0'])
            self.assertTrue((out/'report.json').is_file())
            self.assertFalse((out/'COMPACT.MF').exists())


def sphere_graph():
    g = Graph(); one = g.add('CONS', target=1.); distance = g.add('TTHI', 1, 1)
    curvatures = {s:g.add('CVVA', s, 1) for s in (1, 2)}
    geometry, domain = [], []
    faces = {1:1.3, 2:1.8}
    def sag(surface, radius):
        c = curvatures[surface]; square = g.add('PROD', c, c)
        cr2 = g.add('PROB', square, 1, radius*radius)
        d = g.add('DIFF', one, cr2)
        domain.append(dict(value_row=d, bound_rows=g.bound(d, .01, None), surface=surface,
                           evaluated_radius_mm=radius, lower=.01))
        sq = g.add('SQRT', d); denom = g.add('SUMM', one, sq)
        numerator = g.add('PROB', c, 1, radius*radius)
        return g.add('DIVI', numerator, denom)
    for r in [0., .2, .7, 1.2, 1.3, 1.4, 1.6, 1.8, 1.9, 2.]:
        sa, sb = min(r, faces[1]), min(r, faces[2])
        a, b = sag(1, sa), sag(2, sb)
        diff = g.add('DIFF', b, a); value = g.add('SUMM', distance, diff)
        geometry.append(dict(value_row=value, bound_rows=g.bound(value, .3, 2.),
            first_surface=1,last_surface=2, radius_first_mm=r, radius_last_mm=r,
            sag_radius_first_mm=sa,sag_radius_last_mm=sb,kind='gap',lower=.3,upper=2.))
    profiles = [dict(surface=s,sphere_face_radius_mm=r,outer_annulus='plane at sphere-face-edge sag')
                for s,r in faces.items()]
    return g.rows, geometry, domain, profiles


class SphereProofTests(unittest.TestCase):
    def test_random_curvatures_preserve_sampled_feasible_set(self):
        rows, geometry, domain, profiles = sphere_graph()
        support = spherical_support(rows, geometry, domain, profiles)
        self.assertLess(len(support['protected_rows']), len(support['source_rows']))
        self.assertFalse(support['weighted_penalty_preserved'])
        rng = random.Random(91); passed = failed = 0
        for _ in range(120):
            c = {1:rng.uniform(-.5,.5), 2:rng.uniform(-.5,.5)}
            distance = rng.uniform(-1.,3.)
            values = evaluate(rows, lambda row, ctx: c[int(row['params'][0])] if row['operand']=='CVVA' else distance)
            def okay(roots): return all(abs(values[i]-rows[i-1]['target']) < 1e-12 for i in roots)
            original, reduced = okay(support['source_rows']), okay(support['protected_rows'])
            self.assertEqual(original, reduced)
            passed += original; failed += not original
        self.assertGreater(passed,0); self.assertGreater(failed,0)

    def test_proof_rejects_changed_faces_targets_formula_or_domain(self):
        for change in ('face', 'target', 'formula', 'domain'):
            rows, geo, dom, profiles = sphere_graph()
            if change == 'face': profiles[0]['sphere_face_radius_mm'] = 1.1
            if change == 'target': rows[geo[0]['bound_rows'][0]-1]['target'] += .01
            if change == 'formula': rows[geo[0]['value_row']-1]['operand'] = 'PROD'
            if change == 'domain': dom = []
            with self.assertRaises(ValueError): spherical_support(rows, geo, dom, profiles)


if __name__ == '__main__': unittest.main()
