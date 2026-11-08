"""Regression cases for nominal physical solids and configurable process gates."""
from copy import deepcopy
import json
import unittest

from scripts.ai_zemax.standalone_manufacturing import audit_manufacturing


def singlet():
    return {
        'vertices_mm': [10., 12., 100.],
        'indices_by_wave': [[1., 1.5, 1., 1.]],
        'surface_rows': [
            {'curvature_mm_inv': .02, 'aperture_radius_mm': 1., 'mechanical_radius_mm': 1.2},
            {'curvature_mm_inv': -.02, 'aperture_radius_mm': 1., 'mechanical_radius_mm': 1.2},
            {},
        ],
    }


def policy():
    return {
        'schema_version': 1,
        'defaults': {
            'minimum_center_thickness_mm': 1.,
            'minimum_radial_thickness_mm': .5,
            'minimum_clear_to_mechanical_radial_margin_mm': .1,
            'minimum_air_gap_mm': .1,
            'minimum_rim_gap_mm': .1,
        },
        'lenses': [{'front_surface': 1, 'rear_surface': 2}],
    }


class ManufacturingTests(unittest.TestCase):
    def codes(self, report):
        json.dumps(report, allow_nan=False)
        return {row['code'] for row in report['violations']}

    def test_declared_singlet_and_missing_envelope(self):
        model = singlet()
        report = audit_manufacturing(model, policy())
        self.assertTrue(report['passed'])
        self.assertTrue(report['eligible'])
        self.assertGreater(report['lenses'][0]['minimum_radial_thickness_mm'], 1.9)
        model['surface_rows'][0].pop('mechanical_radius_mm')
        report = audit_manufacturing(model, policy())
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['eligible'])
        self.assertFalse(report['passed'])

    def test_defaults_do_not_certify_pure_index_model(self):
        report = audit_manufacturing(singlet())
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['eligible'])
        self.assertIn('explicit_lens_topology_missing', {r['code'] for r in report['missing']})
        model = {'vertices_mm': [10., 20.], 'indices_by_wave': [[1., 1.5, 1.5]],
                 'surface_rows': [{}, {}]}
        self.assertTrue(audit_manufacturing(model)['eligible'])

    def test_interior_asphere_thinner_than_center_and_edge(self):
        # t(r) = 2 - 8 r² + 8 r^4. Endpoints are 2 but t(sqrt(.5)) = 0.
        model = singlet()
        model['surface_rows'][0]['curvature_mm_inv'] = 0.
        model['surface_rows'][1].update(curvature_mm_inv=0., even_coefficients=[-8., 8.])
        for row in model['surface_rows'][:2]:
            row['mechanical_radius_mm'] = 1.
            row['aperture_radius_mm'] = .8
        report = audit_manufacturing(model, policy())
        self.assertIn('nonpositive_radial_thickness', self.codes(report))
        self.assertAlmostEqual(report['lenses'][0]['minimum_at_radius_mm'], .5**.5, places=7)
        self.assertEqual(report['lenses'][0]['edge_thickness_mm'], 2.)

    def test_full_mechanical_domain_exceeds_valid_clear_domain(self):
        model = singlet()
        model['surface_rows'][0].update(curvature_mm_inv=1., aperture_radius_mm=.5,
                                       mechanical_radius_mm=1.2)
        report = audit_manufacturing(model)
        self.assertIn('surface_domain_or_singular_slope', self.codes(report))
        self.assertFalse(report['eligible'])

    def test_positive_center_negative_edge_and_sequential_vertices(self):
        model = singlet()
        model['surface_rows'][0]['curvature_mm_inv'] = .7
        model['surface_rows'][1]['curvature_mm_inv'] = -.7
        model['vertices_mm'][1] = 10.2
        self.assertIn('nonpositive_radial_thickness', self.codes(audit_manufacturing(model)))
        model['vertices_mm'][1] = 9.
        self.assertIn('invalid_input', self.codes(audit_manufacturing(model)))

    def test_mechanical_rim_gap_uses_each_lens_own_radius(self):
        model = {
            'vertices_mm': [10., 12., 12.3, 14.3, 100.],
            'indices_by_wave': [[1., 1.5, 1., 1.5, 1., 1.]],
            'surface_rows': [
                {'mechanical_radius_mm': 2., 'aperture_radius_mm': 1.8},
                {'curvature_mm_inv': .25, 'mechanical_radius_mm': 2., 'aperture_radius_mm': 1.8},
                {'curvature_mm_inv': .25, 'mechanical_radius_mm': 1., 'aperture_radius_mm': .8},
                {'mechanical_radius_mm': 1., 'aperture_radius_mm': .8}, {},
            ],
        }
        p = policy()
        p['lenses'].append({'front_surface': 3, 'rear_surface': 4})
        report = audit_manufacturing(model, p)
        self.assertIn('rim_gap_below_limit', self.codes(report))
        gap = report['gaps'][0]
        self.assertAlmostEqual(gap['minimum_gap_mm'], .3)
        self.assertLess(gap['separate_rim_axial_gap_mm'], 0.)

    def test_nonadjacent_solids_checked_outside_middle_radius(self):
        model = {
            'vertices_mm': [10., 12., 12.2, 12.4, 12.6, 15., 100.],
            'indices_by_wave': [[1., 1.5, 1., 1.5, 1., 1.5, 1., 1.]],
            'surface_rows': [
                {'mechanical_radius_mm': 3., 'aperture_radius_mm': 2.8},
                {'curvature_mm_inv': .2, 'mechanical_radius_mm': 3., 'aperture_radius_mm': 2.8},
                {'mechanical_radius_mm': .5, 'aperture_radius_mm': .3},
                {'mechanical_radius_mm': .5, 'aperture_radius_mm': .3},
                {'mechanical_radius_mm': 3., 'aperture_radius_mm': 2.8},
                {'mechanical_radius_mm': 3., 'aperture_radius_mm': 2.8}, {},
            ],
        }
        p = policy()
        p['defaults']['minimum_center_thickness_mm'] = .1
        p['defaults']['minimum_radial_thickness_mm'] = .1
        p['lenses'] = [{'front_surface': 1, 'rear_surface': 2},
                       {'front_surface': 3, 'rear_surface': 4},
                       {'front_surface': 5, 'rear_surface': 6}]
        report = audit_manufacturing(model, p)
        failures = [r for r in report['violations'] if r['code'] == 'solid_intersection']
        self.assertTrue(any(r['left_lens'] == 1 and r['right_lens'] == 3 for r in failures))

    def test_cemented_and_index_matched_declared_glass(self):
        model = {'vertices_mm': [10., 12., 14., 100.],
                 'indices_by_wave': [[1., 1.5, 1.5, 1., 1.]],
                 'surface_rows': [{'mechanical_radius_mm': 1.2, 'aperture_radius_mm': 1.} for _ in range(3)]+[{}]}
        p = policy()
        p['lenses'] = [{'front_surface': 1, 'rear_surface': 2}, {'front_surface': 2, 'rear_surface': 3}]
        report = audit_manufacturing(model, p)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['lenses']), 2)
        self.assertEqual(report['gaps'][0]['kind'], 'shared_cemented_interface')

    def test_declarations_cannot_omit_or_merge_physical_pieces(self):
        model = {'vertices_mm': [10., 12., 15., 17., 100.],
                 'indices_by_wave': [[1., 1.5, 1., 1.6, 1., 1.]],
                 'surface_rows': [{'mechanical_radius_mm': 1.2, 'aperture_radius_mm': 1.} for _ in range(4)]+[{}]}
        p = policy()
        self.assertIn('invalid_topology', self.codes(audit_manufacturing(model, p)))
        p['lenses'] = [{'front_surface': 1, 'rear_surface': 4}]
        self.assertIn('invalid_topology', self.codes(audit_manufacturing(model, p)))
        model['material_names'] = ['', 'G1', '', 'G2', '', '']
        self.assertIn('invalid_topology', self.codes(audit_manufacturing(model, p)))
        model = {'vertices_mm': [10., 12., 14., 100.],
                 'indices_by_wave': [[1., 1.5, 1.6, 1., 1.]],
                 'surface_rows': [{}, {}, {}, {}]}
        p['lenses'] = [{'front_surface': 1, 'rear_surface': 3}]
        self.assertIn('invalid_topology', self.codes(audit_manufacturing(model, p)))

    def test_configured_budgets_and_sphere_only_rule(self):
        p = policy()
        p['defaults']['minimum_center_thickness_mm'] = 1.9
        p['defaults']['center_thickness_budget_mm'] = .11
        report = audit_manufacturing(singlet(), p)
        self.assertIn('center_thickness_below_limit', self.codes(report))
        model = singlet()
        model['surface_rows'][0]['even_coefficients'] = [0., 1e-3]
        p = policy()
        p['all_spherical'] = True
        self.assertIn('asphere_not_allowed', self.codes(audit_manufacturing(model, p)))

    def test_policy_metadata_and_invalid_values(self):
        p = policy()
        p.update(policy_id='example-only', sources=['project'], notes='Configured values')
        self.assertTrue(audit_manufacturing(singlet(), p)['passed'])
        for bad in (True, float('nan'), -1.):
            p = policy()
            p['defaults']['minimum_center_thickness_mm'] = bad
            self.assertIn('invalid_input', self.codes(audit_manufacturing(singlet(), p)))
        self.assertIn('invalid_input', self.codes(audit_manufacturing(singlet(), {'defalts': {}})))
        self.assertIn('invalid_input', self.codes(audit_manufacturing(singlet(), 3)))


if __name__ == '__main__':
    unittest.main()
