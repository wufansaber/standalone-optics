"""End-to-end thickness gates for curved pieces, in millimetres."""
import copy
import math
import unittest

from scripts.ai_zemax.standalone_manufacturing import audit_manufacturing
from scripts.ai_zemax.standalone_parallel_screen import screen_candidates


def lens(curvature=.1, thickness=1., conic=0.):
    return dict(vertices_mm=[10.,10.+thickness,80.],indices_by_wave=[[1.,1.5,1.,1.]],
                wavelengths_um=[.55],primary_wave=0,stop_surface=1,object_at_infinity=False,
                aperture_type='obna',aperture_value=.01,field_type='object_height',fields=[[0.,0.]],
                surface_rows=[dict(curvature_mm_inv=curvature,conic=conic,aperture_radius_mm=7.,
                                   mechanical_radius_mm=8.) for _ in range(2)]+[{}])


def policy():
    return dict(schema_version=1,sources=['synthetic 16 mm fixture'],
                lenses=[dict(front_surface=1,rear_surface=2)],
                defaults=dict(minimum_center_thickness_mm=.4,minimum_radial_thickness_mm=.4,
                              center_thickness_budget_mm=.01,radial_thickness_budget_mm=.01,
                              minimum_clear_to_mechanical_radial_margin_mm=.5,maximum_surface_slope=2.),
                meniscus=dict(minimum_center_to_diameter_ratio=.03,minimum_face_to_diameter_ratio=.05,
                              maximum_center_thickness_mm=4.,minimum_edge_thickness_mm=.5,
                              minimum_face_distance_mm=.5,face_distance_budget_mm=.01,edge_thickness_budget_mm=.01))


class MeniscusTests(unittest.TestCase):
    def test_axial_thickness_passes_while_oblique_glass_fails(self):
        for sign in (1.,-1.):
            model=lens(sign*.1);config=policy();result=audit_manufacturing(model,config)
            piece=result['lenses'][0]
            self.assertAlmostEqual(piece['center_thickness_mm'],1.)
            self.assertAlmostEqual(piece['minimum_radial_thickness_mm'],1.)
            self.assertAlmostEqual(piece['edge_thickness_mm'],1.)
            closest=piece['face_distance']
            self.assertAlmostEqual(closest['upper_mm'],math.sqrt(113.)-10.,places=12)
            self.assertNotEqual(closest['front_rz_mm'][0],closest['rear_rz_mm'][0])
            self.assertFalse(result['eligible'])
            self.assertEqual([i['code'] for i in result['violations']],['meniscus_face_distance_below_limit'])
            self.assertEqual(screen_candidates(model,[{'id':'thin'}],manufacturing_policy=config)['ranked'],[])

    def test_thickening_converges_to_budgeted_boundary(self):
        lo,hi=.9,2.
        for _ in range(28):
            mid=(lo+hi)/2
            if audit_manufacturing(lens(thickness=mid),policy())['passed']:
                hi=mid
            else:
                lo=mid
        expected=math.sqrt((10.+.81)**2-8.**2)-6.
        self.assertAlmostEqual(hi,expected,places=7)
        self.assertTrue(audit_manufacturing(lens(thickness=hi),policy())['passed'])
        self.assertFalse(audit_manufacturing(lens(thickness=lo),policy())['eligible'])

    def test_budget_ratios_edge_and_center_limit_are_independent(self):
        for key,value,code in [('minimum_center_to_diameter_ratio',.13,'meniscus_center_ratio_below_limit'),
                               ('minimum_face_to_diameter_ratio',.10,'meniscus_face_distance_below_limit'),
                               ('minimum_edge_thickness_mm',2.,'meniscus_edge_below_limit'),
                               ('maximum_center_thickness_mm',2.,'meniscus_center_above_limit')]:
            config=policy();config['meniscus'][key]=value
            report=audit_manufacturing(lens(thickness=2.),config)
            self.assertIn(code,[issue['code'] for issue in report['violations']])

    def test_domain_missing_od_and_missing_allocations_cannot_rank(self):
        for change in ('invalid_radius','missing_od','missing_limit','unknown_limit','no_policy'):
            config=policy();model=lens(thickness=2.)
            if change=='invalid_radius':
                for face in model['surface_rows'][:2]:
                    face['mechanical_radius_mm']=10.
            elif change=='missing_od':
                del model['surface_rows'][0]['mechanical_radius_mm']
            elif change=='missing_limit':
                del config['meniscus']['face_distance_budget_mm']
            elif change=='unknown_limit':
                config['meniscus']['unused']=.2
            else:
                config=None
            with self.subTest(change=change):
                self.assertFalse(audit_manufacturing(model,config)['eligible'])
                self.assertEqual(screen_candidates(model,[{'id':'sample'}],manufacturing_policy=config)['ranked'],[])

    def test_smaller_policy_od_cannot_hide_the_model_edge(self):
        config=policy()
        config['surfaces']={str(i):dict(mechanical_radius_mm=7.8) for i in (1,2)}
        report=audit_manufacturing(lens(thickness=2.),config)
        self.assertIn('meniscus_mechanical_radius_mismatch',[i['code'] for i in report['violations']])

    def test_explicit_outer_shoulder_is_not_a_full_diameter_sphere(self):
        model=lens(thickness=2.)
        model['explicit_nominal_solid_profiles']=[
            dict(surface=i,sphere_face_radius_mm=8.,body_radius_mm=9.) for i in (1,2)]
        report=audit_manufacturing(model,policy())
        self.assertEqual(report['status'],'incomplete')
        self.assertFalse(report['eligible'])
        self.assertIn('meniscus_profile_requires_full_body_distance',[i['code'] for i in report['missing']])
        self.assertNotIn('face_distance',report['lenses'][0])

    def test_aspheric_enclosure_covers_an_analytic_parabola(self):
        model=lens(conic=-1.);config=policy();p=config['meniscus']
        # Translate two identical paraboloids. Closest point projects front rim
        # onto the second curve; solve stationarity independently as a cubic.
        import numpy as np
        roots=np.roots([.005,0.,.78,-8.])
        r=next(float(v.real) for v in roots if abs(v.imag)<1e-12)
        exact=math.hypot(8-r,3.2-(1.+.05*r*r))
        for delta in (-.00005,.00005):
            p['minimum_face_to_diameter_ratio']=.02
            p['minimum_face_distance_mm']=exact+delta-p['face_distance_budget_mm']
            result=audit_manufacturing(model,config)
            interval=result['lenses'][0]['face_distance']
            self.assertLessEqual(interval['lower_mm'],exact+1e-12)
            self.assertGreaterEqual(interval['upper_mm'],exact-1e-12)
            self.assertEqual(result['passed'],delta<0)
            self.assertGreater(interval['segments'],16)

    def test_asphere_shape_uses_actual_power_and_plane_is_exempt(self):
        model=lens(thickness=2.)
        model['surface_rows'][0].update(curvature_mm_inv=-.01,even_coefficients=[.025])
        self.assertEqual(audit_manufacturing(model,policy())['lenses'][0]['shape'],'meniscus')
        model['surface_rows'][0].update(curvature_mm_inv=0.,even_coefficients=[])
        config=copy.deepcopy(policy());del config['meniscus']
        self.assertTrue(audit_manufacturing(model,config)['passed'])
