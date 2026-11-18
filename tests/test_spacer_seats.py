import unittest
from scripts.ai_zemax.standalone_manufacturing import audit_manufacturing


class SpacerSeatTests(unittest.TestCase):
    def flat_case(self):
        import math
        model,config=self.case(20.)
        config['defaults']['maximum_surface_slope']=3.
        lands={}
        for i,row in enumerate(model['surface_rows'][:4]):
            c=row['curvature_mm_inv'];z=model['vertices_mm'][i]
            endpoints=[z+c*r*r/(1+math.sqrt(1-(c*r)**2)) for r in (5.,8.)]
            plane=max(endpoints)+.05 if i%2==0 else min(endpoints)-.05
            lands[str(i+1)]=dict(blend_start_mm=5.,flat_start_mm=5.8,bevel_start_mm=7.5,plane_z_mm=plane)
        config['extended_edges']=dict(tube_lenses=[1,2],common_diameter_mm=16.,maximum_edge_mm=10.,
                                      thickness_budget_mm=.05,minimum_platform_mm=1.5,clear_reserve_mm=.25,lands=lands)
        config['spacer_seats']['interfaces'][0]['kind']='flat_annulus'
        return model,config

    def test_transition_slope_is_checked_and_tangent_ramp_can_pass(self):
        model,config=self.flat_case()
        config['defaults']['maximum_surface_slope']=1.8
        result=audit_manufacturing(model,config)
        self.assertIn('finished_profile_slope_unqualified',[v['code'] for v in result['violations']])
        for key in ('2','3'):
            config['extended_edges']['lands'][key]['transition']='tangent_ramp'
        result=audit_manufacturing(model,config)
        self.assertTrue(result['passed'],result['violations'])

    def test_flat_spacer_requires_two_ground_glass_platforms(self):
        model,config=self.flat_case()
        result=audit_manufacturing(model,config)
        self.assertTrue(result['passed'],result['violations'])
        self.assertEqual(result['spacers'][0]['inner_gap_mm'],result['spacers'][0]['outer_gap_mm'])
        del config['extended_edges']['lands']['3']
        self.assertFalse(audit_manufacturing(model,config)['passed'])

    def test_small_drawing_diameter_cannot_hide_extended_negative_edge(self):
        model,config=self.flat_case()
        model['surface_rows'][0].update(curvature_mm_inv=-.10,mechanical_radius_mm=5.)
        model['surface_rows'][1].update(curvature_mm_inv=.10,mechanical_radius_mm=5.)
        result=audit_manufacturing(model,config)
        self.assertAlmostEqual(result['extended_edges'][0]['extended_edge_mm'],12.)
        codes={v['code'] for v in result['violations']}
        self.assertTrue({'extended_edge_thickness','inconsistent_finished_diameter'}<=codes)
        self.assertFalse(result['eligible'])

    def test_flat_plane_must_be_cut_from_parent_cap_stock(self):
        model,config=self.flat_case()
        config['extended_edges']['lands']['1']['plane_z_mm']=-3.
        result=audit_manufacturing(model,config)
        self.assertIn('land_outside_stock',[v['code'] for v in result['violations']])

    def test_finished_bevel_removes_required_glass(self):
        model,config=self.flat_case()
        config['extended_edges']['lands']['1']['plane_z_mm']=2.1
        config['extended_edges']['lands']['2']['plane_z_mm']=3.2
        result=audit_manufacturing(model,config)
        self.assertIn('finished_glass_thickness_uncertified',[v['code'] for v in result['violations']])

    def test_policy_cannot_extend_a_smaller_prescription_by_declaration(self):
        model,config=self.flat_case()
        model['surface_rows'][0]['mechanical_radius_mm']=6.
        config['surfaces']={'1':{'mechanical_radius_mm':8.}}
        result=audit_manufacturing(model,config)
        self.assertIn('extended_edge_model_radius_mismatch',[v['code'] for v in result['violations']])

    def test_ground_asphere_requires_stock_at_interior_extrema(self):
        model,config=self.flat_case()
        model['surface_rows'][0]['even_coefficients']=[.00089,-.00001]
        config['extended_edges']['lands']['1']['plane_z_mm']=1.017
        result=audit_manufacturing(model,config)
        self.assertIn('land_outside_stock',[v['code'] for v in result['violations']])

    def test_extra_group_and_one_sided_end_lens(self):
        model,config=self.flat_case()
        rules=config['extended_edges']
        rules['tube_lenses']=[1]
        del rules['lands']['4']
        result=audit_manufacturing(model,config)
        self.assertTrue(result['passed'],result['violations'])
        self.assertIn('finished_rim_mm',result['extended_edges'][1])
        rules['additional_common_groups']=[dict(lenses=[2],diameter_mm=16.)]
        self.assertFalse(audit_manufacturing(model,config)['passed'])
        rules['lands']['4']=self.flat_case()[1]['extended_edges']['lands']['4']
        self.assertTrue(audit_manufacturing(model,config)['passed'])
        rules['additional_common_groups'][0]['lenses']=[1]
        self.assertFalse(audit_manufacturing(model,config)['eligible'])

    def test_malformed_land_is_a_structured_failure(self):
        model,config=self.flat_case()
        config['extended_edges']['lands']['3']=None
        result=audit_manufacturing(model,config)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['passed'])

    def case(self,separation=29.):
        model=dict(vertices_mm=[1.,5.,5.+separation,10.+separation,85.],
                   indices_by_wave=[[1.,1.5,1.,1.5,1.,1.]],wavelengths_um=[.55],primary_wave=0,
                   surface_rows=[dict(curvature_mm_inv=c,aperture_radius_mm=4.,mechanical_radius_mm=8.) for c in [0.,-.05,.05,0.]]+[{}])
        config=dict(schema_version=1,lenses=[dict(front_surface=1,rear_surface=2),dict(front_surface=3,rear_surface=4)],
                    defaults=dict(minimum_center_thickness_mm=1.,minimum_radial_thickness_mm=1.,minimum_air_gap_mm=.3,
                                  minimum_rim_gap_mm=.3,minimum_clear_to_mechanical_radial_margin_mm=.5,
                                  maximum_surface_slope=1.,center_thickness_budget_mm=.05,radial_thickness_budget_mm=.05,
                                  air_gap_budget_mm=.03,rim_gap_budget_mm=.03),
                    spacer_seats=dict(interfaces=[dict(left_lens=1,kind='conformal_annulus',inner_radius_mm=6.,outer_radius_mm=7.)],
                                      minimum_width_mm=1.,chip_mm=.25,bevel_mm=.30,minimum_gap_mm=.3,maximum_gap_mm=30.,gap_budget_mm=.03))
        return model,config

    def test_axis_passes_but_actual_seat_exceeds_maximum(self):
        model,config=self.case()
        result=audit_manufacturing(model,config)
        self.assertGreater(result['spacers'][0]['maximum_gap_upper_mm'],30.)
        self.assertIn('spacer_gap_out_of_range',[i['code'] for i in result['violations']])
        self.assertFalse(result['eligible'])

    def test_valid_contact_and_missing_contact(self):
        model,config=self.case(separation=20.)
        self.assertTrue(audit_manufacturing(model,config)['passed'])
        config['spacer_seats']['interfaces'][0]['outer_radius_mm']=8.
        self.assertFalse(audit_manufacturing(model,config)['passed'])
        config['spacer_seats']['interfaces']=[]
        self.assertFalse(audit_manufacturing(model,config)['passed'])
