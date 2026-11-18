import unittest
from scripts.ai_zemax.standalone_manufacturing import audit_manufacturing


class SpacerSeatTests(unittest.TestCase):
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
