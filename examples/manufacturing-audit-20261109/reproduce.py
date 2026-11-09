"""Executable review probes against the unchanged installed manufacturing module."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SRC=HERE.parents[1]/'scripts'
sys.path.insert(0,str(SRC))
from ai_zemax.standalone_manufacturing import audit_manufacturing

def model():
    return dict(vertices_mm=[10.,12.,100.],indices_by_wave=[[1.,1.5,1.,1.]],
                surface_rows=[dict(curvature_mm_inv=.02,aperture_radius_mm=1.,mechanical_radius_mm=1.2),
                              dict(curvature_mm_inv=-.02,aperture_radius_mm=1.,mechanical_radius_mm=1.2),{}])

def policy():
    return dict(schema_version=1,defaults=dict(minimum_center_thickness_mm=1.,
                minimum_radial_thickness_mm=.5,minimum_clear_to_mechanical_radial_margin_mm=.1,
                minimum_air_gap_mm=.1,minimum_rim_gap_mm=.1),
                lenses=[dict(front_surface=1,rear_surface=2)])

def main():
    results=[]
    def add(name,m,p,expectation):
        report=audit_manufacturing(m,p)
        results.append(dict(name=name,expected=expectation,input_model=m,input_policy=p,actual=report))
        print(name,report['status'],[r['code'] for r in report['violations']],flush=True)
    add('baseline',model(),policy(),'nominal geometry passes')
    p=policy();p['lenses'][0]['maximum_surface_slope']=.001
    add('per_lens_slope_ignored',model(),p,'reject or reject unsupported key; actual slope approximately .024 > .001')
    p=policy();p['defaults']['maximum_surface_slope']=.001
    add('default_slope_control',model(),p,'reject')
    p=policy();p['lenses'][0]['minimum_clear_to_mechanical_radial_margin_mm']=.5
    add('per_lens_radial_margin_ignored',model(),p,'reject or reject unsupported key; actual margin .2 < .5')
    p=policy();p['defaults']['minimum_clear_to_mechanical_radial_margin_mm']=.5
    add('default_radial_margin_control',model(),p,'reject')
    pair=dict(vertices_mm=[10.,12.,12.2,14.2,100.],indices_by_wave=[[1.,1.5,1.,1.5,1.,1.]],
              surface_rows=[dict(aperture_radius_mm=1.,mechanical_radius_mm=1.2) for _ in range(4)]+[{}])
    p=policy();p['lenses'].append(dict(front_surface=3,rear_surface=4))
    p['lenses'][0]['minimum_air_gap_mm']=.5
    add('per_lens_air_gap_ignored',pair,p,'reject or reject unsupported key; actual gap .2 < .5')
    p=policy();p['lenses'].append(dict(front_surface=3,rear_surface=4))
    p['lenses'][0]['air_gap_budget_mm']=.2
    add('per_lens_air_budget_ignored',pair,p,'reject or reject unsupported key; .2-.2 < .1')
    p=policy();p['lenses'].append(dict(front_surface=3,rear_surface=4));p['defaults']['minimum_air_gap_mm']=.5
    add('default_air_gap_control',pair,p,'reject')
    p=policy();p['lenses'][0]['center_thickness_budget_mm']=1.5
    add('per_lens_center_budget_control',model(),p,'reject')
    m=model();m.update(fields=[[0.,0.],[0.,100.]],field_type='object_height',aperture_type='obna',aperture_value=.6)
    add('ray_envelope_is_out_of_scope',m,policy(),'nominal geometry may pass; actual ray coverage has not been checked')
    m=model();m['vertices_mm'][1]=210.;m['vertices_mm'][2]=300.
    add('thick_high_aspect_ratio_lens_unbounded',m,policy(),'currently no maximum CT, overall height, aspect ratio or tool reach policy')
    for field in ('minimum_rim_gap_mm','rim_gap_budget_mm'):
        p=policy();p['lenses'].append(dict(front_surface=3,rear_surface=4));p['lenses'][0][field]=.5
        add('per_lens_'+field+'_ignored',pair,p,'reject or reject unsupported key')
    from ai_zemax.standalone_system import OpticalSystem
    from ai_zemax.standalone import trace,directions_from_slopes
    import numpy as np
    m=model()
    for row in m['surface_rows'][:2]:row['aperture_radius_mm']=2.
    m.update(fields=[[0.,0.],[0.,.5]],field_type='object_height',aperture_type='obna',aperture_value=.1,
             wavelengths_um=[.55],stop_surface=1,primary_wave=0,object_at_infinity=False,
             object_space_telecentric=True)
    p=policy();p['surfaces']={str(i):dict(clear_radius_mm=1.,mechanical_radius_mm=1.2) for i in (1,2)}
    add('policy_clear_radius_does_not_check_trace_envelope',m,p,'nominal geometry passes, but model trace can exit physical lens')
    system=OpticalSystem(m)
    point,slope=system.launch_field([0.,.5])
    ray=trace(system.surfaces[0],np.r_[point,0.],directions_from_slopes(slope+system.pupil_scale*np.array([[0.,1.]])),record_path=True)
    results[-1]['ray_probe']=dict(valid=bool(ray.valid[0]),
                                  first_face_hit_radius_mm=float(np.linalg.norm(ray.surface_points[0,0,:2])),
                                  clear_radius_mm=1.,mechanical_radius_mm=1.2)
    from ai_zemax.standalone_parallel_screen import screen_candidates
    results[-1]['screening_probe']=screen_candidates(m,[dict(id='outside_physical_envelope')],
                                                     radial=8,azimuthal=32,manufacturing_policy=p)
    result=dict(module='scripts/ai_zemax/standalone_manufacturing.py',
                module_sha256=hashlib.sha256((SRC/'ai_zemax/standalone_manufacturing.py').read_bytes()).hexdigest(),
                audited_commit='c9a46f168f6750b7187e4167405c2fd08386b30b',probes=results)
    # Historical audit probes are intentionally outside CI. A future fix may change these outcomes.
    for row in results:
        if row['name'].endswith('_ignored'):
            row['finding_reproduced']=row['actual']['passed']
        if 'ray_probe' in row:
            ray=row['ray_probe']
            row['finding_reproduced']=bool(ray['valid'] and ray['first_face_hit_radius_mm']>ray['mechanical_radius_mm']
                                           and row['screening_probe']['ranked'])
    (HERE/'observed.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
