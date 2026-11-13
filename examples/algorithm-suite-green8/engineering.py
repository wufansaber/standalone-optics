"""Report the actual finite-face/shoulder body allocations of a case candidate."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
CASE=Path(__file__).resolve().parent;sys.path[:0]=[str(CASE),str(CASE.parents[1])]
from problem import Design


def review(model):
    design=Design(base=model);g,j=design.geometry(model);pieces=[]
    for first,last in design.geo.solids:
        rows=[r for r in design.geo.last_rows if r['kind']=='glass_full_OD'and r['first_surface']==first+1 and r['last_surface']==last+1]
        worst=min(rows,key=lambda r:r['physical_margin_mm'])
        pieces.append(dict(front_surface=first+1,rear_surface=last+1,material=model['material_names'][first+1],
            group='fixed_window'if first==0 else'tube'if first>=20 else'objective',
            finished_diameter_mm=2*min(design.geo.body[first],design.geo.body[last]),
            center_thickness_mm=model['vertices_mm'][last]-model['vertices_mm'][first],
            minimum_fullbody_mm=worst['clearance_mm'],minimum_radius_mm=worst['radius_first_mm'],
            nominal_required_fullbody_mm=worst['floor_mm'],nominal_margin_mm=worst['physical_margin_mm'],
            provisional_single_error_thickness_budget_mm=0. if first==0 else .03))
    kinds={r['kind']for r in design.geo.last_rows};minima={k:min(r['physical_margin_mm']for r in design.geo.last_rows if r['kind']==k)for k in kinds}
    return dict(nominal_geometry_passed=bool(np.min(g)>=0.),scope='Fixed spherical face cutoff with planar shoulders, coaxial axial solids',
        pieces=pieces,minimum_margin_mm_by_kind=minima,full_optical_acceptance_performed=False,
        manufacturing_release=False,statistical_yield_estimated=False,
        outstanding=['supplier capability and glass procurement','simultaneous errors and justified compensators',
            'surface figure, roughness, wedge and centration','mounts, bevels, retaining hardware, stress and thermal assembly'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Use a new report path')
    raw=args.model.read_bytes();r=review(json.loads(raw));r['input_sha256']=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:r[k]for k in ('nominal_geometry_passed','manufacturing_release')}))

if __name__=='__main__':main()
