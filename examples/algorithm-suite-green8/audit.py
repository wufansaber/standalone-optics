"""Frozen-domain independent acceptance: alternate pupils and local extrema."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
from scipy.optimize import minimize_scalar
import mpmath as mp

from problem import (BASE,SPEC,CAPS,ORIGINAL_WAVES,Design,configured,component,
    NativeReferenceSystem,disk_quadrature,reference_opd_mm,wavefront_rms,paraxial_transfer,best_focus)
from scripts.ai_zemax.standalone_system import rectangular_pupil


def wave_audit(task):
    model,w,field_count,local_search=task;m=configured(model,w);system=NativeReferenceSystem(m);wi=m['wavelengths_um'].index(w)
    nodes,weights=disk_quadrature(12,48);rect,rw=rectangular_pupil(16);a=np.arange(128)*2*np.pi/128
    pupil=np.r_[nodes,rect,np.c_[np.cos(a),np.sin(a)],[[0.,0.]]]
    splits=[(slice(0,len(nodes)),weights),(slice(len(nodes),len(nodes)+len(rect)),rw)]
    fields=np.linspace(0.,1.25,field_count)if np.isscalar(field_count)else np.array(field_count,dtype=float);cache={};foot=np.zeros(26);minimum_path=math.inf;ray_count=0
    matrix=paraxial_transfer(system.surfaces[wi],0.,1.);mag=float(matrix[0,0]-matrix[0,1]/1e10)
    def evaluate(h):
        nonlocal foot,minimum_path,ray_count
        h=float(h)
        if h in cache:return cache[h]
        rays=system._shoot(wi,pupil,*system.launch_field([0.,h]),record_path=True);ray_count+=len(pupil)
        if not rays.valid.all()or np.any(rays.negative_segment_count):raise ValueError('Dense physical ray failure; no filtering allowed')
        chief=replace(rays,points=rays.points[-1:],directions=rays.directions[-1:],opl_mm=rays.opl_mm[-1:],valid=rays.valid[-1:],failed_surface=rays.failed_surface[-1:],reason=rays.reason[-1:],surface_points=None)
        opd=reference_opd_mm(rays,chief,system.reference_distance_mm/chief.directions[0,2],1.)
        rms=[float(wavefront_rms(opd[sl],weight,w,rays.directions[sl,:2])[0])for sl,weight in splits]
        field_foot=np.linalg.norm(rays.surface_points[:26,:,:2],axis=-1).max(axis=1);foot=np.maximum(foot,field_foot)
        path=float(np.min(np.diff(rays.surface_points[1:,:,2],axis=0)));minimum_path=min(minimum_path,path)
        margins=[]
        for i,radius in enumerate(field_foot):
            data=m['surface_rows'][i];reserve=0. if i<2 else .08 if i<20 else .25;stock=0. if i<2 else .20
            margins.extend([data['clear_radius_mm']-radius-reserve,data['mechanical_radius_mm']-radius-stock])
        value=dict(field_mm=h,RMS_waves=max(rms),GQ12x48=rms[0],RECT16=rms[1],chief_y_mm=float(chief.points[0,1]),
            angle_deg=float(np.degrees(np.arcsin(np.linalg.norm(chief.directions[0,:2])))),
            physical_slack_mm=float(min(min(margins),path-.01)),
            distortion_percent=float(100*(chief.points[0,1]/mag/h-1))if h else 0.)
        cache[h]=value;return value
    rows=[evaluate(h)for h in fields];values=[r['RMS_waves']for r in rows];locals=[]
    if local_search:
        for i in range(1,len(fields)-1):
            if values[i]>=values[i-1]and values[i]>=values[i+1]and(values[i]>values[i-1]or values[i]>values[i+1]):
                fit=minimize_scalar(lambda h:-evaluate(h)['RMS_waves'],bounds=(fields[i-1],fields[i+1]),method='bounded',options={'xatol':1e-8})
                locals.append(dict(evaluate(fit.x),success=bool(fit.success)))
    reserves=[]
    for i,radius in enumerate(foot):
        row=m['surface_rows'][i];reserve=0. if i<2 else .08 if i<20 else .25;stock=0. if i<2 else .20
        reserves.extend([row['clear_radius_mm']-radius-reserve,row['mechanical_radius_mm']-radius-stock])
    cap=CAPS.get(w);worst=max(cache.values(),key=lambda r:r['RMS_waves'])
    return dict(wavelength_um=w,cap=cap,rows=rows,local_extrema=locals,worst=worst,
        normalized=worst['RMS_waves']/cap if cap else None,mag=mag,
        minimum_CA_face_reserve_mm=min(reserves),minimum_forward_path_mm=minimum_path,rays=ray_count,
        physical_passed=bool(min(reserves)>=0. and minimum_path>.01),
        valid=True,all_local_searches_succeeded=all(r['success']for r in locals))


def focus_audit(task):
    model,w,fields=task;m=deepcopy(model);m['primary_wave']=m['wavelengths_um'].index(.536);system=NativeReferenceSystem(m);wi=m['wavelengths_um'].index(w)
    nodes,weights=disk_quadrature(20,64);weights=np.asarray(weights)/sum(weights);a=np.arange(128)*2*np.pi/128;pupil=np.r_[nodes,np.c_[np.cos(a),np.sin(a)]]
    matrix=paraxial_transfer(system.surfaces[wi],0.,1.);mag=float(matrix[0,0]-matrix[0,1]/1e10);result=[]
    for h in fields:
        rays=system._shoot(wi,pupil,*system.launch_field([0.,float(h)]))
        if not rays.valid.all()or np.any(rays.negative_segment_count):raise ValueError('Dense focus ray failure')
        f=best_focus(rays.points[:len(nodes),:2],rays.directions[:len(nodes)],weights,tangential_axis=(0.,1.))
        result.append(dict(wavelength_um=w,field_mm=float(h),full=float(f['radial_rms_shift']/mag**2),mean=float(f['mean_shift']/mag**2)))
    return result


def exact_geometry(model,*,perturbed=False):
    d=Design(base=model);g,j=d.geometry(model,perturbed=perturbed);mp.mp.dps=80
    D=lambda x:mp.mpf(str(float(x)))
    margins=[]
    for row in d.geo.last_rows:
        a,b=row['first_surface']-1,row['last_surface']-1
        def surface(i,r):
            c=D(model['surface_rows'][i]['curvature_mm_inv']);r=D(min(r,d.geo.face[i]));root=mp.sqrt(1-c*c*r*r)
            return c*r*r/(1+root)
        separation=D(model['vertices_mm'][b])-D(model['vertices_mm'][a])+surface(b,row['radius_last_mm'])-surface(a,row['radius_first_mm'])
        margin=D(row['nominal_optical_solid_span_limit_mm'])-separation if row['kind']=='maximum_complete_tube_solid_span'else separation-D(row['floor_mm'])
        margins.append(margin)
    domains=[1-D(r['curvature_mm_inv'])**2*D(d.geo.face[i])**2-D(.1)**2 for i,r in enumerate(model['surface_rows'])if d.geo.face[i]>0.]
    return dict(passed=bool(min(margins)>=0. and min(domains)>=0. and np.min(g)>=0.),
        minimum_margin_mm=float(min(margins)),minimum_domain_slack=float(min(domains)),precision_digits=80,
        full_spherical_shoulder_domain=True,manufacturing_release=False)


def full_audit(model,*,executor=None,fields=65,local_search=True,include_focus=True,perturbed=False):
    tasks=[(model,w,fields,local_search)for w in ORIGINAL_WAVES]
    try:rows=list(executor.map(wave_audit,tasks))if executor else[wave_audit(t)for t in tasks]
    except (ValueError,FloatingPointError,np.linalg.LinAlgError)as error:return dict(passed=False,error=str(error),invalid_ray_failure=True)
    by={r['wavelength_um']:r for r in rows};quality=max(by[w]['normalized']for w in CAPS)
    physical=all(r['physical_passed']for r in rows);body=exact_geometry(model,perturbed=perturbed)
    fo=[];normalized_fo=[]
    for w in CAPS:
        r=by[w];values=[.01-abs(r['mag']+8.9),.07-max(x['angle_deg']for x in r['rows']),.6-max(abs(x['distortion_percent'])for x in r['rows'])];fo.extend(values);normalized_fo.extend(np.array(values)/[.01,.07,.6])
    reference=np.array([r['chief_y_mm']for r in by[.536]['rows']])
    for w in SPEC['hard_green_lateral_color']['reference_measured_waves_um']:
        value=.0018974173564139107-max(abs(np.array([r['chief_y_mm']for r in by[w]['rows']])-reference));fo.append(value);normalized_fo.append(value/.0018974173564139107)
    wi=model['wavelengths_um'].index(.595);ports=SPEC['ports_mm'];zeros=np.zeros((27,0))
    for first,last,entry,exit,nom,tol in [(2,19,ports['objective_entry'],ports['objective_exit'],20.,.01),(20,25,ports['tube_entry'],ports['tube_exit'],178.065,.890325)]:
        m,_=component(model,wi,first,last,entry,exit,zeros,zeros);efl=-1/m[1,0];fo.append(tol-abs(efl-nom));normalized_fo.append(fo[-1]/tol)
        if first==20:fo.append(135.+m[0,0]/m[1,0]);normalized_fo.append(fo[-1]/135.)
    m=NativeReferenceSystem(configured(model,.595));prefix=paraxial_transfer(m.surfaces[m.primary][:20],0.,1.);fo.append(.002-abs(prefix[1,1]));normalized_fo.append(fo[-1]/.002)
    z=model['vertices_mm'];ports_passed=bool(z[:2]==[0.,.725]and z[-1]==SPEC['fixed_image_z_mm']and
        all(ports['objective_entry']<=v<=ports['objective_exit']for v in z[2:20])and all(ports['tube_entry']<=v<=ports['tube_exit']for v in z[20:26]))
    # Perturbation stack-datum conventions do not redefine nominal interface
    # allocations; report any excursion separately, without certifying assembly.
    nominal_ports=ports_passed or perturbed
    focus=dict(performed=False,passed=None)
    if include_focus:
        axial_tasks=[(model,nm/1000.,[0.])for nm in range(570,621)]
        axial=list(executor.map(focus_audit,axial_tasks))if executor else[focus_audit(t)for t in axial_tasks]
        axial=[r[0]for r in axial];grid=np.linspace(0.,1.25,fields);fc=focus_audit((model,.595,grid));origin=fc[0]['mean'];values=np.array([r['mean']-origin for r in fc]);cache={float(h):float(v)for h,v in zip(grid,values)};extrema=[]
        def field_focus(h):
            h=float(h)
            if h not in cache:cache[h]=focus_audit((model,.595,[h]))[0]['mean']-origin
            return cache[h]
        if local_search:
            for i in range(1,len(grid)-1):
                if abs(values[i])>=abs(values[i-1])and abs(values[i])>=abs(values[i+1]):
                    fit=minimize_scalar(lambda h:-abs(field_focus(h)),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-8})
                    extrema.append(dict(field_mm=float(fit.x),value_mm=field_focus(fit.x),success=bool(fit.success)))
        lca=float(np.ptp([r['full']for r in axial]));fcmax=float(max(abs(v)for v in cache.values()))
        focus=dict(performed=True,passed=lca<=.0007 and fcmax<=.0007 and all(r['success']for r in extrema),
            LCA_um=lca*1000,FC_um=fcmax*1000,sampling=[20,64],axial_wavelengths=51,field_grid=fields,local_extrema=extrema,
            axial_rows=axial,field_rows=[dict(r,relative_object_mm=float(v))for r,v in zip(fc,values)])
    all_local=all(r['all_local_searches_succeeded']for r in rows)
    passed=quality<=1. and physical and body['passed']and min(fo)>=0. and nominal_ports and all_local and (focus['passed']if include_focus else True)
    return dict(passed=bool(passed),green8_WFE_norm=quality,physical_all15_passed=physical,fullbody=body,
        first_order_passed=bool(min(fo)>=0.),minimum_first_order_slack=float(min(fo)),nominal_ports_passed=ports_passed,
        minimum_normalized_first_order_slack=float(min(normalized_fo)),
        perturbed_port_datums_not_certified=perturbed,focus=focus,fields=fields,WFE_pupils=['GQ12x48','RECT16'],
        full_NA=.5,local_extrema_searched=local_search,all_local_searches_succeeded=all_local,
        physical_rays=sum(r['rays']for r in rows),per_wave=rows,manufacturing_release=False,statistical_yield_estimated=False)
