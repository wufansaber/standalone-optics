"""Actual frozen 14-piece case: fixed image, analytic OPD/focus and solid gates.

Only this case's finite telecentric launch/reference is implemented here.
No black-box internals, vendor catalog file, compensator or refocus is used.
"""
from copy import deepcopy
from dataclasses import replace
import gzip,hashlib,json,math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.ai_zemax.standalone import disk_quadrature,trace,directions_from_slopes,reference_opd_mm,paraxial_transfer,wavefront_rms
from scripts.ai_zemax.standalone_derivatives import (RayParameter,TraceDerivativeResult,trace_with_jacobian,
    reference_opd_with_jacobian,wavefront_rms_with_jacobian,paraxial_transfer_with_jacobian,cache_primal_trace,subset_primal_cache)
from scripts.ai_zemax.standalone_analysis import best_focus
from scripts.ai_zemax.standalone_streaming import StreamingQR
from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin
from scripts.ai_zemax.standalone_solid_profiles import FullProfileGeometry
from reference import NativeReferenceSystem

HERE=Path(__file__).resolve().parent
RAW=gzip.decompress((HERE/'FROZEN_MODEL.json.gz').read_bytes())
PINS=json.loads((HERE/'INPUT_PINS.json').read_text())
if hashlib.sha256(RAW).hexdigest()!=PINS['frozen_extended_serialization_sha256']:raise ValueError('Frozen model identity changed')
BASE=json.loads(RAW);SPEC_RAW=(HERE/'SPECIFICATION.json').read_bytes()
if 'specification_sha256'in PINS and hashlib.sha256(SPEC_RAW).hexdigest()!=PINS['specification_sha256']:raise ValueError('Frozen acceptance specification changed')
SPEC=json.loads(SPEC_RAW);CAPS={float(k):v for k,v in SPEC['WFE_targets_waves'].items()}
ORIGINAL_WAVES=tuple(BASE['wavelengths_um'][:15]);PRIMARY={1:.536,2:.469,3:.595,4:.790}


def configured(model,w):
    m=deepcopy(model);cfg=4 if w==.790 else 3 if w==.595 else 2 if w<.5 else 1
    m['primary_wave']=m['wavelengths_um'].index(PRIMARY[cfg])
    if cfg==4:m['vertices_mm']=[z+(.002824479276 if i>=2 else 0.)for i,z in enumerate(m['vertices_mm'])]
    return m


def sliced(value,sl):
    r=value.trace
    result=replace(r,points=r.points[sl],directions=r.directions[sl],opl_mm=r.opl_mm[sl],valid=r.valid[sl],
                   failed_surface=r.failed_surface[sl],reason=r.reason[sl],surface_points=None)
    return TraceDerivativeResult(result,value.points_jacobian[sl],value.directions_jacobian[sl],value.opl_jacobian[sl],
                                 value.differentiable[sl],value.derivative_reason[sl],value.parameters)


def focal(xy,dx,d,dd,weights,m,dm):
    t=d[:,:2]/d[:,2,None];dt=dd[:,:2]/d[:,2,None,None]-d[:,:2,None]*dd[:,2,None,:]/d[:,2,None,None]**2
    x=xy-weights@xy;t=t-weights@t
    jx=dx-np.einsum('n,nip->ip',weights,dx);jt=dt-np.einsum('n,nip->ip',weights,dt)
    cov=np.sum(weights[:,None]*x*t,axis=0);var=np.sum(weights[:,None]*t*t,axis=0)
    if np.any(var<=0.):raise ValueError('Degenerate focal covariance')
    jc=np.einsum('n,nip,ni->ip',weights,jx,t)+np.einsum('n,ni,nip->ip',weights,x,jt)
    jv=2*np.einsum('n,ni,nip->ip',weights,t,jt)
    f=-cov/var;jf=-jc/var[:,None]+cov[:,None]*jv/var[:,None]**2
    full=-sum(cov)/sum(var);jfull=-sum(jc)/sum(var)+sum(cov)*sum(jv)/sum(var)**2
    return dict(full=float(full/m**2),full_J=jfull/m**2-2*full*dm/m**3,
                mean=float(np.mean(f)/m**2),mean_J=np.mean(jf,axis=0)/m**2-2*np.mean(f)*dm/m**3)


class Design:
    def __init__(self,base=BASE,*,scope='all'):
        self.base=deepcopy(base);self.c0=np.array([r['curvature_mm_inv']for r in base['surface_rows']]);self.z0=np.array(base['vertices_mm'])
        self.curved=list(range(20 if scope=='tube'else 2,26));self.positions=[i for i in self.curved if i not in (2,20)]
        self.cs=np.maximum(abs(self.c0[self.curved])*.02,.0003);self.zs=.02
        self.n=len(self.curved)+len(self.positions);self.cjac=np.zeros((27,self.n));self.zjac=np.zeros_like(self.cjac)
        for j,(i,s)in enumerate(zip(self.curved,self.cs)):self.cjac[i,j]=s
        for j,i in enumerate(self.positions):self.zjac[i,len(self.curved)+j]=self.zs
        self.parameters=tuple([RayParameter(i,'curvature_mm_inv',scale=s)for i,s in zip(self.curved,self.cs)]+
                              [RayParameter(i,'vertex_mm',scale=self.zs)for i in self.positions])
        self.geo=FullProfileGeometry(base,self.cjac,self.zjac,min_CT=.6,min_air=.25,min_rim=.25,guard=0.,
            minimum_domain_root=.1,tube_axial_span_max_mm=62.,tube_span_reserve_mm=.2,bins=3)
        floors={(r['front_surface']-1,r['rear_surface']-1):r['minimum_fullbody_nominal_mm']for r in base['manufacturing_policy']['piece_floors']}
        for pair in self.geo.pairs:
            if pair['kind']=='glass_full_OD':pair['floor']=floors[pair['first'],pair['last']]
        bounds=[[-1.,1.]for i in self.curved]
        p=SPEC['ports_mm']
        for i in self.positions:
            lo,hi=(p['objective_entry'],p['objective_exit'])if i<20 else(p['tube_entry'],p['tube_exit'])
            bounds.append([max(-5.,(lo-self.z0[i])/self.zs),min(5.,(hi-self.z0[i])/self.zs)])
        self.bounds=np.array(bounds)

    def model(self,q):
        q=np.asarray(q,dtype=float);m=deepcopy(self.base);c=self.c0+self.cjac@q;m['vertices_mm']=(self.z0+self.zjac@q).tolist()
        for i,r in enumerate(m['surface_rows']):r.update(curvature_mm_inv=float(c[i]),conic=0.,even_coefficients=[])
        for i,r in enumerate(m['explicit_nominal_solid_profiles']):r['optical_sphere_curvature_mm_inv']=float(c[i])
        return m

    def geometry(self,model,*,perturbed=False,cjac=None):
        # Scenario thickness allowances consume the PREDECLARED .03mm budget;
        # nominal allocations remain unchanged. This is not a new supplier spec.
        floors=[p['floor']for p in self.geo.pairs];prior_cjac=self.geo.cjac
        cj=self.cjac if cjac is None else cjac;self.geo.cjac=cj
        wi=model['wavelengths_um'].index(.595)
        for pair in self.geo.pairs:
            if pair['kind']!='glass_full_OD' or pair['first']==0:continue
            a,b=pair['first'],pair['last'];n=model['indices_by_wave'][wi][a+1]
            ca,cb=[model['surface_rows'][i]['curvature_mm_inv']for i in (a,b)];ct=model['vertices_mm'][b]-model['vertices_mm'][a]
            power=(n-1)*(ca-cb)+ct/n*(n-1)**2*ca*cb;od=2*min(self.geo.body[a],self.geo.body[b])
            role=(2.03 if a>=20 else 1.53)if power>=0 else max(.6,.06*od)+.03
            pair['floor']=max(pair['floor'],role)
        if perturbed:
            for p in self.geo.pairs:
                if p['kind']=='glass_full_OD'and p['first']>0:p['floor']-=.03
        try:g,j,_=self.geo.constraints(model)
        finally:
            self.geo.cjac=prior_cjac
            for p,floor in zip(self.geo.pairs,floors):p['floor']=floor
        c=np.array([r['curvature_mm_inv']for r in model['surface_rows']]);face=self.geo.face
        domain=1-c*c*face*face-.01;dj=-2*c[:,None]*face[:,None]**2*cj
        ct=[];ctj=[]
        for a,b in self.geo.solids:
            if a:ct.append(20.-(model['vertices_mm'][b]-model['vertices_mm'][a]));ctj.append(self.zjac[a]-self.zjac[b])
        return np.r_[g,domain,ct],np.r_[j,dj,ctj]


def apply_scenario(model,payload):
    m=deepcopy(model);kind=payload.get('kind','nominal')
    if kind=='radius':
        i=payload['surface']-1;m['surface_rows'][i]['curvature_mm_inv']/=1+payload['relative_R_error']
        m['explicit_nominal_solid_profiles'][i]['optical_sphere_curvature_mm_inv']=m['surface_rows'][i]['curvature_mm_inv']
    elif kind in ('CT','air'):
        first=payload['rear_surface']-1 if kind=='CT'else 23
        for i in range(first,26):m['vertices_mm'][i]+=payload['axial_error_mm']
    elif kind!='nominal':raise ValueError('Unknown physical error scenario')
    return m


def scenario_set():
    result={'nominal':dict(kind='nominal')}
    for i in range(21,27):
        for sign in (-1,1):result[f'R_S{i}_{sign:+d}']=dict(kind='radius',surface=i,relative_R_error=sign*.0001)
    for name,rear in [('T1',22),('T2',23),('T3',25),('T4',26)]:
        for sign in (-1,1):result[f'CT_{name}_{sign:+d}']=dict(kind='CT',rear_surface=rear,axial_error_mm=sign*.02)
    for sign in (-1,1):result[f'AIR_{sign:+d}']=dict(kind='air',axial_error_mm=sign*.02)
    return result


def wave_job(task):
    model,parameters,w,fields,sampling,derivatives,*options=task;compressed=options[0]if options else False
    supplied_cache=options[1]if len(options)>1 else None;restoration=options[2]if len(options)>2 else False
    m=configured(model,w)
    if restoration:
        # Explicit uncut spherical continuation is a RESTORATION SURROGATE.
        # Real CA/face limits remain negative gates and final acceptance always
        # uses the original physical apertures. No failed ray is filtered.
        for data in m['surface_rows'][2:]:data['aperture_radius_mm']=None
    system=NativeReferenceSystem(m);wi=m['wavelengths_um'].index(w)
    if sampling[0]=='rect':
        from scripts.ai_zemax.standalone_system import rectangular_pupil
        nodes,weights=rectangular_pupil(sampling[1])
    else:nodes,weights=disk_quadrature(*sampling)
    weights=np.asarray(weights)/sum(weights);n=len(nodes)
    a=np.arange(64)*2*np.pi/64;pupil=np.r_[nodes,[[0.,0.]],np.c_[np.cos(a),np.sin(a)]];stride=len(pupil)
    origins=[];raw=[]
    for h in fields:
        origin,slopes=system.launch_field([0.,h],wi);direction=directions_from_slopes(slopes+system.pupil_scale*pupil)
        origins.extend(np.broadcast_to(np.r_[origin,0.],direction.shape));raw.extend(direction)
    prefix=trace(system.surfaces[wi][:2],origins,raw,1.,record_path=True)
    if not prefix.valid.all():raise ValueError('Fixed window invalid ray')
    local=tuple(replace(p,surface_index=p.surface_index-2)for p in parameters)
    cached=supplied_cache if supplied_cache is not None else cache_primal_trace(system.surfaces[wi][2:],prefix.points,prefix.directions,float(system.indices[wi,2]))
    if not cached.result.valid.all()or np.any(cached.result.negative_segment_count):raise ValueError('Invalid/backward physical ray; no ray filtering')
    if derivatives:
        rays=trace_with_jacobian(system.surfaces[wi][2:],prefix.points,prefix.directions,local,
            object_index=float(system.indices[wi,2]),record_path=True,primal_cache=cached)
        if not rays.differentiable.all():raise ValueError('Ray derivative lies on a branch/aperture boundary')
        rays.trace.opl_mm+=prefix.opl_mm
        reference=system.reference_tangent(parameters)
        matrix,mj=paraxial_transfer_with_jacobian(system.surfaces[wi],0.,parameters)
    else:
        primal=replace(cached.result,opl_mm=cached.result.opl_mm+prefix.opl_mm)
        rays=None;matrix=paraxial_transfer(system.surfaces[wi],0.,1.);mj=np.zeros((2,2,len(parameters)))
    primal=rays.trace if rays else primal
    mag=float(matrix[0,0]-matrix[0,1]/1e10);magj=mj[0,0]-mj[0,1]/1e10
    cap=CAPS.get(w);phase=[];phasej=[];phaseR=[];rms=[];rmsj=[];chief=[];chiefj=[];foci=[];physical=[];physicalj=[]
    for fi,h in enumerate(fields):
        off=fi*stride;sl=slice(off,off+n);ci=off+n
        if derivatives:
            bundle=sliced(rays,sl);ch=sliced(rays,slice(ci,ci+1));dz=ch.trace.directions[0,2]
            radius=reference.distance_mm/dz;dr=reference.distance_jacobian/dz-reference.distance_mm*ch.directions_jacobian[0,2]/dz**2
            opd,opdj=reference_opd_with_jacobian(bundle,ch,radius,pupil_distance_jacobian=dr)
            wf=wavefront_rms_with_jacobian(opd,opdj,weights,w,bundle.trace.directions[:,:2],tilt_jacobian=bundle.directions_jacobian[:,:2])
            if cap:
                pv=np.sqrt(weights)*wf.residual_waves/cap;pj=np.sqrt(weights)[:,None]*wf.residual_jacobian/cap;phase.append(pv)
                if compressed:
                    qr=StreamingQR(len(parameters)+1)
                    for begin in range(0,n,64):qr.update(np.c_[pj[begin:begin+64],pv[begin:begin+64]])
                    phaseR.append(qr.result())
                else:phasej.append(pj)
                rms.append(wf.rms_waves/cap);rmsj.append(wf.rms_jacobian/cap)
            chief.append([primal.points[ci,1],primal.directions[ci,1]]);chiefj.append([rays.points_jacobian[ci,1],rays.directions_jacobian[ci,1]])
            foci.append(focal(primal.points[sl,:2],rays.points_jacobian[sl,:2],primal.directions[sl],rays.directions_jacobian[sl],weights,mag,magj))
        else:
            ch=replace(primal,points=primal.points[ci:ci+1],directions=primal.directions[ci:ci+1],opl_mm=primal.opl_mm[ci:ci+1],valid=primal.valid[ci:ci+1],failed_surface=primal.failed_surface[ci:ci+1],reason=primal.reason[ci:ci+1],surface_points=None)
            opd=reference_opd_mm(replace(primal,points=primal.points[sl],directions=primal.directions[sl],opl_mm=primal.opl_mm[sl],valid=primal.valid[sl],failed_surface=primal.failed_surface[sl],reason=primal.reason[sl],surface_points=None),ch,system.reference_distance_mm/ch.directions[0,2],1.)
            if cap:
                value,residual=wavefront_rms(opd,weights,w,primal.directions[sl,:2]);rms.append(value/cap);phase.append(np.sqrt(weights)*residual/cap)
            chief.append([primal.points[ci,1],primal.directions[ci,1]])
            f=best_focus(primal.points[sl,:2],primal.directions[sl],weights,tangential_axis=(0.,1.))
            foci.append(dict(full=float(f['radial_rms_shift']/mag**2),mean=float(f['mean_shift']/mag**2)))
    # Physical envelope includes the integration pupil, full rim and chiefs at
    # every field and EVERY physical surface, not just the image plane.
    oldz=prefix.points[:,2];oldj=np.zeros((len(oldz),len(parameters)))
    for j in range(25):
        si=j+2;xy=primal.surface_points[j,:,:2];r=np.linalg.norm(xy,axis=1);ix=int(np.argmax(r))
        if si<26:
            data=m['surface_rows'][si];allowed=min(data['clear_radius_mm']-(.25 if si>=20 else .08),data['mechanical_radius_mm']-.20)
            physical.append((allowed-r[ix])/.1)
            if derivatives:physicalj.append(-xy[ix]@rays.path_jacobians[j,ix,:2]/max(r[ix],1e-20)/.1)
        z=primal.surface_points[j,:,2];gap=z-oldz;ix=int(np.argmin(gap));physical.append((gap[ix]-.01)/.1)
        if derivatives:
            zj=rays.path_jacobians[j,:,2];physicalj.append((zj[ix]-oldj[ix])/.1);oldj=zj
        oldz=z
    result=dict(wavelength=w,phase=np.array(phase),rms=np.array(rms),chief=np.array(chief),focus=foci,
                mag=mag,physical=np.array(physical),rays=len(primal.points),fields=list(fields),cache_reused=supplied_cache is not None)
    if not derivatives:result['primal_cache']=cached
    if derivatives:
        result.update(rms_J=np.array(rmsj),chief_J=np.array(chiefj),mag_J=magj,physical_J=np.array(physicalj))
        result['phase_R'if compressed else'phase_J']=np.array(phaseR if compressed else phasej)
    return result


def component(model,wi,first,last,input_z,output_z,cjac,zjac):
    matrix=np.eye(2);j=np.zeros((2,2,cjac.shape[1]));previous=float(input_z);pj=np.zeros(cjac.shape[1]);index=model['indices_by_wave'][wi][first]
    for i in range(first,last+1):
        gap=model['vertices_mm'][i]-previous;drift=np.array([[1.,gap/index],[0.,1.]]);dj=np.zeros_like(j);dj[0,1]=(zjac[i]-pj)/index
        j=np.einsum('abp,bc->acp',dj,matrix)+np.einsum('ab,bcp->acp',drift,j);matrix=drift@matrix
        after=model['indices_by_wave'][wi][i+1];power=(after-index)*model['surface_rows'][i]['curvature_mm_inv'];bend=np.array([[1.,0.],[-power,1.]]);bj=np.zeros_like(j);bj[1,0]=-(after-index)*cjac[i]
        j=np.einsum('abp,bc->acp',bj,matrix)+np.einsum('ab,bcp->acp',bend,j);matrix=bend@matrix
        previous=model['vertices_mm'][i];pj=zjac[i];index=after
    drift=np.array([[1.,(output_z-previous)/index],[0.,1.]]);dj=np.zeros_like(j);dj[0,1]=-pj/index
    return drift@matrix,np.einsum('abp,bc->acp',dj,matrix)+np.einsum('ab,bcp->acp',drift,j)


class GreenProblem(EfficientSolveMixin):
    def __init__(self,fields,*,design=None,sampling=(4,12),focus_waves=(.570,.580,.595,.610,.620),
                 executor=None,scenario=None,include_span=True,physical_all15=True,wfe_guard=1.,strict_physical=True,constraint_buffer=0.):
        self.design=design or Design();self.bounds=self.design.bounds;self.fields=sorted(set([0.,*map(float,fields)]));self.sampling=sampling
        self.focus_waves=tuple(focus_waves);self.executor=executor;self.scenario=scenario or {'kind':'nominal'};self.include_span=include_span;self.physical_all15=physical_all15
        self.wfe_guard=float(wfe_guard);self.local_norm_specs={'wfe':dict(vector='wave_vector',gain=1.,offset=self.wfe_guard)}
        self.strict_physical=bool(strict_physical)
        self.constraint_buffer=float(constraint_buffer)
        if not np.isfinite(self.constraint_buffer)or self.constraint_buffer<0.:raise ValueError('Use a finite nonnegative working constraint buffer')
        self.cache={};self.counts=dict(primal_calls=0,analytic_calls=0,primal_rays=0,analytic_rays=0,reused_primal_rays=0)
    def model(self,q):return self.design.model(q)
    def evaluate(self,q,derivatives=False,compressed=True):
        q=np.asarray(q,dtype=float);key=(q.tobytes(),derivatives,compressed)
        if key in self.cache:return self.cache[key]
        model=apply_scenario(self.model(q),self.scenario)
        geometry_cjac=self.design.cjac.copy();parameters=self.design.parameters
        if self.scenario['kind']=='radius':
            si=self.scenario['surface']-1;factor=1+self.scenario['relative_R_error'];geometry_cjac[si]/=factor
            parameters=tuple(replace(p,scale=p.scale/factor)if p.surface_index==si and p.attribute=='curvature_mm_inv'else p for p in parameters)
        geometry,gj=self.design.geometry(model,perturbed=self.scenario['kind']!='nominal',cjac=geometry_cjac)
        if self.strict_physical and np.min(geometry)<0.:return None
        waves=sorted(set(CAPS)|set(self.focus_waves)|(set(ORIGINAL_WAVES)if self.physical_all15 else set()))
        tasks=[];prior=self.cache.get((q.tobytes(),False,True))
        for w in waves:
            fields=self.fields if w in CAPS or w in ORIGINAL_WAVES else[0.]
            chunk=4 if derivatives else len(fields)
            for begin in range(0,len(fields),chunk):
                chosen=fields[begin:begin+chunk];cached=None
                if derivatives and prior is not None and w in prior['by_wave']:
                    old=prior['by_wave'][w];stride=len(old['primal_cache'].directions)//len(old['fields'])
                    selected=np.concatenate([np.arange(old['fields'].index(h)*stride,(old['fields'].index(h)+1)*stride)for h in chosen])
                    cached=subset_primal_cache(old['primal_cache'],selected)
                tasks.append((model,parameters,w,chosen,self.sampling,derivatives,compressed,cached,not self.strict_physical))
        try:pieces=list(self.executor.map(wave_job,tasks))if self.executor else[wave_job(t)for t in tasks]
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):return None
        rows=[]
        for w in waves:
            parts=[r for r in pieces if r['wavelength']==w];r=dict(parts[0])
            for key0 in ['phase','rms','chief']+(['rms_J','chief_J','phase_R'if compressed else'phase_J']if derivatives else[]):
                r[key0]=np.concatenate([part[key0]for part in parts])
            # A physical row is the envelope over ALL fields/rays, independent
            # of derivative memory blocks. Select its actual minimizing block
            # and matching tangent; concatenation would change the gate layout.
            physical=np.stack([part['physical']for part in parts]);which=np.argmin(physical,axis=0)
            r['physical']=physical[which,np.arange(physical.shape[1])]
            if derivatives:r['physical_J']=np.stack([part['physical_J']for part in parts])[which,np.arange(physical.shape[1])]
            r['focus']=[value for part in parts for value in part['focus']];r['fields']=[value for part in parts for value in part['fields']];r['rays']=sum(part['rays']for part in parts);rows.append(r)
        if self.strict_physical and any(np.min(r['physical']) < 0. for r in rows):return None
        self.counts['analytic_calls'if derivatives else'primal_calls']+=1;self.counts['analytic_rays'if derivatives else'primal_rays']+=sum(r['rays']for r in rows)
        self.counts['reused_primal_rays']+=sum(r['rays']for r in pieces if r['cache_reused'])
        by={r['wavelength']:r for r in rows};p=len(q);gates={'geometry':geometry};jac={'geometry':gj};phase=[];phasej=[]
        def add(name,value,derivative=None):
            gates[name]=np.asarray(value,dtype=float)
            if derivatives:jac[name]=np.asarray(derivative,dtype=float)
        add('physical',np.concatenate([r['physical']for r in rows]),np.concatenate([r['physical_J']for r in rows])if derivatives else None)
        mags=[];magsj=[];angles=[];anglesj=[];dist=[];distj=[];colors=[];colorj=[]
        for w in sorted(CAPS):
            r=by[w];phase.extend(r['phase']);dm=r.get('mag_J');dy=r.get('chief_J')
            if derivatives:phasej.extend(r['phase_R'if compressed else'phase_J'])
            a=(r['mag']+8.9)/.01;mags.extend([1-a,1+a])
            y=r['chief'][:,0];d=r['chief'][:,1]/math.sin(math.radians(.07));angles.extend(1-d);angles.extend(1+d)
            h=np.array(self.fields[1:]);de=(y[1:]/r['mag']/h-1.)/.006;dist.extend(1-de);dist.extend(1+de)
            if derivatives:
                magsj.extend([-dm/.01,dm/.01]);anglej=dy[:,1]/math.sin(math.radians(.07));anglesj.extend(-anglej);anglesj.extend(anglej)
                dd=(dy[1:,0]/r['mag']-y[1:,None]*dm/r['mag']**2)/h[:,None]/.006;distj.extend(-dd);distj.extend(dd)
            if w in SPEC['hard_green_lateral_color']['reference_measured_waves_um']:
                c=(r['chief'][:,0]-by[.536]['chief'][:,0])/.0018974173564139107;colors.extend(1-c);colors.extend(1+c)
                if derivatives:
                    cj=(dy[:,0]-by[.536]['chief_J'][:,0])/.0018974173564139107;colorj.extend(-cj);colorj.extend(cj)
        vector=np.array(phase);add('wfe',self.wfe_guard-np.linalg.norm(vector,axis=-1),np.array([rj for w in sorted(CAPS)for rj in by[w]['rms_J']])*-1 if derivatives else None)
        for name,values,jvalues in [('magnification',mags,magsj),('chief_angle',angles,anglesj),('distortion',dist,distj),('lateral_color',colors,colorj)]:add(name,values,jvalues)
        wi=model['wavelengths_um'].index(.595);ports=SPEC['ports_mm'];fo=[];foj=[]
        for first,last,entry,exit,nom,tol in [(2,19,ports['objective_entry'],ports['objective_exit'],20.,.01),(20,25,ports['tube_entry'],ports['tube_exit'],178.065,.890325)]:
            m,mj=component(model,wi,first,last,entry,exit,geometry_cjac,self.design.zjac);efl=-1/m[1,0];dj=mj[1,0]/m[1,0]**2;v=(efl-nom)/tol;fo.extend([1-v,1+v]);foj.extend([-dj/tol,dj/tol])
            if first==20:
                bfl=-m[0,0]/m[1,0];bj=-mj[0,0]/m[1,0]+m[0,0]*mj[1,0]/m[1,0]**2;fo.append((135.-bfl)/1.);foj.append(-bj)
        system=NativeReferenceSystem(configured(model,.595));mp,jp=paraxial_transfer_with_jacobian(system.surfaces[system.primary][:20],0.,tuple(p for p in parameters if p.surface_index<20))
        dd=np.zeros(p);selection=[i for i,p0 in enumerate(self.design.parameters)if p0.surface_index<20];dd[selection]=jp[1,1];v=mp[1,1]/.002;fo.extend([1-v,1+v]);foj.extend([-dd/.002,dd/.002]);add('first_order',fo,foj)
        fvalues=np.array([by[w]['focus'][0]['full']for w in self.focus_waves]);fJ=np.array([by[w]['focus'][0]['full_J']for w in self.focus_waves])if derivatives else None
        if self.include_span:
            low,high=int(np.argmin(fvalues)),int(np.argmax(fvalues));add('lca',[(.0007-(fvalues[high]-fvalues[low]))/.0007],[(fJ[low]-fJ[high])/.0007]if derivatives else None)
        f595=by[.595]['focus'];fc=np.array([r['mean']-f595[0]['mean']for r in f595]);add('field_curvature',np.r_[1-fc/.0007,1+fc/.0007],np.r_[-np.array([r['mean_J']-f595[0]['mean_J']for r in f595])/.0007,np.array([r['mean_J']-f595[0]['mean_J']for r in f595])/.0007]if derivatives else None)
        for name in gates:
            if name not in ('geometry','physical','wfe'):gates[name]=gates[name]-self.constraint_buffer
        result=dict(gates=gates,wave_vector=vector,span_values=fvalues,by_wave=by,model=model)
        if derivatives:result.update(gate_J=jac,span_J=fJ,**{'wave_R'if compressed else'wave_J':np.array(phasej)})
        self.cache[key]=result
        # Retain only the current physical point and its analytic result.
        if len(self.cache)>4:self.cache={key:result}
        return result

    def batch(self,qs):
        result=[]
        for q in qs:
            try:result.append(self.evaluate(q))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError):result.append(None)
        return result
    @staticmethod
    def slacks(row):return row['gates']
    def jacobian(self,q,base,keys,*,compressed=True):
        row=self.evaluate(q,derivatives=True,compressed=compressed)
        if row is None:raise ValueError('Analytic derivative invalid at current physical point')
        result=dict(slacks=np.concatenate([row['gate_J'][k].reshape(-1,len(q))for k in keys]),span_jacobian=row['span_J'])
        if compressed:
            # Complete weighted pupil vectors are compressed by field; no ray
            # sample or integration weight is dropped.
            result['compressed_vectors']={'wave_vector':row['wave_R']}
        else:result['vectors']={'wave_vector':row['wave_J']}
        return result


RULES={'gq4x12':(4,12),'gq12x48':(12,48),'rect16':('rect',16),'gq20x64':(20,64)}


def chief_job(task):
    model,parameters,w,derivatives,count=task;m=configured(model,w)
    from scripts.ai_zemax.standalone_system import OpticalSystem
    system=OpticalSystem(m);wi=m['wavelengths_um'].index(w);fields=np.linspace(0.,1.25,count)
    origins=np.c_[np.zeros(count),fields,np.zeros(count)];raw=directions_from_slopes(np.c_[np.zeros(count),fields*system.chief_slope_scale])
    prefix=trace(system.surfaces[wi][:2],origins,raw,1.)
    local=tuple(replace(p,surface_index=p.surface_index-2)for p in parameters)
    if derivatives:
        result=trace_with_jacobian(system.surfaces[wi][2:],prefix.points,prefix.directions,local,object_index=system.indices[wi,2])
        if not result.differentiable.all():raise ValueError('Dense chief derivative invalid')
        primal=result.trace;matrix,mj=paraxial_transfer_with_jacobian(system.surfaces[wi],0.,parameters)
    else:
        primal=trace(system.surfaces[wi][2:],prefix.points,prefix.directions,system.indices[wi,2]);matrix=paraxial_transfer(system.surfaces[wi],0.,1.)
    if not primal.valid.all()or np.any(primal.negative_segment_count):raise ValueError('Dense chief physical failure')
    mag=float(matrix[0,0]-matrix[0,1]/1e10);answer=dict(wavelength=w,y=primal.points[:,1],direction=primal.directions[:,1],mag=mag)
    if derivatives:answer.update(y_J=result.points_jacobian[:,1],direction_J=result.directions_jacobian[:,1],mag_J=mj[0,0]-mj[0,1]/1e10)
    return answer


def dense_first_order(model,parameters,*,executor=None,derivatives=False,fields=65):
    if type(fields)is not int or fields<2:raise ValueError("At least two chief field samples required")
    tasks=[(model,parameters,w,derivatives,fields)for w in sorted(CAPS)]
    rows=list(executor.map(chief_job,tasks))if executor else[chief_job(t)for t in tasks];by={r['wavelength']:r for r in rows}
    values=[];jac=[];h=np.linspace(0.,1.25,fields)[1:]
    for r in rows:
        angle=r['direction']/math.sin(math.radians(.07));dist=(r['y'][1:]/r['mag']/h-1)/.006;mag=(r['mag']+8.9)/.01
        values.extend(np.r_[1-angle,1+angle,1-dist,1+dist,1-mag,1+mag])
        if derivatives:
            aj=r['direction_J']/math.sin(math.radians(.07));dj=(r['y_J'][1:]/r['mag']-r['y'][1:,None]*r['mag_J']/r['mag']**2)/h[:,None]/.006
            jac.extend(np.r_[-aj,aj,-dj,dj,-r['mag_J'][None]/.01,r['mag_J'][None]/.01])
    for w in SPEC['hard_green_lateral_color']['reference_measured_waves_um']:
        r=by[w];c=(r['y']-by[.536]['y'])/.0018974173564139107;values.extend(np.r_[1-c,1+c])
        if derivatives:
            cj=(r['y_J']-by[.536]['y_J'])/.0018974173564139107;jac.extend(np.r_[-cj,cj])
    return np.array(values),np.array(jac)if derivatives else None


class WorkingProblem(EfficientSolveMixin):
    """Mixed complete pupil rules, dynamic fields and separate focal sampling."""
    def __init__(self,samples,*,design=None,executor=None,scenario=None,include_span=True,wfe_guard=1.,constraint_buffer=0.):
        self.design=design or Design();self.bounds=self.design.bounds;self.parts=[]
        self.samples=deepcopy(list(samples));self.include_span=include_span
        optical={};focus_fields={0.,.625,1.25,.8833608046477396};focus_waves={.570,.580,.595,.610,.620};focus_rule='gq4x12'
        for sample in self.samples:
            kind=sample.get('kind','wfe');rule=sample.get('pupil_rule','gq4x12')
            if rule not in RULES:raise ValueError('Unknown complete pupil integration rule')
            if kind=='wfe':optical.setdefault(rule,set()).add(float(sample['field_mm']))
            elif kind=='fc':focus_fields.add(float(sample['field_mm']));focus_rule='gq20x64'
            elif kind=='lca':focus_waves.add(sample['wavelength_nm']/1000.);focus_rule='gq20x64'
            else:raise ValueError('Unknown working optical sample kind')
        if not optical:raise ValueError('At least one wavefront sample is required')
        for rule,fields in sorted(optical.items()):self.parts.append(GreenProblem(sorted(fields),design=self.design,sampling=RULES[rule],executor=executor,scenario=scenario,include_span=False,wfe_guard=wfe_guard,constraint_buffer=constraint_buffer))
        self.focus=GreenProblem(sorted(focus_fields),design=self.design,sampling=RULES[focus_rule],focus_waves=tuple(sorted(focus_waves)),executor=executor,scenario=scenario,include_span=include_span,physical_all15=False,constraint_buffer=constraint_buffer)
        self.local_norm_specs={'wfe':dict(vector='wave_vector',gain=1.,offset=wfe_guard)}
    def model(self,q):return self.design.model(q)
    def batch(self,qs):
        result=[]
        for q in qs:
            parts=[p.batch([q])[0]for p in self.parts];focus=self.focus.batch([q])[0]
            if any(r is None for r in parts)or focus is None:result.append(None);continue
            vectors=[r['wave_vector']for r in parts];width=max(v.shape[-1]for v in vectors)
            vector=np.concatenate([np.pad(v,((0,0),(0,width-v.shape[-1])))for v in vectors])
            result.append(dict(parts=parts,focus=focus,wave_vector=vector,span_values=focus['span_values']))
        return result
    def slacks(self,row):
        gates={}
        for name in row['parts'][0]['gates']:
            if name in ('field_curvature','lca'):continue
            selected=row['parts'][:1]if name in ('geometry','first_order','magnification')else row['parts']
            gates[name]=np.concatenate([np.ravel(r['gates'][name])for r in selected])
        for name in ('field_curvature','lca'):
            if name in row['focus']['gates']:gates[name]=row['focus']['gates'][name]
        return gates
    def jacobian(self,q,base,keys):
        evaluated=[p.evaluate(q,True)for p in self.parts];focus=self.focus.evaluate(q,True)
        if any(r is None for r in evaluated)or focus is None:raise ValueError('Mixed-rule analytic base invalid')
        blocks={}
        for name in keys:
            if name in ('field_curvature','lca'):blocks[name]=focus['gate_J'][name].reshape(-1,len(q))
            else:
                selected=evaluated[:1]if name in ('geometry','first_order','magnification')else evaluated
                blocks[name]=np.concatenate([r['gate_J'][name].reshape(-1,len(q))for r in selected])
        compressed=[]
        for r in evaluated:
            compressed.append(r['wave_R'])
        height=max(r.shape[-2]for r in compressed)
        return dict(slacks=np.concatenate([blocks[k]for k in keys]),
                    compressed_vectors={'wave_vector':np.concatenate([np.pad(r,((0,0),(0,height-r.shape[-2]),(0,0)))for r in compressed])},span_jacobian=focus['span_J'])
