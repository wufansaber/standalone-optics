"""Full nominal coaxial solids with finite spherical faces and flat shoulders.

Face cutoffs are FIXED during one local solve.  All optical rays must separately
fit the declared CA.  A shoulder is a non-imaging mechanical annulus; it never
changes the traced optical wavefront.  This module does not certify mounting,
bevels, stress, tolerances or supplier capability.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

MODULE_SOURCE=Path(__file__).read_bytes()
MODULE_SHA=hashlib.sha256(MODULE_SOURCE).hexdigest()


class FullProfileGeometry:
    """Exact tangent of sphere(min(r, face_cutoff)) + shared-block finished OD.

    constraints(model) -> (slacks_mm, jacobian_mm_per_parameter,
                           minimum_unguarded_physical_margin_mm)

    Returned slacks INCLUDE the engineering guard.  Geometry residual and SQP
    callers can use the same rows; the physical minimum excludes the guard.
    In each radial interval bounded by the two face cutoffs, the difference of
    two spherical sags is monotone.  Thus endpoints/cutoffs suffice for the exact
    nominal minimum of this supported profile.  Extra bins aid diagnostics.
    """

    def __init__(self, model, cjac, zjac, *, min_CT, min_air, minimum_domain_root,
                 tube_axial_span_max_mm, tube_span_reserve_mm,
                 guard=0., bins=9, min_rim=None):
        self.min_CT=float(min_CT)
        self.min_air=float(min_air)
        self.min_rim=float(min_air if min_rim is None else min_rim)
        self.guard=float(guard)
        self.minimum_domain_root=float(minimum_domain_root)
        self.tube_axial_span_max_mm=float(tube_axial_span_max_mm)
        self.tube_span_reserve_mm=float(tube_span_reserve_mm)
        if not np.isfinite([self.min_CT,self.min_air,self.min_rim,self.guard,self.minimum_domain_root,
                            self.tube_axial_span_max_mm,self.tube_span_reserve_mm]).all():
            raise ValueError('Solid allocations must all be finite')
        if type(bins)is not int or bins<2:raise ValueError('Use at least two radial support bins')
        if not 0.<=self.tube_span_reserve_mm<self.tube_axial_span_max_mm:
            raise ValueError('Tube span reserve must be less than documented housing length')
        if min(self.min_CT,self.min_air,self.min_rim,self.guard)<0.:
            raise ValueError('Geometry floors and guard must be nonnegative')
        if not 0.<self.minimum_domain_root<1.:
            raise ValueError('Sphere domain root guard must lie in (0,1)')
        self.cjac=np.asarray(cjac,dtype=float)
        self.zjac=np.asarray(zjac,dtype=float)
        self.count=len(model['vertices_mm'])
        if self.cjac.ndim!=2 or self.cjac.shape!=self.zjac.shape or self.cjac.shape[0]!=self.count or not np.isfinite(self.cjac).all() or not np.isfinite(self.zjac).all():
            raise ValueError('Curvature and vertex tangents must align with model surfaces')
        self.nz=self.cjac.shape[1]
        materials=model['material_names']
        if len(materials)!=self.count+1:raise ValueError('Materials must include object channel')
        self.solids=[(i-1,i) for i in range(1,self.count) if materials[i]]
        self.tube_start=model.get('group_boundaries',{}).get('first_tube_surface_index')
        if not self.solids:raise ValueError('No physical solids to audit')
        self.face=np.zeros(self.count)
        for i,row in enumerate(model['surface_rows'][:-1]):
            if row.get('conic',0.)!=0. or any(row.get('even_coefficients',[])):
                raise ValueError('FullProfileGeometry supports spherical optical faces only')
            face=row.get('mechanical_radius_mm')
            if face is None:face=row.get('clear_radius_mm',row.get('aperture_radius_mm'))
            if face is None or not np.isfinite(face) or face<=0.:
                raise ValueError(f'S{i+1} lacks a finite positive spherical face cutoff')
            clear=row.get('clear_radius_mm',row.get('aperture_radius_mm'))
            if clear is not None and face<clear:
                raise ValueError(f'S{i+1} face cutoff is smaller than optical clear aperture')
            self.face[i]=float(face)
        # Explicit inherited OD is part of the physical model, independently
        # of finite optical sphere cutoffs. Validate before freezing pair/rim
        # radii; updating self.body after pair construction would be too late.
        profiles=model.get('explicit_nominal_solid_profiles')
        if not isinstance(profiles,list) or len(profiles)!=self.count-1:
            raise ValueError('Require one explicit body profile per physical non-image surface')
        declared=np.zeros(self.count);seen=set()
        for profile in profiles:
            if not isinstance(profile,dict):raise ValueError('Body profile must be a mapping')
            sn=profile.get('surface')
            if type(sn)is not int or not 1<=sn<self.count or sn in seen:
                raise ValueError('Body profile numbering must uniquely cover all one-based physical surfaces')
            seen.add(sn);i=sn-1;row=model['surface_rows'][i]
            for name in ('body_radius_mm','sphere_face_radius_mm','optical_sphere_curvature_mm_inv','clear_radius_mm'):
                value=profile.get(name)
                if isinstance(value,bool)or not isinstance(value,(int,float))or not np.isfinite(value):
                    raise ValueError(f'S{sn} requires finite explicit '+name)
            if float(profile['sphere_face_radius_mm'])!=self.face[i]:
                raise ValueError(f'S{sn} explicit sphere face differs from unchanged finite cutoff')
            if float(profile['optical_sphere_curvature_mm_inv'])!=float(row['curvature_mm_inv']):
                raise ValueError(f'S{sn} explicit curvature differs from optical sphere')
            clear=row.get('clear_radius_mm',row.get('aperture_radius_mm'))
            if clear is None or float(profile['clear_radius_mm'])!=float(clear):
                raise ValueError(f'S{sn} explicit CA differs from optical clear metadata')
            if profile.get('outer_annulus')!='plane at sphere-face-edge sag' or profile.get('imaging_rays_in_annulus')is not False:
                raise ValueError(f'S{sn} unsupported explicit annulus semantics')
            body=float(profile['body_radius_mm'])
            if body<self.face[i]:raise ValueError(f'S{sn} explicit body radius is smaller than its sphere face')
            declared[i]=body
        if seen!=set(range(1,self.count)):
            raise ValueError('Explicit body profiles omit a physical surface')
        self.declared_body=declared.copy()
        self.body=np.maximum(self.face,declared)
        start=None
        for medium in range(1,len(materials)):
            if materials[medium] and start is None:start=medium-1
            if not materials[medium] and start is not None:
                last=medium-1
                self.body[start:last+1]=float(np.max(np.maximum(self.face[start:last+1],declared[start:last+1])))
                start=None
        self.pairs=[]
        for lens,(front,rear) in enumerate(self.solids,1):
            radius=min(self.body[front],self.body[rear])
            self.pairs.append(self._pair('glass_full_OD',front,rear,radius,self.min_CT,
                                         lens=lens,bins=bins))
        for left,(front_a,rear_a) in enumerate(self.solids):
            for right,(front_b,rear_b) in enumerate(self.solids[left+1:],left+1):
                if rear_a==front_b:continue  # intended shared cemented interface
                radius=min(self.body[rear_a],self.body[front_b])
                self.pairs.append(self._pair('ordered_solid_separation',rear_a,front_b,
                    radius,self.min_air,left_lens=left+1,right_lens=right+1,bins=bins))
                if right==left+1:
                    self.pairs.append(dict(kind='separate_mechanical_rim',first=rear_a,last=front_b,
                        r_first=np.asarray([self.body[rear_a]]),r_last=np.asarray([self.body[front_b]]),
                        floor=self.min_rim,left_lens=left+1,right_lens=right+1))
        # A housing LENGTH constrains the complete optical-solid axial span,
        # not the absolute front/rear vertex reference planes. Every endpoint
        # pair avoids differentiating a nonsmooth min/max. A .2mm provisional
        # stack reserve is explicit; supports/threads still need separate room.
        if self.tube_start is not None:
            tube=[(a,b) for a,b in self.solids if a>=self.tube_start]
            for front,_ in tube:
                for _,rear in tube:
                    ra,rb=np.meshgrid([0.,self.body[front]],[0.,self.body[rear]],indexing='ij')
                    self.pairs.append(dict(kind='maximum_complete_tube_solid_span',first=front,last=rear,
                        r_first=ra.ravel(),r_last=rb.ravel(),floor=0.,
                        upper_span_mm=self.tube_axial_span_max_mm-self.tube_span_reserve_mm))
        self.row_count=sum(len(pair['r_first']) for pair in self.pairs)
        self.constraint_count=self.row_count
        self.last_rows=[]
        self._fixed_profile_identity=self._profile_identity(model)

    @staticmethod
    def _profile_identity(model):
        """Changing an aperture, shoulder or solid topology requires rebuilding."""
        rows=model['surface_rows']
        profiles=model['explicit_nominal_solid_profiles']
        for profile in profiles:
            surface=profile['surface']-1
            if profile['optical_sphere_curvature_mm_inv']!=rows[surface]['curvature_mm_inv']:
                raise ValueError('Explicit profile curvature differs from current optical face')
        fixed=dict(
            material_occupancy=[bool(value)for value in model['material_names']],
            tube_start=model.get('group_boundaries',{}).get('first_tube_surface_index'),
            rows=[{key:row.get(key)for key in ('conic','even_coefficients',
                'mechanical_radius_mm','clear_radius_mm','aperture_radius_mm')}for row in rows],
            profiles=[{key:value for key,value in profile.items()
                if key!='optical_sphere_curvature_mm_inv'}for profile in profiles])
        return json.dumps(fixed,sort_keys=True,allow_nan=False)

    def _pair(self,kind,first,last,radius,floor,*,bins=9,**metadata):
        r=np.unique(np.r_[np.linspace(0.,radius,bins),
            np.clip([self.face[first],self.face[last]],0.,radius)])
        return dict(kind=kind,first=first,last=last,r_first=r,r_last=r,
                    floor=floor,**metadata)

    @property
    def curvature_bound_radii(self):
        """Use these actual finite-face radii, not obsolete patent display radii."""
        return self.face.copy()

    def maximum_abs_curvature(self):
        return np.divide(np.sqrt(1.-self.minimum_domain_root**2),self.face,
                         out=np.full(self.count,np.inf),where=self.face>0.)

    def _sag(self,model,surface,radius):
        c=float(model['surface_rows'][surface]['curvature_mm_inv'])
        radius=np.minimum(np.asarray(radius,dtype=float),self.face[surface])
        if not np.isfinite(c) or not np.isfinite(radius).all() or np.any(radius<0.):raise ValueError('Nonfinite sphere/radius in explicit solid model')
        r2=radius*radius
        squared=1.-c*c*r2
        # Bound is also exposed to the optimizer. Numerical guard accepts the
        # exact boundary while refusing undefined/back-branch spherical faces.
        if np.any(squared < self.minimum_domain_root**2-1e-10):
            raise ValueError(f'Full spherical face domain root guard violated at S{surface+1}')
        root=np.sqrt(np.maximum(squared,0.))
        sag=c*r2/(1.+root)
        dc=r2/(root*(1.+root))
        return sag,dc[:,None]*self.cjac[surface]

    def constraints(self,model):
        z=np.asarray(model['vertices_mm'],dtype=float)
        if z.shape!=(self.count,)or not np.isfinite(z).all():raise ValueError('Geometry surface count/nonfinite vertices changed')
        if self._profile_identity(model)!=self._fixed_profile_identity:
            raise ValueError('Fixed solid profiles or topology changed; rebuild FullProfileGeometry')
        slack=[];jac=[];rows=[];minimum=np.inf
        for pair in self.pairs:
            first,last=pair['first'],pair['last']
            a,ja=self._sag(model,first,pair['r_first'])
            b,jb=self._sag(model,last,pair['r_last'])
            clearance=z[last]-z[first]+b-a
            margin=clearance-pair['floor']
            tangent=self.zjac[last]-self.zjac[first]+jb-ja
            if pair['kind']=='maximum_complete_tube_solid_span':
                clearance=pair['upper_span_mm']-clearance
                margin=clearance
                tangent=-tangent
            minimum=min(minimum,float(np.min(margin)))
            slack.extend(margin-self.guard)
            jac.extend(tangent)
            for j,value in enumerate(margin):
                row=dict(kind=pair['kind'],first_surface=first+1,last_surface=last+1,
                    radius_first_mm=float(pair['r_first'][j]),radius_last_mm=float(pair['r_last'][j]),
                    clearance_mm=float(clearance[j]),floor_mm=pair['floor'],
                    physical_margin_mm=float(value),guard_mm=self.guard)
                if pair['kind']=='maximum_complete_tube_solid_span':
                    row.update(actual_endpoint_span_mm=float(pair['upper_span_mm']-clearance[j]),
                        nominal_optical_solid_span_limit_mm=pair['upper_span_mm'],
                        housing_length_mm=self.tube_axial_span_max_mm,
                        reserved_stack_budget_mm=self.tube_span_reserve_mm,
                        source='explicit project housing-length allocation')
                rows.append(row)
        self.last_rows=rows
        return np.asarray(slack),np.asarray(jac),minimum

    def audit(self,model):
        slack,_,minimum=self.constraints(model)
        violations=[row for row in self.last_rows if row['physical_margin_mm']<0.]
        guarded=[row for row in self.last_rows if row['physical_margin_mm']<self.guard]
        return dict(status='passed' if not violations else 'invalid',
            nominal_full_profile_passed=not violations,guarded_feasible=not guarded,
            minimum_physical_margin_mm=minimum,minimum_guarded_slack_mm=float(np.min(slack)),
            constraints=self.row_count,violations=violations,
            fixed_face_cutoffs_mm=self.face.tolist(),fixed_finished_body_radii_mm=self.body.tolist(),
            full_axial_extents=self.extents(model),
            assumptions={'profile':'finite spherical face + outer planar annular mechanical shoulder',
                'all_optical_rays_must_fit_CA_separately':True,
                'shared_cemented_surface_gap_mm':0.,'min_CT_full_radial_mm':self.min_CT,
                'min_air_and_nonlocal_body_separation_mm':self.min_air,'min_rim_mm':self.min_rim,
                'guard_mm':self.guard,'minimum_domain_root':self.minimum_domain_root,
                'constraint_policy_version':'explicit_spherical_profiles_v1',
                'body_radius_source':'validated explicit inherited body, shared block max with unchanged sphere faces',
                'tube_housing_length_max_mm':self.tube_axial_span_max_mm,
                'tube_reserved_stack_budget_mm':self.tube_span_reserve_mm,
                'tube_optical_solid_span_limit_before_guard_mm':self.tube_axial_span_max_mm-self.tube_span_reserve_mm,
                'tube_span_provenance':'explicit project housing-length allocation',
                'nominal_radial_minimum':'all monotone-interval endpoints and face cutoffs included',
                'not_evaluated':['bevels','mounting','insertion','stress','tolerances','supplier capability']})

    def extents(self,model):
        """Report full solid extents; reference vertex planes are not body boxes.

        The public black-box interface planes are optical reference/vertex
        positions. Curved sag may cross them; no extra containment is inferred.
        A separate documented housing-span limit can be compared to this result.
        """
        z=np.asarray(model['vertices_mm'])
        pieces=[]
        for lens,(front,rear) in enumerate(self.solids,1):
            radius=min(self.body[front],self.body[rear])
            r=np.asarray([0.,radius])
            a,_=self._sag(model,front,r)
            b,_=self._sag(model,rear,r)
            kind=('window' if front==0 and rear==1 else 'tube' if self.tube_start is not None
                  and front>=self.tube_start else 'objective')
            low=float(z[front]+np.min(a));high=float(z[rear]+np.max(b))
            pieces.append(dict(lens=lens,group=kind,front_surface=front+1,rear_surface=rear+1,
                full_solid_z_min_mm=low,full_solid_z_max_mm=high,full_axial_span_mm=high-low,
                finished_diameter_mm=2.*radius))
        groups={}
        for kind in ('window','objective','tube'):
            selected=[row for row in pieces if row['group']==kind]
            if selected:
                low=min(row['full_solid_z_min_mm'] for row in selected)
                high=max(row['full_solid_z_max_mm'] for row in selected)
                groups[kind]=dict(full_solid_z_min_mm=low,full_solid_z_max_mm=high,
                    full_axial_span_mm=high-low,maximum_finished_diameter_mm=max(
                    row['finished_diameter_mm'] for row in selected))
        return dict(pieces=pieces,groups=groups,
            port_interpretation='Preserve optical reference/vertex interface planes; no undisclosed solid bounding box inferred',
            documented_tube_housing_length_mm=self.tube_axial_span_max_mm,
            reserved_stack_budget_mm=self.tube_span_reserve_mm,
            nominal_optical_solid_span_limit_mm=self.tube_axial_span_max_mm-self.tube_span_reserve_mm,
            tube_housing_span_within_limit=(groups['tube']['full_axial_span_mm']<=self.tube_axial_span_max_mm
                                          if 'tube' in groups else None),
            tube_span_with_reserve_passed=(groups['tube']['full_axial_span_mm']<=
                self.tube_axial_span_max_mm-self.tube_span_reserve_mm if 'tube' in groups else None),
            housing_scope='Axial optical-solid span diagnostic; mounting, spacers and clearances require separate engineering review')

    def describe(self):
        pairs=[]
        for pair in self.pairs:
            item=deepcopy(pair)
            item['r_first']=item['r_first'].tolist()
            item['r_last']=item['r_last'].tolist()
            pairs.append(item)
        return dict(face_cutoffs_mm=self.face.tolist(),body_radii_mm=self.body.tolist(),
                    pairs=pairs,row_count=self.row_count)
