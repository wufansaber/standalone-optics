"""Check pre-grind full-diameter edges and subtractive plane annular lands."""
import math
import numpy as np
from .standalone_meniscus import _curvature_bound


def _raw_bounds(surface, start, end):
    """Bound cap height and meridional slope, including interior aspheric extrema."""
    r = np.linspace(start, end, 1025)
    sag, gradient = surface.sag_and_gradient(r*r)
    heights = surface.vertex_mm + sag
    h = (end-start)/(len(r)-1)
    curvature = _curvature_bound(surface, end)
    interpolation = curvature*h*h/8
    # Convert d(sag)/d(r squared) to the meridional derivative.
    slope = max(abs(2*r*gradient)) + curvature*h/2
    return float(min(heights)-interpolation), float(max(heights)+interpolation), float(slope)



def _finished_profile(surface, land, radius, side, radii):
    raw_slope = _raw_bounds(surface, 0., radius)[2]
    if land is None:
        return surface.vertex_mm + surface.sag_and_gradient(radii**2)[0], raw_slope
    start, flat, bevel, plane = (land[k] for k in ('blend_start_mm', 'flat_start_mm', 'bevel_start_mm', 'plane_z_mm'))
    raw = surface.vertex_mm + surface.sag_and_gradient(radii**2)[0]
    width=flat-start
    t=np.clip((radii-start)/width,0.,1.)
    if land.get('transition')=='tangent_ramp':
        # A piecewise-linear slope law with integrated end tangencies.
        f=.06
        z0,derivative=surface.sag_and_gradient(start**2)
        z0+=surface.vertex_mm
        s0=2*start*derivative
        mid=((plane-z0)/width-f*s0/2)/(1-f)
        u=np.minimum(t,f)
        v=np.clip(t-f,0.,1-2*f)
        w=np.maximum(t-1+f,0.)
        integral=s0*u+(mid-s0)*u*u/(2*f)+mid*v+mid*w-mid*w*w/(2*f)
        result=np.where(radii<start,raw,z0+width*integral)
        result=np.where(radii>=flat,plane,result)+side*np.maximum(radii-bevel,0.)
        return result,max(1.,raw_slope,abs(mid),abs(s0))
    blend=t*t*(3.-2.*t)
    result=(1.-blend)*raw+blend*plane+side*np.maximum(radii-bevel,0.)
    lo,hi,_=_raw_bounds(surface,start,flat)
    delta=max(abs(plane-lo),abs(plane-hi))
    rr=np.linspace(start,flat,2049)
    zz,grad=surface.sag_and_gradient(rr**2)
    tt=(rr-start)/width
    derivative=(1-tt**2*(3-2*tt))*(2*rr*grad)+6*tt*(1-tt)*(plane-surface.vertex_mm-zz)/width
    k=_curvature_bound(surface,flat)+3*raw_slope/width+6*delta/width**2
    return result,max(1.,raw_slope,float(max(abs(derivative)))+k*(rr[1]-rr[0])/2)



def _check_finished_glass(surfaces, lens, lands, radius, row, issue):
    """Bound thickness through both transitions, flat lands and 45-degree rim bevels."""
    indices = (lens['front_surface'], lens['rear_surface'])
    definitions = [lands.get(str(i)) for i in indices]
    knots = [d[k] for d in definitions if d is not None for k in ('blend_start_mm', 'flat_start_mm', 'bevel_start_mm')]
    limits = lens['effective_limits']
    radial_floor = limits.get('minimum_radial_thickness_mm', 0.) + limits.get('radial_thickness_budget_mm', 0.)
    req = lens.get('meniscus_conditions', {})
    edge_floor = max(radial_floor, req.get('minimum_edge_thickness_mm', 0.) + req.get('edge_thickness_budget_mm', 0.))
    face_floor = max(req.get('minimum_face_distance_mm', 0.), 2.*radius*req.get('minimum_face_to_diameter_ratio', 0.)) + req.get('face_distance_budget_mm', 0.)
    for points in (257, 1025, 4097):
        r = np.unique(np.r_[np.linspace(0., radius, points), knots])
        (front, sf), (back, sb) = [_finished_profile(surfaces[i-1], d, radius, side, r)
                                   for i, d, side in zip(indices, definitions, (1, -1))]
        step = max(np.diff(r))
        vertical = back-front
        radial = float(min(vertical) - (sf+sb)*step/2.)
        # Lipschitz graphs separated axially by h are separated in the plane
        # by at least h / sqrt(1+slope**2), using either graph's slope.
        nearest = radial/math.hypot(1., min(sf, sb))
        if nearest < face_floor and radial >= radial_floor:
            sampled = min(float(np.min((x[:,None]-r)**2+(z[:,None]-back)**2))
                          for x,z in zip(np.array_split(r, (len(r)+63)//64),
                                         np.array_split(front, (len(r)+63)//64)))
            nearest = max(nearest, math.sqrt(sampled) - step*(math.hypot(1.,sf)+math.hypot(1.,sb))/2.)
        if radial >= radial_floor and nearest >= face_floor:
            break
    slope_limit=limits.get('maximum_surface_slope')
    row['finished_surface_slope_upper']=float(max(sf,sb))
    if slope_limit is not None and max(sf,sb)>slope_limit:
        issue('violations','finished_profile_slope_unqualified',
              'Finished optical, transition and bevel profiles exceed the specified maximum slope',lens=lens['lens'])
    row.update(finished_rim_mm=float(vertical[-1]), finished_minimum_axial_lower_mm=radial,
               finished_face_distance_lower_mm=nearest)
    if vertical[-1] < edge_floor or radial < radial_floor or nearest < face_floor:
        issue('violations', 'finished_glass_thickness_uncertified',
              'Finished blend, flat and bevel must retain the specified radial, edge and face thickness budgets',
              lens=lens['lens'])


def audit_extended_edges(surfaces,lenses,clear,mechanical,config,issue):
    keys={'maximum_edge_mm','thickness_budget_mm','tube_lenses','common_diameter_mm','lands','minimum_platform_mm','clear_reserve_mm'}
    if not isinstance(config,dict) or set(config)-{'additional_common_groups'}!=keys:
        raise ValueError('Incomplete extended-edge and ground-land specification')
    scalar=keys-{'tube_lenses','lands'}
    if any(type(config[k]) not in (int,float) or not math.isfinite(config[k]) or config[k]<0 for k in scalar):
        raise ValueError('Extended-edge dimensions must be nonnegative and finite')
    if any(config[k] == 0 for k in ('common_diameter_mm','maximum_edge_mm','minimum_platform_mm')):
        raise ValueError('Diameter, upper edge limit and minimum platform width must be positive')
    ids=config['tube_lenses'];lands=config['lands']
    if not isinstance(ids,list) or not ids or not all(type(i) is int and 1<=i<=len(lenses) for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Specify distinct physical lens numbers for the common-diameter group')
    diameters = dict.fromkeys(ids,config['common_diameter_mm'])
    extra = config.get('additional_common_groups',[])
    if not isinstance(extra,list):
        raise ValueError('Additional common-diameter groups must be a list')
    for group in extra:
        if not isinstance(group,dict) or set(group)!={'lenses','diameter_mm'}:
            raise ValueError('Specify a diameter and lens numbers for every additional group')
        members=group['lenses'];diameter=group['diameter_mm']
        if type(diameter) not in (int,float) or not math.isfinite(diameter) or diameter<=0:
            raise ValueError('Common diameter must be finite and positive')
        if not isinstance(members,list) or not members:
            raise ValueError('Additional common-diameter group has no lenses')
        for number in members:
            if type(number) is not int or not 1<=number<=len(lenses) or number in diameters:
                raise ValueError('Common-diameter groups require distinct physical lens numbers')
            diameters[number]=diameter
    needed={str(l[k]) for l in lenses if l['lens'] in diameters for k in ('front_surface','rear_surface')}
    known={str(l[k]) for l in lenses for k in ('front_surface','rear_surface')}
    if not isinstance(lands,dict) or not needed<=set(lands)<=known:
        raise ValueError('Define both lands on each group member; additional lands must belong to real glass')
    report=[]
    for l in lenses:
        num=l['lens'];a,b=l['front_surface'],l['rear_surface'];common=num in diameters
        if any(mechanical.get(i) is None or clear.get(i) is None for i in (a,b)):
            issue('missing','extension_domain_missing','Both optical and mechanical radii are required',lens=num)
            continue
        radius=diameters[num]/2 if common else min(mechanical[a],mechanical[b])
        if abs(mechanical[a]-mechanical[b])>1e-9 or (common and abs(mechanical[a]-radius)>1e-9):
            issue('violations','inconsistent_finished_diameter','Both physical faces must reach the specified finished radius',lens=num)
        sphere_z=[]
        for n in (a,b):
            s=surfaces[n-1]
            if 1-(1+s.conic)*(s.curvature_mm_inv*radius)**2<=0:
                issue('violations','extended_surface_undefined','Declared cap cannot extend to common radius',lens=num,surface=n)
                break
            sphere_z.append(s.vertex_mm+float(s.sag_and_gradient(radius**2)[0]))
        if len(sphere_z)!=2:continue
        et=sphere_z[1]-sphere_z[0]
        row=dict(lens=num,radius_mm=radius,extended_edge_mm=et);report.append(row)
        if not math.isfinite(et) or et<=0 or et+config['thickness_budget_mm']>config['maximum_edge_mm']:
            issue('violations','extended_edge_thickness','Uncut extended faces cross or exceed maximum edge thickness including budget',lens=num,value=et)
        if not any(str(i) in lands for i in (a,b)):continue
        planes=[]
        valid_lands=True
        for n,sign in ((a,1),(b,-1)):
            if str(n) not in lands:
                continue
            s=surfaces[n-1];g=lands[str(n)]
            if not isinstance(g,dict) or set(g)-{'transition'}!={'blend_start_mm','flat_start_mm','bevel_start_mm','plane_z_mm'} or g.get('transition','smoothstep') not in ('smoothstep','tangent_ramp') or any(type(v) not in (int,float) or not math.isfinite(v) for k,v in g.items() if k!='transition'):
                raise ValueError('Malformed ground annular land')
            start,flat,bevel,plane=[g[k] for k in ('blend_start_mm','flat_start_mm','bevel_start_mm','plane_z_mm')]
            if not clear[n]+config['clear_reserve_mm']<=start<flat<bevel<radius or bevel-flat<config['minimum_platform_mm']-1e-9:
                valid_lands=False
                issue('violations','insufficient_flat_land','A full flat platform outside the clear zone and before the bevel is required',surface=n)
            if not 0 <= start < radius:
                planes.append(plane)
                continue
            if not s.conic and not any(s.even_coefficients):
                z0=s.vertex_mm+s.sag_and_gradient(start*start)[0]
                z1=sphere_z[0 if n==a else 1]
                low,high=min(z0,z1),max(z0,z1)
            else:
                low,high,_=_raw_bounds(s,start,radius)
            if (plane-high if sign==1 else low-plane) < -1e-10:
                valid_lands=False
                issue('violations','land_outside_stock','The prescribed plane requires adding material to the parent cap',surface=n)
            if g.get('transition')=='tangent_ramp' and valid_lands:
                width=flat-start
                f=.06
                z0,ds=s.sag_and_gradient(start**2)
                s0=2*start*ds
                mid=((plane-s.vertex_mm-z0)/width-.5*f*s0)/(1-f)
                curvature=_curvature_bound(s,flat)
                # From the tangent contact at start, the first ramp curves entirely
                # into available stock. Bound all subsequent interpolation intervals.
                knot=start+f*width
                r=np.unique(np.r_[np.linspace(knot,flat,2049),flat-f*width])
                cut=_finished_profile(s,g,radius,sign,r)[0]
                stock=sign*(cut-s.vertex_mm-s.sag_and_gradient(r*r)[0])
                error=(abs(mid)/(f*width)+curvature)*max(np.diff(r))**2/8
                if sign*(mid-s0)/(f*width)<curvature or min(stock)<error-1.e-10:
                    valid_lands=False
                    issue('violations','ramp_outside_stock','Tangent transition cannot be certified by subtractive grinding',surface=n)
            planes.append(plane)
        if valid_lands:
            _check_finished_glass(surfaces,l,lands,radius,row,issue)
        if len(planes)<2:
            continue
        row['finished_land_thickness_mm']=planes[1]-planes[0]
        if planes[1]-planes[0]<=0 or planes[1]-planes[0]+config['thickness_budget_mm']>config['maximum_edge_mm']:
            issue('violations','finished_edge_thickness','Opposing ground lands leave invalid glass thickness',lens=num)
    return report
