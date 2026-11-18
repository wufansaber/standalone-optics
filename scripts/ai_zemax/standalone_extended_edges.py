"""Check pre-grind full-diameter edges and subtractive plane annular lands."""
import math
import numpy as np


def _finished_profile(surface, land, radius, side, radii):
    start, flat, bevel, plane = (land[k] for k in ('blend_start_mm', 'flat_start_mm', 'bevel_start_mm', 'plane_z_mm'))
    raw = surface.vertex_mm + surface.sag_and_gradient(radii**2)[0]
    t = np.clip((radii-start)/(flat-start), 0., 1.)
    blend = t*t*(3.-2.*t)
    result = (1.-blend)*raw + blend*plane + side*np.maximum(radii-bevel, 0.)
    c = abs(surface.curvature_mm_inv)
    raw_slope = c*radius/math.sqrt(1.-(c*radius)**2)
    endpoints = surface.vertex_mm + surface.sag_and_gradient(np.array([start, flat])**2)[0]
    slope_bound = max(1., raw_slope + 1.5*np.max(abs(plane-endpoints))/(flat-start))
    return result, slope_bound


def _check_finished_glass(surfaces, lens, lands, radius, row, issue):
    """Bound thickness through both transitions, flat lands and 45-degree rim bevels."""
    indices = (lens['front_surface'], lens['rear_surface'])
    definitions = [lands[str(i)] for i in indices]
    knots = [d[k] for d in definitions for k in ('blend_start_mm', 'flat_start_mm', 'bevel_start_mm')]
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
    row.update(finished_rim_mm=float(vertical[-1]), finished_minimum_axial_lower_mm=radial,
               finished_face_distance_lower_mm=nearest)
    if vertical[-1] < edge_floor or radial < radial_floor or nearest < face_floor:
        issue('violations', 'finished_glass_thickness_uncertified',
              'Finished blend, flat and bevel must retain the specified radial, edge and face thickness budgets',
              lens=lens['lens'])


def audit_extended_edges(surfaces,lenses,clear,mechanical,config,issue):
    keys={'maximum_edge_mm','thickness_budget_mm','tube_lenses','common_diameter_mm','lands','minimum_platform_mm','clear_reserve_mm'}
    if not isinstance(config,dict) or set(config)!=keys:
        raise ValueError('Incomplete extended-edge and ground-land specification')
    scalar=keys-{'tube_lenses','lands'}
    if any(type(config[k]) not in (int,float) or not math.isfinite(config[k]) or config[k]<0 for k in scalar):
        raise ValueError('Extended-edge dimensions must be nonnegative and finite')
    if any(config[k] == 0 for k in ('common_diameter_mm','maximum_edge_mm','minimum_platform_mm')):
        raise ValueError('Diameter, upper edge limit and minimum platform width must be positive')
    ids=config['tube_lenses'];lands=config['lands']
    if not isinstance(ids,list) or not ids or not all(type(i) is int and 1<=i<=len(lenses) for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Specify distinct physical lens numbers for the common-diameter group')
    needed={str(l[k]) for l in lenses if l['lens'] in ids for k in ('front_surface','rear_surface')}
    if not isinstance(lands,dict) or set(lands)!=needed:
        raise ValueError('Define both ground lands on every common-diameter lens')
    report=[]
    for l in lenses:
        num=l['lens'];a,b=l['front_surface'],l['rear_surface'];common=num in ids
        if any(mechanical.get(i) is None or clear.get(i) is None for i in (a,b)):
            issue('missing','extension_domain_missing','Both optical and mechanical radii are required',lens=num)
            continue
        radius=config['common_diameter_mm']/2 if common else min(mechanical[a],mechanical[b])
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
        if not common:continue
        planes=[]
        valid_lands=True
        for n,sign in ((a,1),(b,-1)):
            s=surfaces[n-1];g=lands[str(n)]
            if not isinstance(g,dict) or set(g)!={'blend_start_mm','flat_start_mm','bevel_start_mm','plane_z_mm'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in g.values()):
                raise ValueError('Malformed ground annular land')
            start,flat,bevel,plane=[g[k] for k in ('blend_start_mm','flat_start_mm','bevel_start_mm','plane_z_mm')]
            if s.conic or any(s.even_coefficients):
                raise ValueError('Ground land certification supports a spherical or plane parent cap')
            if not clear[n]+config['clear_reserve_mm']<=start<flat<bevel<radius or bevel-flat<config['minimum_platform_mm']-1e-9:
                valid_lands=False
                issue('violations','insufficient_flat_land','A full flat platform outside the clear zone and before the bevel is required',surface=n)
            # A spherical meridian is monotone. Both endpoint inequalities
            # certify the required stock over the entire grinding annulus.
            if not 0 <= start < radius:
                planes.append(plane)
                continue
            z0=s.vertex_mm+s.sag_and_gradient(start*start)[0]
            if min(sign*(plane-z0),sign*(plane-sphere_z[0 if n==a else 1])) < -1e-10:
                valid_lands=False
                issue('violations','land_outside_stock','The prescribed plane requires adding material to the parent cap',surface=n)
            planes.append(plane)
        if valid_lands:
            _check_finished_glass(surfaces,l,lands,radius,row,issue)
        row['finished_land_thickness_mm']=planes[1]-planes[0]
        if planes[1]-planes[0]<=0 or planes[1]-planes[0]+config['thickness_budget_mm']>config['maximum_edge_mm']:
            issue('violations','finished_edge_thickness','Opposing ground lands leave invalid glass thickness',lens=num)
    return report
