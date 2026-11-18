"""Certify coaxial conformal annular seats on the two real facing surfaces."""
import math
import numpy as np
from .standalone_meniscus import _curvature_bound


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def audit_spacers(surfaces,lenses,clear,mechanical,config,issue,lands=None):
    keys={'interfaces','minimum_width_mm','chip_mm','bevel_mm','minimum_gap_mm','maximum_gap_mm','gap_budget_mm'}
    if not isinstance(config,dict) or set(config)!=keys:
        raise ValueError('Spacer policy must specify all contact dimensions, budgets and adjacent interface roles')
    if any(not _finite(config[k]) or config[k]<0 for k in keys-{'interfaces'}):
        raise ValueError('Invalid spacer dimension')
    if config['minimum_width_mm']<=0 or config['maximum_gap_mm']<=config['minimum_gap_mm']:
        raise ValueError('Invalid spacer width or limits')
    interfaces=config['interfaces']
    if not isinstance(interfaces,list) or len(interfaces)!=len(lenses)-1:
        raise ValueError('Declare a spacer or open interface for every adjacent pair')
    rows=[]
    for n,(left,right,cfg) in enumerate(zip(lenses,lenses[1:],interfaces),1):
        if not isinstance(cfg,dict) or type(cfg.get('left_lens')) is not int or cfg['left_lens']!=n:
            raise ValueError('Spacer declarations must follow the physical lens order')
        if cfg.get('kind')=='open_group' and set(cfg)=={'left_lens','kind','function'} and cfg['function']:
            rows.append(dict(cfg));continue
        if set(cfg)!={'left_lens','kind','inner_radius_mm','outer_radius_mm'} or cfg['kind'] not in ('conformal_annulus','flat_annulus'):
            raise ValueError('Unsupported spacer profile or missing annular contact radii')
        ri,ro=cfg['inner_radius_mm'],cfg['outer_radius_mm']
        if not all(_finite(v) for v in (ri,ro)) or not 0<=ri<ro:
            raise ValueError('Invalid annular contact radii')
        a,b=left['rear_surface'],right['front_surface']
        if a>=b:raise ValueError('An annular air spacer requires separate glass bodies')
        row=dict(cfg,front_contact_surface=a,rear_contact_surface=b)
        rows.append(row)
        if any(clear.get(i) is None or mechanical.get(i) is None for i in (a,b)):
            issue('missing','spacer_face_radius_missing','Both finite glass contact profiles are required',left_lens=n)
            continue
        ca=max(clear[a],clear[b]);od=min(mechanical[a],mechanical[b])
        if ri<ca+config['chip_mm']-1e-10 or ro>od-config['bevel_mm']+1e-10 or ro-ri<config['minimum_width_mm']-1e-10:
            issue('violations','spacer_contact_outside_usable_glass','Both full-width contacts must lie on glass outside CA and bevel',left_lens=n)
            continue
        if cfg['kind']=='flat_annulus':
            if not lands or not all(str(i) in lands for i in (a,b)):
                issue('missing','spacer_flat_land_missing','Both facing glass surfaces require actual ground planes',left_lens=n)
                continue
            f,t=[lands[str(i)] for i in (a,b)]
            if ri<max(f['flat_start_mm'],t['flat_start_mm']) or ro>min(f['bevel_start_mm'],t['bevel_start_mm']):
                issue('violations','spacer_outside_flat_land','Entire contact annulus must lie on both flat lands',left_lens=n)
                continue
            gap=t['plane_z_mm']-f['plane_z_mm']
            row.update(minimum_gap_lower_mm=gap,maximum_gap_upper_mm=gap,inner_gap_mm=gap,outer_gap_mm=gap,interpolation_error_mm=0.)
            if gap-config['gap_budget_mm']<config['minimum_gap_mm'] or gap+config['gap_budget_mm']>config['maximum_gap_mm']:
                issue('violations','spacer_gap_out_of_range','Parallel plane spacing exceeds allocated interval',left_lens=n)
            continue
        grid=np.linspace(ri,ro,257)
        s,t=surfaces[a-1],surfaces[b-1]
        z1=s.vertex_mm+s.sag_and_gradient(grid**2)[0]
        z2=t.vertex_mm+t.sag_and_gradient(grid**2)[0]
        err=(_curvature_bound(s,ro)+_curvature_bound(t,ro))*(grid[1]-grid[0])**2/8
        delta=z2-z1
        lo,hi=float(min(delta)-err),float(max(delta)+err)
        row.update(minimum_gap_lower_mm=lo,maximum_gap_upper_mm=hi,
                   inner_gap_mm=float(delta[0]),outer_gap_mm=float(delta[-1]),interpolation_error_mm=float(err))
        if lo-config['gap_budget_mm']<config['minimum_gap_mm'] or hi+config['gap_budget_mm']>config['maximum_gap_mm']:
            issue('violations','spacer_gap_out_of_range','Actual annular seat-to-seat distance exceeds its allocated bounds',left_lens=n,**{'lower_mm':lo,'upper_mm':hi})
    return rows
