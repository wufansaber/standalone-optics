"""Build an explicit compact MF working set from the frozen green8 row graphs.

This is a case adapter, not a universal wavelength/field prescription. Its
working set is deliberately smaller than acceptance; re-enrich from the fixed
independent audit with run_constraint_exchange before accepting a new design.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from ai_zemax.standalone_merit_graph import compact, write_native
from ai_zemax.standalone_merit_sphere import spherical_support


def build(source, output):
    input_hashes = {}
    def load(name):
        path = source/name
        data = path.read_bytes() if path.exists() else gzip.decompress((source/(name+'.gz')).read_bytes())
        input_hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    full, core = load('MF_FULL_GRAPH.json'), load('MF_CORE_GRAPH.json')
    geometry, focus, wfe = [load(n) for n in ('MF_GEOMETRY_REFS.json','MF_FOCUS_REFS.json','MF_WFE_REFS.json')]
    model = load('SPHERICAL_PROFILES.json')
    formal = load('FORMAL_FOCUS_ACCEPTANCE.json')
    if formal['model_sha256'] != wfe['source_model_file_sha256']:
        raise ValueError('Formal focus evidence must identify the same frozen prescription')
    if full[:len(core)] != core:
        raise ValueError('Core graph must be the exact prefix of the full graph')
    support = spherical_support(full, geometry['geometry'], geometry['domain'], model['explicit_nominal_solid_profiles'])
    weighted = {i for i,r in enumerate(full,1) if r['weight'] > 0.}
    protected = set(support['protected_rows'])
    for key in ('vertices','ports','cardinals','face_witnesses'):
        protected.update(i for rec in geometry[key] for i in rec['bound_rows'])
    dref = load('MF_OBJECTIVE_D_REFS.json')['value_row']
    d_bounds = {i for i,r in enumerate(full,1) if r['operand'] in {'OPGT','OPLT'} and r['params'][0] == dref and r['weight'] > 0.}
    if len(d_bounds) != 2: raise ValueError('Require both original objective-D bounds')
    protected.update(d_bounds)
    known = set(support['source_rows']) | protected
    for key in ('first_order','lateral_color'):
        known.update(i for r in geometry[key] for i in r['bound_rows'])
    known.update(c[k]['upper_bound_row'] for c in wfe['conditions'] for k in ('RWCE','RWRE') if c[k].get('upper_bound_row'))
    known.update(i for key in ('LCA_pair_bounds','FC595_fields') for r in focus[key] for i in r['bound_rows'])
    if weighted-known:
        raise ValueError('Unclassified weighted rows cannot be dropped by the case adapter')

    fields = [0., .3125, .625, .9375, 1.25]
    close = lambda h: any(abs(float(h)-v) < 1e-12 for v in fields)
    active = set()
    for key in ('first_order','lateral_color'):
        active.update(i for r in geometry[key] if close(r['field_y_mm']) for i in r['bound_rows'] if i in weighted)
    for rec in wfe['conditions']:
        if close(rec['field_mm']) and rec['role'] == 'green_hard_cap':
            active.update(rec[k]['upper_bound_row'] for k in ('RWCE','RWRE'))
    axial = formal['axial_rows']
    waves = sorted({570,580,595,610,620,
        min(axial,key=lambda r:r['object_full2D_shift_mm'])['wavelength_nm'],
        max(axial,key=lambda r:r['object_full2D_shift_mm'])['wavelength_nm']})
    for rec in focus['LCA_pair_bounds']:
        if rec['i_nm'] in waves and rec['j_nm'] in waves:
            active.update(rec['bound_rows'])
    fc_fields = [0.,1.25]+[r['field_mm'] for r in formal['FC_local_extrema']]
    for h in fc_fields:
        rec = min(focus['FC595_fields'],key=lambda r:abs(r['field_radius_mm']-h))
        if abs(rec['field_radius_mm']-h)>1e-10:
            raise ValueError('The source MF must contain the frozen local FC worst point')
        active.update(rec['bound_rows'])
    # Keep any source weighted row reached for arithmetic as a zero-weight
    # dependency unless it is explicitly an active/protected root.
    identity = ('green8-independent-full-audit:model-sha256:'+wfe['source_model_file_sha256']+
                ':scope-sha256:'+wfe['scope_sha256']+':focus-definition-sha256:'+formal['definition_sha256']+
                ':source-graph-sha256:'+input_hashes['MF_FULL_GRAPH.json'])
    selection = dict(active_rows=sorted(active),protected_rows=sorted(protected),audit_identity=identity)
    output.mkdir(parents=True,exist_ok=False)
    save = lambda name,data: (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')
    summary = dict(schema='green8_mf_budget_v1',source_model_sha256=wfe['source_model_file_sha256'],
        scope_sha256=wfe['scope_sha256'], working_fields_mm=fields,working_LCA_wavelengths_nm=waves,
        working_FC_fields_mm=fc_fields,
        complete_acceptance_changed=False,new_design_accepted=False,
        input_sha256=input_hashes,
        full_body_and_all15wave_physical_audit_required=True,
        independent_WFE65_and_local_extrema_required=True,formal_LCA51_FC65_plus_local_required=True,
        manufacturing_release=False,artifacts={})
    core_selection = dict(selection,active_rows=[i for i in selection['active_rows'] if i <= len(core)])
    for name, graph, options in [('CORE_EQUIVALENT',core,{}),('FULL_EQUIVALENT',full,{}),
                                  ('WORKING_CORE',core,dict(**core_selection,budget={'rows':10000,'wavefront_queries':80})),
                                  ('WORKING',full,dict(**selection,budget={'rows':20000,'wavefront_queries':80}))]:
        result, report = compact(graph,**options)
        save(name+'_REPORT.json',report)
        if not report['output']['budget_passed']:
            raise ValueError('Explicit case MF budget exceeded; inspect report without relaxing acceptance')
        write_native(output/(name+'.MF'),result)
        summary['artifacts'][name] = {k:report[k] for k in ('mode','input','output','merged_rows','omitted_rows','retained_weighted_rows','full_objective_preserved')}
        summary['artifacts'][name]['sha256'] = hashlib.sha256((output/(name+'.MF')).read_bytes()).hexdigest()
        print(name,report['input']['counts']['rows'],'->',len(result),flush=True)
    save('WORKING_SELECTION.json',selection); save('SPHERICAL_SUPPORT.json',support);save('SUMMARY.json',summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.out)
