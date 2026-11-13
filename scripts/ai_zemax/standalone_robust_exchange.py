"""Worst-scenario exchange for actual physical tolerances and shared controls."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .standalone_constraint_exchange import run_constraint_exchange


def run_robust_exchange(scenarios, initial_scenarios, initial_design, *,
                         problem_factory, audit_scenario, output_directory,
                         solve_options, protected_scenarios=(), **exchange_options):
    """Refine only active scenarios, then independently audit EVERY scenario.

    problem_factory(selected_payload_mapping) returns ScenarioProblem. Its
    design coordinates must be identical for every subset. audit_scenario
    (design,id,payload,controls) traces the complete frozen final optical domain
    and returns gates/valid. An inactive scenario uses its zero-or-bound-clipped
    feasible control; after promotion it gets its own bounded shared control.
    No measurement gets an independent focus or hidden compensator.
    """
    scenarios=deepcopy(scenarios)
    if not scenarios or not set(initial_scenarios)<=set(scenarios) or not set(protected_scenarios)<=set(scenarios):
        raise ValueError('Initial/protected scenarios must belong to a nonempty fixed scenario domain')
    folder=Path(output_directory);folder.mkdir(parents=True,exist_ok=False)
    template=problem_factory(deepcopy(scenarios))
    design=np.asarray(initial_design,dtype=float)
    if design.shape!=(template.design_size,):raise ValueError('Initial shared design has wrong dimension')
    controls={key:np.clip(np.zeros(sl.stop-sl.start),template.bounds[sl,0],template.bounds[sl,1])
              for key,sl in template.controls.items()}
    previous_controls={}
    def solve_round(q,samples,iteration):
        previous_controls.clear();previous_controls.update({key:value.copy()for key,value in controls.items()})
        chosen=[s['scenario']for s in samples]
        problem=problem_factory({key:deepcopy(scenarios[key])for key in chosen})
        if not np.array_equal(problem.design_bounds,template.design_bounds):
            raise ValueError('Scenario enrichment changed the shared design coordinates/bounds')
        for key,sl in problem.controls.items():
            if not np.array_equal(problem.bounds[sl],template.bounds[template.controls[key]]):
                raise ValueError('Scenario enrichment changed a compensator definition')
        initial=np.concatenate([q,*[controls[key]for key in chosen]])
        options=dict(solve_options);derivatives=options.pop('analytic',False)
        if derivatives:options['jacobian_provider']=problem.jacobian
        result=problem.solve(folder/f'round-{iteration:02d}',initial_q=initial,**options)
        expanded=np.asarray(result['q'])
        for key,sl in problem.controls.items():controls[key]=expanded[sl].copy()
        return expanded[:problem.design_size],result
    def audit(q):
        records=[]
        for key,payload in scenarios.items():
            result=audit_scenario(q.copy(),key,deepcopy(payload),controls[key].copy())
            if not isinstance(result,dict)or 'gates'not in result:raise ValueError('Full scenario audit must return explicit gates')
            records.append(dict(result,sample={'scenario':key}))
        return records
    def full_domain_merit(records):
        slacks=[]
        for record in records:
            if not record.get('valid',True):return 1e12
            for value in record['gates'].values():
                value=float(np.min(value))
                if not np.isfinite(value):return 1e12
                slacks.append(value)
        worst=min(slacks)
        return 1.-worst if worst<0. else-worst
    user_callback=exchange_options.pop('callback',None)
    def callback(item):
        if not item['candidate_accepted']:
            controls.clear();controls.update({key:value.copy()for key,value in previous_controls.items()})
        if user_callback:user_callback(item)
    exchange_options.setdefault('audit_merit',full_domain_merit)
    result=run_constraint_exchange(solve_round,audit,[dict(scenario=i)for i in initial_scenarios],design,
        protected_samples=[dict(scenario=i)for i in protected_scenarios],promote_worst=True,callback=callback,**exchange_options)
    result['scenario_controls']={k:v.tolist()for k,v in controls.items()}
    result['independent_scenario_count']=len(scenarios)
    result['statistical_yield_estimated']=False
    result['manufacturing_release']=False
    result['robust_margin_optimum_certified']=False
    (folder/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result
