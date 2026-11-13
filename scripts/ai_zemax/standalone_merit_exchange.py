"""Connect adaptive optical samples to budgeted native MF graph recompilation."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .standalone_constraint_exchange import run_constraint_exchange, _identity
from .standalone_merit_graph import MeritGraph, compact, write_native


class _BudgetExceeded(Exception): pass


def run_merit_exchange(rows, groups, active_samples, initial_q, *, protected_rows,
                       audit_identity, solve_compiled, audit, output_directory,
                       budget, strict_validator=None, **exchange_options):
    """Compile → solve → full fixed-domain audit → add/drop → recompile.

    ``groups`` contains {sample: JSON identity, rows: weighted source roots}.
    ``solve_compiled(q, rows, report, round_index)`` owns the physical solver.
    It may use native software or independent tracing; this module opens no
    application. Every new audit identity must have a corresponding row group;
    adapters must regenerate their source pool when new fields/waves are needed.
    """
    pool=MeritGraph(rows); registry={}; n=len(pool._rows);protected_rows=tuple(protected_rows)
    for group in groups:
        key=_identity(group['sample'])
        if key in registry:raise ValueError('Duplicate MF sample identity')
        refs=list(group['rows'])
        if not refs or any(type(i)is not int or not 1<=i<=n or pool._rows[i-1]['weight']<=0. for i in refs):
            raise ValueError('MF sample groups require nonempty valid weighted roots')
        registry[key]=refs
    if not registry:raise ValueError('MF sample registry cannot be empty')
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=False)
    state=dict(q=np.asarray(initial_q,dtype=float).copy(),reports=[],history=[])
    def solve(q,samples,iteration):
        roots=set()
        for sample in samples:
            key=_identity(sample)
            if key not in registry:raise ValueError('Generate native rows for this new sample before compiling')
            roots.update(registry[key])
        compiled,report=compact(pool,active_rows=sorted(roots),protected_rows=protected_rows,
                                audit_identity=audit_identity,budget=budget)
        folder=directory/f'round-{iteration:02d}';folder.mkdir()
        (folder/'compile.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
        state['reports'].append(report)
        if not report['output']['budget_passed']:raise _BudgetExceeded()
        write_native(folder/'WORKING.MF',compiled)
        trial,metadata=solve_compiled(q,compiled,deepcopy(report),iteration)
        state['q']=np.asarray(trial,dtype=float).copy()
        return trial,dict(physical_solver=metadata,MF_counts=report['output']['counts'],
                          protected_roots=len(report['protected_rows']))
    def fixed_audit(q):
        records=list(audit(q))
        if any(_identity(r['sample'])not in registry for r in records):
            raise ValueError('Audit found a sample without generated MF rows; rebuild the source pool')
        return records
    user_callback=exchange_options.pop('callback',None)
    def callback(item):
        state['history'].append(deepcopy(item))
        if user_callback:user_callback(item)
    try:
        result=run_constraint_exchange(solve,fixed_audit,active_samples,initial_q,callback=callback,**exchange_options)
    except _BudgetExceeded:
        result=dict(q=state['q'].tolist(),passed=False,termination='MF_budget_exhausted',history=state['history'],
                    continuous_domain_certified=False)
    result['working_domain_passed']=result['passed']
    result['full_audit_identity']=audit_identity
    result['compiled_rounds']=[r['output']['counts']for r in state['reports']]
    result['final_acceptance']=dict(status='not_run',passed=None)
    if result['passed'] and strict_validator is not None:
        final=strict_validator(np.array(result['q']))
        if not isinstance(final,dict)or type(final.get('passed'))is not bool:
            raise ValueError('Final independent validator must return an explicit boolean passed')
        result['final_acceptance']=dict(final,status='complete')
        result['passed']=final['passed']
    else:
        # A work-domain audit is not authority for final design acceptance.
        result['passed']=False
    (directory/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result
