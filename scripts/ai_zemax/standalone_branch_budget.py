"""Budgeted multi-fidelity branch continuation with explicit diversity slots."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _advance(advance,branch,state,additional_budget,target):
    local_branch,local_target=deepcopy(branch),deepcopy(target)
    frozen,target_hash=_digest(local_branch),_digest(local_target)
    result=advance(local_branch,deepcopy(state),additional_budget,local_target)
    if _digest(local_branch)!=frozen or _digest(local_target)!=target_hash:raise ValueError('Frozen branch/target identity was mutated by continuation')
    if not isinstance(result,dict)or type(result.get('valid'))is not bool:
        raise ValueError('Branch continuation must return an explicit boolean valid')
    if result['valid']and(not math.isfinite(float(result['score'])) or 'state'not in result):
        raise ValueError('A valid branch requires finite score and a resumable state')
    json.dumps(result,allow_nan=False)
    # A screened state is not an accepted optical design.
    result=dict(result);result['screening_only']=True
    return result


def run_budgeted_branches(branches, *, target, rung_budgets, advance, validate,
                           output_directory, reduction=2, workers=1,
                           executor='thread', explore_slots=1, seed=0):
    """Allocate ADDITIONAL work per rung; preserve each family and exploration.

    advance(branch,state,additional_budget,target) returns valid/score/state.
    validate(branch,state,target) alone can mark a final candidate passed.
    Branch payloads should contain frozen real glass identities/dispersion and
    a physical seed; callbacks must restore thick-lens geometry and all gates.
    Process mode needs importable top-level callbacks. Native API instances
    must not be shared across threads or processes; this scheduler opens none.
    """
    branches=deepcopy(list(branches));target=deepcopy(target)
    budgets=list(rung_budgets)
    ids=[b.get('id')for b in branches]
    if not branches or any(not isinstance(i,str)or not i for i in ids)or len(set(ids))!=len(ids):
        raise ValueError('Require nonempty uniquely named branch specifications')
    if (not budgets or any(type(b)is not int or b<1 for b in budgets)
        or type(reduction)is not int or reduction<2 or type(workers)is not int or workers<1
        or type(explore_slots)is not int or explore_slots<0 or executor not in ('thread','process')):
        raise ValueError('Invalid branch schedule/budgets/worker settings')
    folder=Path(output_directory);folder.mkdir(parents=True,exist_ok=False)
    states={i:None for i in ids};specs=dict(zip(ids,branches));active=ids.copy()
    rng=np.random.default_rng(seed);history=[];start=perf_counter();total=0
    target_hash=_digest(target);frozen_hashes={i:_digest(b)for i,b in specs.items()}
    pool_class=ThreadPoolExecutor if executor=='thread'else ProcessPoolExecutor
    with pool_class(max_workers=workers)as pool:
        for rung,budget in enumerate(budgets):
            futures={i:pool.submit(_advance,advance,specs[i],states[i],budget,target)for i in active}
            records=[]
            for i in active:
                try:
                    row=futures[i].result()
                    if row['valid']:states[i]=row['state']
                except Exception as error:
                    row=dict(valid=False,error=f'{type(error).__name__}: {error}',screening_only=True)
                records.append(dict(id=i,**row));total+=budget
            ranked=sorted((r for r in records if r['valid']),key=lambda r:(float(r['score']),r['id']))
            selected=[];explored=[]
            if rung==len(budgets)-1:
                selected=[r['id']for r in ranked]
            elif ranked:
                families={str(specs[r['id']].get('family','default'))for r in ranked}
                quota=max(len(families),math.ceil(len(ranked)/reduction))
                # Best member of each declared topology/material family.
                for family in sorted(families):
                    selected.append(next(r['id']for r in ranked if str(specs[r['id']].get('family','default'))==family))
                available=[r['id']for r in ranked if r['id']not in selected]
                reserve=min(explore_slots,max(0,quota-len(selected)),len(available))
                exploit=max(0,quota-len(selected)-reserve)
                selected.extend(available[:exploit]);available=available[exploit:]
                if reserve:
                    explored=list(rng.choice(available,size=reserve,replace=False));selected.extend(explored)
            item=dict(rung=rung,additional_budget=budget,records=records,selected=selected,
                      exploration_selected=explored,allocated_work=total)
            history.append(item)
            (folder/f'rung-{rung:02d}.json').write_text(json.dumps(item,indent=2,allow_nan=False)+'\n',encoding='utf8')
            active=selected
            if not active:break
    finals=[]
    for i in active:
        local_branch,local_state,local_target=deepcopy(specs[i]),deepcopy(states[i]),deepcopy(target)
        checked=validate(local_branch,local_state,local_target)
        if not isinstance(checked,dict)or type(checked.get('passed'))is not bool:
            raise ValueError('Independent branch validator must return explicit boolean passed')
        if _digest(local_branch)!=frozen_hashes[i]or _digest(local_target)!=target_hash:
            raise ValueError('Independent branch validator changed the frozen branch/target')
        if _digest(local_state)!=_digest(states[i])and 'validated_state'not in checked:
            raise ValueError('A polished/changed state must be returned explicitly as validated_state')
        checked=dict(checked);final_state=checked.pop('validated_state',states[i]);json.dumps(final_state,allow_nan=False)
        finals.append(dict(id=i,validation=checked,passed=checked['passed'],state=final_state,validated_state_sha256=_digest(final_state)))
    if _digest(target)!=target_hash or any(_digest(specs[i])!=frozen_hashes[i]for i in ids):
        raise ValueError('Frozen branch/target identity changed')
    result=dict(history=history,finalists=finals,any_passed=any(r['passed']for r in finals),
        allocated_work=total,equal_full_budget_work=len(branches)*sum(budgets),workers=workers,executor=executor,
        target_sha256=target_hash,branch_sha256=frozen_hashes,seconds=perf_counter()-start,
        global_optimum_certified=False,statistical_yield_estimated=False)
    (folder/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result
