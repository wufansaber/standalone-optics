"""Adaptive complete optical samples, fixed audits and a full-domain filter."""
from copy import deepcopy
import json
import math
from time import perf_counter
import numpy as np


def _identity(sample):
    if not isinstance(sample,dict)or not sample:raise ValueError('Each sample must be a nonempty JSON object')
    return json.dumps(sample,sort_keys=True,separators=(',',':'),allow_nan=False)


class _AuditDomain:
    def __init__(self,allow_discovered):
        self.allow_discovered=allow_discovered;self.domain=None;self.layout=None;self.discovered={}
    def inspect(self,records,active):
        if not records:raise ValueError('Independent audit returned no samples')
        identities=[_identity(r['sample'])for r in records]
        if len(set(identities))!=len(identities):raise ValueError('Independent audit contains duplicate samples')
        current={};layouts={};values={};samples={}
        for identity,record in zip(identities,records):
            for flag in ('valid','discovered'):
                if flag in record and not isinstance(record[flag],(bool,np.bool_)):raise ValueError('Audit flags must be boolean')
            discovered=record.get('discovered',False)
            if discovered and not self.allow_discovered:raise ValueError('Enable allow_discovered for a declared worst-point oracle')
            gates=record.get('gates')
            if not isinstance(gates,dict)or not gates:raise ValueError('Each audit sample requires nonempty signed gates')
            if any(not isinstance(key,str)or not key for key in gates):raise ValueError('Audit gate names must be nonempty strings')
            shape={key:np.asarray(value).shape for key,value in gates.items()}
            if discovered:
                if identity in self.discovered and self.discovered[identity]!=shape:raise ValueError('A discovered sample changed its gate layout')
                self.discovered[identity]=shape
            else:current[identity]=record;layouts[identity]=shape
            measured={}
            for key,value in gates.items():
                array=np.asarray(value,dtype=float)
                if not array.size:raise ValueError('Audit gates cannot be empty')
                number=float(np.min(array));measured[key]=number if math.isfinite(number)else-math.inf
            if not record.get('valid',True):measured['valid_ray_trace']=-math.inf
            values[identity]=measured;samples[identity]=record['sample']
        if not current:raise ValueError('The immutable audit core cannot be empty')
        if self.domain is None:self.domain=set(current);self.layout=layouts
        elif set(current)!=self.domain:raise ValueError('Independent audit domain changed between rounds')
        elif layouts!=self.layout:raise ValueError('Independent audit gate layout changed between rounds')
        if any(key in self.discovered and key not in values for key in active):
            raise ValueError('Independent oracle omitted an active discovered sample')
        minima={};worst={};failed=[];safe={}
        for identity,measured in values.items():
            for key,value in measured.items():
                minima[key]=min(minima.get(key,math.inf),value)
                if key not in worst or value<worst[key][0]:worst[key]=(value,identity,samples[identity])
            slack=min(measured.values());safe[identity]=slack
            if slack<0.:failed.append((slack,identity,samples[identity]))
        return dict(passed=not failed,minima=minima,worst=worst,failed=failed,safe=safe,
                    samples=len(records),fixed=len(current),records=records)


def run_constraint_exchange(solve_round,audit,active_samples,initial_q,*,max_rounds=4,add_per_round=8,
                            callback=None,protected_samples=(),drop_after=0,drop_slack=None,
                            max_active=None,promote_worst=False,allow_discovered=False,audit_merit=None):
    """Solve, audit the full fixed domain, exchange samples, rebuild local model.

    Optional audit_merit(records), lower is better, audits the initial point and
    rejects proposals worsening that COMPLETE-domain score. Rejected proposals
    still contribute their worst constraints; returned coordinates remain the
    accepted point. Callback candidate_accepted permits compensator rollback.
    A discovered oracle may move, while the immutable core and every active
    discovered point remain audited. WFE samples are complete quadratures.
    """
    if type(max_rounds)is not int or max_rounds<1 or type(add_per_round)is not int or add_per_round<1:raise ValueError('Round/add budgets must be positive integers')
    if type(drop_after)is not int or drop_after<0:raise ValueError('drop_after must be a nonnegative integer')
    if drop_after and(drop_slack is None or not math.isfinite(drop_slack)or drop_slack<=0.):raise ValueError('Dropping requires a positive finite normalized drop_slack')
    if max_active is not None and(type(max_active)is not int or max_active<1):raise ValueError('max_active must be a positive integer')
    q=np.asarray(initial_q,dtype=float).copy()
    if q.ndim!=1 or not len(q)or not np.isfinite(q).all():raise ValueError('initial_q must be a finite nonempty vector')
    active={_identity(s):deepcopy(s)for s in active_samples};protected={_identity(s):deepcopy(s)for s in protected_samples};active.update(protected)
    if not active:raise ValueError('The initial working set cannot be empty')
    if max_active is not None and len(active)>max_active:raise ValueError('Initial/protected working set exceeds its explicit budget')
    domain=_AuditDomain(allow_discovered);history=[];wide_counts={};start=perf_counter();passed=False;reason='round_limit';worst_set_complete=False
    accepted_snapshot=None;best_score=None
    def merit(records):
        score=float(audit_merit(deepcopy(records)))
        if not math.isfinite(score):raise ValueError('Full-domain audit merit must be finite')
        return score
    if audit_merit is not None:
        accepted_snapshot=domain.inspect(list(audit(q.copy())),active);best_score=merit(accepted_snapshot['records']);passed=accepted_snapshot['passed']
    for iteration in range(max_rounds):
        trial,solver=solve_round(q.copy(),deepcopy(list(active.values())),iteration);trial=np.asarray(trial,dtype=float)
        if trial.shape!=q.shape or not np.isfinite(trial).all():raise ValueError('The local solver returned an invalid parameter vector')
        observed=domain.inspect(list(audit(trial.copy())),active)
        score=None if audit_merit is None else merit(observed['records'])
        accept=audit_merit is None or score<=best_score+1e-12
        if accept:q=trial.copy();accepted_snapshot=observed;best_score=score
        passed=accepted_snapshot['passed']
        selected_worst=observed['worst']if promote_worst else{k:v for k,v in observed['worst'].items()if v[0]<0.}
        worst_ids={entry[1]for entry in selected_worst.values()}
        for identity in active:
            wide=accepted_snapshot['safe'].get(identity,-math.inf)>(drop_slack if drop_after else math.inf)
            wide_counts[identity]=wide_counts.get(identity,0)+1 if wide else 0
        dropped=[]
        if drop_after and not passed:
            eligible=[key for key in active if key not in protected and key not in worst_ids and wide_counts.get(key,0)>=drop_after]
            for identity in sorted(eligible,key=lambda key:(-accepted_snapshot['safe'][key],key)):
                if len(active)<=1:break
                dropped.append(active.pop(identity));wide_counts.pop(identity,None)
        candidates=sorted(selected_worst.values())+sorted(observed['failed']);added=[];blocked=[];seen=set()
        for _,identity,sample in candidates:
            if identity in active or identity in seen:continue
            seen.add(identity)
            if max_active is not None and len(active)>=max_active:blocked.append(deepcopy(sample));continue
            active[identity]=deepcopy(sample);added.append(deepcopy(sample))
            if len(added)==add_per_round:break
        worst_set_complete=not added and not blocked
        minima=accepted_snapshot['minima']
        item=dict(round=iteration,working_samples=len(active)-len(added)+len(dropped),audited_samples=observed['samples'],
            fixed_audit_samples=observed['fixed'],discovered_audit_samples=observed['samples']-observed['fixed'],
            failed_samples=len(observed['failed']),gate_minima={k:v if math.isfinite(v)else None for k,v in minima.items()},
            added_samples=added,dropped_samples=dropped,budget_blocked_samples=blocked,next_working_samples=len(active),
            independent_audit_passed=passed,trial_independent_audit_passed=observed['passed'],candidate_accepted=accept,
            accepted_audit_merit=best_score,trial_audit_merit=score,solver=solver,seconds=perf_counter()-start)
        history.append(item)
        if callback is not None:callback(deepcopy(item))
        if passed and(not promote_worst or worst_set_complete):reason='independent_audit_passed';break
        if passed and blocked:reason='feasible_working_budget_exhausted';break
        if not added:
            reason='working_budget_exhausted'if blocked else'full_audit_filter_stalled'if not accept else'active_set_stalled';break
    return dict(q=q.tolist(),passed=passed,termination=reason,history=history,active_samples=list(active.values()),
        seconds=perf_counter()-start,protected_samples=list(protected.values()),worst_set_complete=worst_set_complete,
        accepted_audit_merit=best_score,full_domain_filter_enabled=audit_merit is not None,continuous_domain_certified=False)
