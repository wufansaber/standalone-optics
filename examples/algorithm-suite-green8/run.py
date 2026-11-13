"""Run the six algorithm upgrades on the actual frozen green8 14-piece case."""
import argparse,json,os,sys,time
from dataclasses import replace
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from problem import BASE,SPEC,CAPS,Design,GreenProblem,WorkingProblem,apply_scenario,scenario_set,component,configured,dense_first_order
from audit import full_audit,wave_audit,focus_audit
from scripts.ai_zemax.standalone_constraint_exchange import run_constraint_exchange
from scripts.ai_zemax.standalone_reduced import equality_nullspace,AffineReducedProblem
from scripts.ai_zemax.standalone_span import SpanEpigraphProblem
from scripts.ai_zemax.standalone_scenarios import ScenarioProblem
from scripts.ai_zemax.standalone_robust_exchange import run_robust_exchange
from scripts.ai_zemax.standalone_branch_budget import run_budgeted_branches

def save(path,value):Path(path).write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
def summary(audit):
    result={k:v for k,v in audit.items()if k!='per_wave'}
    if 'focus'in result:result['focus']={k:v for k,v in result['focus'].items()if k not in ('axial_rows','field_rows')}
    return result
INITIAL=[dict(kind='wfe',field_mm=h,pupil_rule='gq4x12')for h in (0.,.625,1.25)]


def derivative_check(folder,pool):
    result=[];rng=np.random.default_rng(174)
    for scope in ('tube','all'):
        design=Design(scope=scope);p=GreenProblem([0.,.625,1.25],design=design,executor=pool);q=np.zeros(design.n)
        base=p.batch([q])[0];start=time.perf_counter();provided=p.jacobian(q,base,tuple(base['gates']));elapsed=time.perf_counter()-start
        raw=p.evaluate(q,True);a=raw['gate_J']['physical'][base['gates']['physical']<.001]
        a=np.r_[a,raw['gate_J']['first_order']];norm=np.linalg.norm(a,axis=1);a=a[norm>1e-10]/norm[norm>1e-10,None]
        z,reduction=equality_nullspace(a);checks=[]
        for _ in range(3):
            direction=z@rng.normal(size=z.shape[1]);direction/=np.linalg.norm(direction);errors=[]
            analytic=provided['slacks']@direction
            for step in (1e-4,3e-5,1e-5):
                plus,minus=p.batch([q+step*direction,q-step*direction])
                if plus is None or minus is None:continue
                flatten=lambda r:np.concatenate([np.ravel(r['gates'][k])for k in base['gates']])
                finite=(flatten(plus)-flatten(minus))/(2*step)
                errors.append(dict(step=step,relative_L2=float(np.linalg.norm(finite-analytic)/max(1.,np.linalg.norm(analytic))),maximum_absolute=float(max(abs(finite-analytic)))))
            if not errors or min(v['relative_L2']for v in errors)>2e-4:raise ValueError('Actual-case analytic directional derivative failed')
            checks.append(errors)
        result.append(dict(scope=scope,parameters=design.n,seconds=elapsed,reduction=reduction,
                           directions=checks,counters=p.counts,passed=True))
    save(folder/'derivatives.json',result);return result


def audit_records(model,checked,pool,extra=()):
    if 'per_wave'not in checked:
        # Never hide a failed ray by dropping the affected domain.
        checked=full_audit(model,executor=pool)
        if 'per_wave'not in checked:raise ValueError('Complete independent optical audit could not be evaluated')
    by={r['wavelength_um']:r for r in checked['per_wave']};waves=sorted(CAPS);records=[];known=set()
    def wfe_record(h,rule,measurements,discovered=False):
        actual=[measurements[w]for w in waves];reference=measurements[.536]['chief_y_mm']
        gates=dict(wfe=np.array([1-r['GQ12x48'if rule=='gq12x48'else'RECT16']/CAPS[w]for w,r in zip(waves,actual)]),
            angle=np.array([1-r['angle_deg']/.07 for r in actual]),distortion=np.array([1-abs(r['distortion_percent'])/.6 for r in actual]),
            lateral_color=np.array([1-abs(measurements[w]['chief_y_mm']-reference)/.0018974173564139107 for w in SPEC['hard_green_lateral_color']['reference_measured_waves_um']]),
            physical=min(r['physical_slack_mm']for r in measurements.values())/.1,
            first_order=checked['minimum_first_order_slack'])
        sample=dict(kind='wfe',field_mm=float(h),pupil_rule=rule);key=json.dumps(sample,sort_keys=True)
        if key not in known:records.append(dict(sample=sample,gates=gates,discovered=discovered));known.add(key)
    grid=[r['field_mm']for r in by[waves[0]]['rows']]
    for i,h in enumerate(grid):
        measurements={w:by[w]['rows'][i]for w in by}
        for rule in ('gq12x48','rect16'):wfe_record(h,rule,measurements)
    local_fields={round(r['field_mm'],7)for wave in checked['per_wave']if wave['wavelength_um']in CAPS for r in wave['local_extrema']}
    local_fields.update(float(s['field_mm'])for s in extra if s.get('kind')=='wfe'and s.get('pupil_rule')in ('gq12x48','rect16')and s['field_mm']not in grid)
    for h in sorted(local_fields-set(grid)):
        tasks=[(model,w,[h],False)for w in by];values=list(pool.map(wave_audit,tasks))if pool else[wave_audit(t)for t in tasks]
        measurements={r['wavelength_um']:r['rows'][0]for r in values}
        for rule in ('gq12x48','rect16'):wfe_record(h,rule,measurements,True)
    focus=checked['focus'];axial=focus['axial_rows'];f=np.array([r['full']for r in axial])
    for i,r in enumerate(axial):
        span=max(abs(f[i]-f));records.append(dict(sample=dict(kind='lca',wavelength_nm=int(round(r['wavelength_um']*1000)),pupil_rule='gq20x64'),gates={'lca':1-span/.0007}))
    for r in focus['field_rows']:
        records.append(dict(sample=dict(kind='fc',field_mm=r['field_mm'],pupil_rule='gq20x64'),gates={'field_curvature':1-abs(r['relative_object_mm'])/.0007}))
    fc_fields={round(r['field_mm'],7)for r in focus['local_extrema']}
    fc_fields.update(s['field_mm']for s in extra if s.get('kind')=='fc')
    reference=focus['field_rows'][0]['mean']
    for h in sorted(fc_fields-set(grid)):
        value=focus_audit((model,.595,[h]))[0]['mean']-reference
        records.append(dict(sample=dict(kind='fc',field_mm=float(h),pupil_rule='gq20x64'),gates={'field_curvature':1-abs(value)/.0007},discovered=True))
    return records


def strategy_run(mode,initial,folder,pool,*,rounds=3,iters=5,margin_iters=3,design=None,initial_samples=None):
    design=design or Design();initial_samples=INITIAL if initial_samples is None else initial_samples
    state={'samples':initial_samples,'audits':[],'reports':[],'best':None};start=time.perf_counter()
    def solve(q,samples,index):
        save(folder/f'working-samples-{index:02d}.json',samples)
        state['samples']=samples;p=WorkingProblem(samples,design=design,executor=pool,include_span=mode!='lifted-reduced',wfe_guard=.98,constraint_buffer=.001)
        initial_q=q;provider=None if mode=='finite-difference'else p.jacobian;problem=p
        expand=lambda x:np.array(x)
        reduction=None
        if mode=='lifted-reduced':
            lift=SpanEpigraphProblem(p,.0007*.999,auxiliary_bounds=[[-.02,.02],[-.02,.02]],scale=.001,jacobian_provider=p.jacobian)
            anchor=lift.initial(q);base=lift.batch([anchor])[0];jac=lift.jacobian(anchor,base,tuple(lift.slacks(base)))['slacks']
            blocks=lift.slacks(base);offset=0;critical=[]
            for name,value in blocks.items():
                size=np.asarray(value).size;part=jac[offset:offset+size];offset+=size
                if name=='first_order'and index==0:critical.extend(part)
                if name=='physical':critical.extend(part[np.ravel(value)<.001])
            a=np.array(critical);norm=np.linalg.norm(a,axis=1);a=a[norm>1e-10]/norm[norm>1e-10,None];z,reduction=equality_nullspace(a)
            reduced=AffineReducedProblem(lift,anchor,a,a@anchor,reduced_bounds=[[-.5,.5]]*z.shape[1],jacobian_provider=lift.jacobian)
            problem=reduced;provider=reduced.jacobian;initial_q=np.zeros(z.shape[1]);expand=lambda x:reduced.expand(x)[:design.n]
        hard=['geometry','physical']+(['_parameter_bounds']if mode=='lifted-reduced'else[])
        result=problem.solve(folder/f'round-{index:02d}',iters=iters,start_trust=.05,variant=.2,damping=1e-6,
            phase2_iterations=margin_iters,margin_keys=['wfe'],margin_scales={'wfe':1.},hard_gate_keys=hard,
            initial_q=initial_q,jacobian_provider=provider,subproblem_backend='clarabel',difference_step=1e-4,
            norm_rank_tolerance=0. if mode=='finite-difference'else 1e-10,verbose=False)
        physical=expand(result['q']);state['reports'].append(dict(solver=result,reduction=reduction))
        return physical,result
    def audit(q):
        checked=full_audit(design.model(q),executor=pool);state['audits'].append(summary(checked))
        if checked['passed']and(state['best']is None or checked['green8_WFE_norm']<state['best']['audit']['green8_WFE_norm']):state['best']=dict(q=q.tolist(),audit=summary(checked))
        save(folder/f'audit-{len(state["audits"])-1:02d}.json',checked)
        return audit_records(design.model(q),checked,pool,state['samples'])
    folder.mkdir(parents=True,exist_ok=False)
    result=run_constraint_exchange(solve,audit,initial_samples,initial,max_rounds=rounds,add_per_round=4,
        protected_samples=[INITIAL[0]],promote_worst=True,drop_after=2,drop_slack=.5,max_active=20,allow_discovered=True)
    final=full_audit(design.model(result['q']),executor=pool);state['final']=summary(final)
    result.update(mode=mode,seconds=time.perf_counter()-start,full_final_acceptance=summary(final),best_complete_feasible=state['best'],
        strategy_reports=state['reports'],physical_source_modified=False,source_scope_sha256=SPEC['scope_sha256'])
    save(folder/'FINAL_AUDIT.json',final);save(folder/'RESULT.json',result);save(folder/'CANDIDATE_MODEL.json',design.model(result['q']))
    return result


def make_robust_problem(selected,design,pool):
    samples=[dict(kind='wfe',field_mm=h,pupil_rule=rule)for h in (0.,.625,1.25)for rule in ('gq4x12','rect16')]
    adapters={key:WorkingProblem(samples,design=design,executor=pool,scenario=payload,wfe_guard=.98)for key,payload in selected.items()}
    dense_cache={}
    def key_of(payload):return next(k for k,v in selected.items()if v==payload)
    def dense(q,payload,derivatives,fields):
        key=(np.asarray(q).tobytes(),json.dumps(payload,sort_keys=True),derivatives,fields)
        if key not in dense_cache:
            parameters=design.parameters
            if payload['kind']=='radius':
                parameters=tuple(replace(p,scale=p.scale/(1+payload['relative_R_error']))
                    if p.surface_index==payload['surface']-1 and p.attribute=='curvature_mm_inv'else p for p in parameters)
            dense_cache[key]=dense_first_order(apply_scenario(design.model(q),payload),parameters,
                executor=pool,derivatives=derivatives,fields=fields)
        if len(dense_cache)>100:
            saved=dense_cache[key];dense_cache.clear();dense_cache[key]=saved
        return dense_cache[key]
    def evaluate(q,payload,controls):
        p=adapters[key_of(payload)];row=p.batch([q])[0]
        if row is None:return None
        gates=p.slacks(row);nominal=payload['kind']=='nominal';optical=[k for k in gates if k not in ('geometry','physical','wfe','magnification','chief_angle','distortion','lateral_color')]
        other=np.concatenate([np.ravel(gates[k])for k in optical]);dense_values=dense(q,payload,False,65)[0]if nominal else np.ones(2860)
        mask=row['wave_vector']if nominal else np.zeros_like(row['wave_vector'])
        gates=dict(gates,scenario_first_order=dense(q,payload,False,17)[0],nominal_other=np.r_[other,dense_values]if nominal else np.ones(len(other)+len(dense_values)),
                   nominal_wfe=.98-np.linalg.norm(mask,axis=-1))
        return dict(gates=gates,vectors={'wave_vector':row['wave_vector'],'nominal_wave_vector':mask})
    def derivative(q,payload,controls,base):
        p=adapters[key_of(payload)];row=p.batch([q])[0];keys=tuple(p.slacks(row));supplied=p.jacobian(q,row,keys);blocks={};start=0
        for k,v in p.slacks(row).items():
            size=np.asarray(v).size;blocks[k]=supplied['slacks'][start:start+size].reshape(np.asarray(v).shape+(len(q),));start+=size
        optical=[k for k in keys if k not in ('geometry','physical','wfe','magnification','chief_angle','distortion','lateral_color')];j=np.concatenate([blocks[k].reshape(-1,len(q))for k in optical])
        nominal=payload['kind']=='nominal';dense_j=dense(q,payload,True,65)[1]if nominal else np.zeros((2860,len(q)))
        blocks['scenario_first_order']=dense(q,payload,True,17)[1]
        blocks['nominal_other']=np.r_[j,dense_j]if nominal else np.zeros((len(j)+len(dense_j),len(q)))
        blocks['nominal_wfe']=blocks['wfe']if nominal else np.zeros_like(blocks['wfe'])
        vectors=dict(supplied['compressed_vectors']);vectors['nominal_wave_vector']=vectors['wave_vector'].copy()if nominal else np.zeros_like(vectors['wave_vector'])
        return dict(gates=blocks,vectors={},compressed_vectors=vectors)
    return ScenarioProblem(design.bounds,selected,evaluate,design.model,local_jacobian=derivative,
                           local_norm_specs={'wfe':dict(vector='wave_vector',gain=1.,offset=.98),
                                             'nominal_wfe':dict(vector='nominal_wave_vector',gain=1.,offset=.98)})


def robust_run(folder,pool,*,iters=8,margin_iters=5,rounds=4,initial_design=None):
    design=Design();scenarios=scenario_set();start=time.perf_counter();latest={}
    def factory(selected):return make_robust_problem(selected,design,pool)
    def audit(q,key,payload,controls):
        model=apply_scenario(design.model(q),payload)
        # Exact same dense domain for each scenario; final nominal separately
        # includes formal focus and local extrema. No scenario can be filtered.
        checked=full_audit(model,executor=pool,fields=17,local_search=False,include_focus=False,perturbed=payload['kind']!='nominal')
        latest[key]=summary(checked)
        if 'green8_WFE_norm'not in checked:return{'gates':{'wfe':-1.,'first_order':-1.,'physical':-1.},'valid':False}
        return{'gates':{'wfe':1-checked['green8_WFE_norm'],'first_order':checked['minimum_normalized_first_order_slack'],
                        'physical':1. if checked['physical_all15_passed']and checked['fullbody']['passed']else-1.},'valid':True}
    folder.mkdir(parents=True,exist_ok=False)
    baseline={key:summary(full_audit(apply_scenario(BASE,payload),executor=pool,fields=17,local_search=False,
                    include_focus=False,perturbed=payload['kind']!='nominal'))for key,payload in scenarios.items()}
    save(folder/'BASELINE_SCENARIOS.json',baseline)
    initial=np.zeros(design.n)if initial_design is None else np.asarray(initial_design,dtype=float)
    initial_audit=full_audit(design.model(initial),executor=pool);save(folder/'INITIAL_FULL_AUDIT.json',initial_audit)
    if not initial_audit['passed']:raise ValueError('Robust continuation must start from a fully accepted nominal design')
    initial_scenarios={key:summary(full_audit(apply_scenario(design.model(initial),payload),executor=pool,fields=17,local_search=False,
                    include_focus=False,perturbed=payload['kind']!='nominal'))for key,payload in scenarios.items()}
    save(folder/'INITIAL_SCENARIOS.json',initial_scenarios)
    initial_active=['nominal','R_S21_-1','R_S21_+1','CT_T2_-1','CT_T2_+1','AIR_-1','AIR_+1','R_S24_+1']
    result=run_robust_exchange(scenarios,initial_active,initial,
        problem_factory=factory,audit_scenario=audit,output_directory=folder/'exchange',protected_scenarios=['nominal'],
        max_rounds=rounds,add_per_round=3,drop_after=2,drop_slack=.5,max_active=23,
        solve_options=dict(iters=iters,start_trust=.02,variant=.2,damping=1e-6,analytic=True,
            phase2_iterations=margin_iters,margin_keys=['wfe'],margin_scales={'wfe':1.},
            hard_gate_keys=['geometry','physical','nominal_other','nominal_wfe'],subproblem_backend='clarabel',
            norm_rank_tolerance=1e-10,verbose=False))
    model=design.model(result['q']);nominal=full_audit(model,executor=pool);save(folder/'NOMINAL_FULL_AUDIT.json',nominal)
    final={key:summary(full_audit(apply_scenario(model,payload),executor=pool,fields=17,local_search=False,
                    include_focus=False,perturbed=payload['kind']!='nominal'))for key,payload in scenarios.items()}
    result.update(nominal_full_acceptance=summary(nominal),scenario_before=baseline,scenario_initial=initial_scenarios,scenario_after=final,
        initial_WFE_failed=sum(r.get('green8_WFE_norm',float('inf'))>1. for r in initial_scenarios.values()),
        before_WFE_failed=sum(r.get('green8_WFE_norm',float('inf'))>1. for r in baseline.values()),
        after_WFE_failed=sum(r.get('green8_WFE_norm',float('inf'))>1. for r in final.values()),
        worst_WFE_initial=max(r.get('green8_WFE_norm',float('inf'))for r in initial_scenarios.values()),
        worst_WFE_before=max(r.get('green8_WFE_norm',float('inf'))for r in baseline.values()),
        worst_WFE_after=max(r.get('green8_WFE_norm',float('inf'))for r in final.values()),
        seconds=time.perf_counter()-start,physical_source_modified=False,
        scenario_scope='22 fixed-image single-error probes; GQ12x48+RECT16, 17 fields. Not simultaneous errors or yield.')
    save(folder/'RESULT.json',result);save(folder/'CANDIDATE_MODEL.json',model);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=['baseline','derivatives','benchmark','robust'],required=True)
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--cpu-count',type=int)
    ap.add_argument('--iters',type=int,default=5);ap.add_argument('--rounds',type=int,default=3);ap.add_argument('--margin-iters',type=int,default=3)
    ap.add_argument('--initial-q',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.cpu_count:
        import psutil;psutil.Process().cpu_affinity(list(range(args.cpu_count)))
    with ProcessPoolExecutor(max_workers=args.workers)as pool:
        if args.stage=='baseline':
            started=time.perf_counter();r=full_audit(BASE,executor=pool);r['seconds']=time.perf_counter()-started;save(args.out/'RESULT.json',r)
            print(json.dumps(summary(r)),flush=True)
            if not r['passed']:raise SystemExit('Frozen source no longer passes the unchanged complete acceptance')
        elif args.stage=='derivatives':print(json.dumps(derivative_check(args.out,pool)),flush=True)
        elif args.stage=='benchmark':
            import platform,scipy,clarabel,psutil,hashlib
            warm_start=time.perf_counter();warm=full_audit(BASE,executor=pool)
            if not warm['passed']:raise ValueError('Benchmark warm-up baseline no longer passes')
            save(args.out/'BENCHMARK_ENVIRONMENT.json',dict(python=platform.python_version(),numpy=np.__version__,
                scipy=scipy.__version__,clarabel=clarabel.__version__,workers=args.workers,
                cpu_affinity=psutil.Process().cpu_affinity(),logical_cpus=psutil.cpu_count(),
                blas_threads=1,warmup_seconds=time.perf_counter()-warm_start,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                iterations_per_round=args.iters,margin_iterations_per_round=args.margin_iters,
                max_rounds=args.rounds,scope_sha256=SPEC['scope_sha256'],warmup_in_run_times=False))
            design=Design();starts=[np.zeros(design.n)]
            error=np.zeros(design.n);i=design.curved.index(20);error[i]=(design.c0[20]/(1-.0001)-design.c0[20])/design.cs[i];starts.append(error)
            records=[]
            for si,start in enumerate(starts):
                for mode in (('finite-difference','analytic','lifted-reduced')if si%2==0 else('lifted-reduced','analytic','finite-difference')):
                    r=strategy_run(mode,start,args.out/f'{mode}-{si}',pool,rounds=args.rounds,iters=args.iters,margin_iters=args.margin_iters)
                    item=dict(mode=mode,start=si,seconds=r['seconds'],passed=r['full_final_acceptance']['passed'],
                              WFE_norm=r['full_final_acceptance'].get('green8_WFE_norm'),rounds=len(r['history']),termination=r['termination'])
                    records.append(item);save(args.out/'BENCHMARK.json',records);print(json.dumps(item),flush=True)
        else:
            initial=None if args.initial_q is None else json.loads(args.initial_q.read_text())['q']
            r=robust_run(args.out/'robust',pool,iters=args.iters,margin_iters=args.margin_iters,rounds=args.rounds,initial_design=initial)
            print(json.dumps({k:r[k]for k in ['passed','before_WFE_failed','after_WFE_failed','worst_WFE_before','worst_WFE_after','seconds']}),flush=True)

if __name__=='__main__':main()
