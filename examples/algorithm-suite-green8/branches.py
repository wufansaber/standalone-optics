"""Real-catalog branches, thick-system restoration and budgeted continuation."""
import argparse,os,sys,json,gzip,hashlib,time
from pathlib import Path
from copy import deepcopy
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
from problem import BASE,SPEC,PINS,Design,GreenProblem,WorkingProblem,component
from audit import full_audit
from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin
from scripts.ai_zemax.standalone_branch_budget import run_budgeted_branches

RAW=gzip.decompress((HERE/'GLASS_BRANCHES.json.gz').read_bytes())
if hashlib.sha256(RAW).hexdigest()!=PINS['glass_branches_serialization_sha256']:raise ValueError('Frozen real-glass branch data changed')
DATA=json.loads(RAW)
save=lambda p,r:Path(p).write_bytes((json.dumps(r,indent=2,allow_nan=False)+'\n').encode())


def features(design,q):
    model=design.model(q);wi=model['wavelengths_um'].index(.595);ports=SPEC['ports_mm'];values=[];derivatives=[]
    for a,b,lo,hi,scale in [(2,19,ports['objective_entry'],ports['objective_exit'],.05),(20,25,ports['tube_entry'],ports['tube_exit'],1/178.065)]:
        matrix,j=component(model,wi,a,b,lo,hi,design.cjac,design.zjac)
        for i,k,s in [(0,0,1.),(1,0,scale),(1,1,1.)]:values.append(matrix[i,k]/s);derivatives.append(j[i,k]/s)
    return np.array(values),np.array(derivatives)


class PhysicalRestorer(EfficientSolveMixin):
    def __init__(self,design):
        self.design=design;self.bounds=design.bounds
        self.p=GreenProblem([0.,.625,1.25],design=design,strict_physical=False)
        self.geometry_mask=np.linalg.norm(design.geometry(design.base)[1],axis=1)>1e-14
    def batch(self,qs):return self.p.batch(qs)
    def slacks(self,row):
        result={}
        for name in ('geometry','physical','first_order'):
            value=np.array(row['gates'][name],copy=True)
            if name=='physical':value-=1e-5
            if name=='geometry':
                # The fixed window's exact zero is immutable; no invented
                # thickness is demanded there. Variable rows get a tiny guard.
                value[self.geometry_mask]-=1e-7
            result[name]=value
        return result
    def model(self,q):return self.design.model(q)
    def jacobian(self,q,base,keys):
        row=self.p.evaluate(q,True)
        if row is None:raise ValueError('Restoration derivative is undefined')
        return np.concatenate([row['gate_J'][key].reshape(-1,len(q))for key in keys])


def realize(branch,folder):
    model=deepcopy(BASE)
    for change in branch['changes']:
        medium=change['medium'];material=DATA['materials'][change['name']]
        if material['snapshot_sha256']!=DATA['catalog_sha256']:raise ValueError('Glass snapshot changed')
        for row,n in zip(model['indices_by_wave'],material['indices_by_wave']):row[medium]=n
        model['material_names'][medium]=material['name']
        for piece in model['manufacturing_policy']['piece_floors']:
            if piece['front_surface']==medium:piece['material']=material['name']
    if branch['changes']:
        design=Design(base=model);target,_=features(Design(),np.zeros(46));nc=len(design.curved)
        def expand(c):return np.r_[c,np.zeros(design.n-nc)]
        answer=least_squares(lambda c:features(design,expand(c))[0]-target,np.zeros(nc),
            jac=lambda c:features(design,expand(c))[1][:,:nc],bounds=(-10*np.ones(nc),10*np.ones(nc)),
            max_nfev=40,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        if max(abs(features(design,expand(answer.x))[0]-target))>1e-6:raise ValueError('Actual thick-system first-order restoration did not converge')
        model=design.model(expand(answer.x))
        restored=PhysicalRestorer(Design(base=model));raw=restored.batch([np.zeros(restored.design.n)])[0]
        if raw is None:raise ValueError('Restoration surrogate has an invalid ray branch')
        # Remove immutable geometric zeros from the numerical guard by deriving
        # their exact Jacobian, without changing any original physical floor.
        analytic=restored.p.evaluate(np.zeros(restored.design.n),True)
        if analytic is None:raise ValueError('First-order seed has singular restoration derivatives')
        solve=restored.solve(folder/'physical-restoration',iters=12,start_trust=.1,variant=.2,damping=1e-6,
             jacobian_provider=restored.jacobian,subproblem_backend='clarabel',norm_rank_tolerance=1e-10,verbose=False)
        model=restored.model(np.array(solve['q']))
        strict=GreenProblem([0.,.625,1.25],design=Design(base=model))
        if strict.batch([np.zeros(strict.design.n)])[0]is None:raise ValueError('Uncut restoration did not produce a physically valid seed')
    return model


def advance(branch,state,budget,target):
    folder=Path(os.environ['OPTICS_BRANCH_RUN_ROOT'])/branch['id'];folder.mkdir(parents=True,exist_ok=True)
    if state is None:model=realize(branch,folder);q=np.zeros(46);spent=0
    else:model=state['base_model'];q=np.array(state['q']);spent=state['allocated_iterations']
    design=Design(base=model)
    if state is None:
        i=design.curved.index(20)
        if branch['seed']=='power':q[i]=.002;q[design.curved.index(23)]=-.002
        elif branch['seed']=='spacing':q[len(design.curved)+design.positions.index(23)]=.01
    samples=[dict(kind='wfe',field_mm=h,pupil_rule=rule)for h in (0.,.625,1.25)for rule in ('gq4x12','rect16')]
    p=WorkingProblem(samples,design=design,wfe_guard=.98);base=p.batch([q])[0]
    if base is None:return dict(valid=False,error='No physical seed after thick-system restoration')
    feasible=all(np.min(v)>=0. for v in p.slacks(base).values())
    result=p.solve(folder/f'budget-{spent+budget:03d}',iters=0 if feasible else budget,start_trust=.03,variant=.2,damping=1e-6,
        phase2_iterations=budget if feasible else 0,margin_keys=['wfe'],margin_scales={'wfe':1.},
        hard_gate_keys=['geometry','physical'],initial_q=q,jacobian_provider=p.jacobian,subproblem_backend='clarabel',norm_rank_tolerance=1e-10,verbose=False)
    q=np.array(result['q']);row=p.batch([q])[0]
    if row is None:return dict(valid=False,error='Continuation ended at an invalid physical candidate')
    gates=p.slacks(row);deficit=max(max(0.,-float(np.min(v)))for v in gates.values());norm=float(np.max(np.linalg.norm(row['wave_vector'],axis=-1)))
    score=100.*deficit+norm
    return dict(valid=True,score=score,state=dict(base_model=model,q=q.tolist(),allocated_iterations=spent+budget),
        solver_counts=result['counters'],solver_seconds=result['timings']['seconds'],worst_working_WFE_norm=norm,
        any_working_violation=deficit,materials_frozen=True)


def make_branches():
    variants=[dict(id='current',family='current',changes=[])]
    variants.extend(dict(id=r['id'],family=r['family'],changes=[dict(medium=r['medium'],name=r['name'])])for r in DATA['branches'])
    first=[r for r in DATA['branches']if r['family']=='O1'][0];last=[r for r in DATA['branches']if r['family']=='T4'][0]
    variants.append(dict(id='paired-O1-T4',family='two-material',changes=[dict(medium=r['medium'],name=r['name'])for r in (first,last)]))
    return[dict(v,id=v['id']+'-'+seed,seed=seed)for v in variants for seed in ('base','power','spacing')]


def validate_branch(branch,state,folder,pool):
    design=Design(base=state['base_model']);model=design.model(state['q']);checked=full_audit(model,executor=pool)
    polishing=None
    if not checked['passed']:
        from run import strategy_run,audit_records,INITIAL
        records=audit_records(model,checked,pool);worst={}
        for record in records:
            for name,value in record['gates'].items():
                number=float(np.min(value))
                if name not in worst or number<worst[name][0]:worst[name]=(number,record['sample'])
        seeds=[*INITIAL,*[entry[1]for entry in worst.values()]]
        polishing=strategy_run('analytic',np.array(state['q']),folder/('polish-'+branch['id']),pool,
            rounds=3,iters=6,margin_iters=1,design=design,initial_samples=seeds)
        state=dict(state,q=polishing['q']);model=design.model(state['q']);checked=full_audit(model,executor=pool)
    save(folder/(branch['id']+'-FULL_AUDIT.json'),checked)
    if checked['passed']:save(folder/(branch['id']+'-ACCEPTED_MODEL.json'),model)
    return dict(passed=checked['passed'],green8_WFE_norm=checked.get('green8_WFE_norm'),
        physical_all15_passed=checked.get('physical_all15_passed'),scope_sha256=SPEC['scope_sha256'],manufacturing_release=False,
        validated_state=state,polishing_seconds=polishing['seconds']if polishing else 0.)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--cpu-count',type=int)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);os.environ['OPTICS_BRANCH_RUN_ROOT']=str((args.out/'solves').resolve())
    if args.cpu_count:
        import psutil;psutil.Process().cpu_affinity(list(range(args.cpu_count)))
    with ProcessPoolExecutor(max_workers=args.workers)as audit_pool:
        def validate(branch,state,target):return validate_branch(branch,state,args.out,audit_pool)
        result=run_budgeted_branches(make_branches(),target={'scope_sha256':SPEC['scope_sha256'],'WFE_caps':SPEC['WFE_targets_waves']},
            rung_budgets=[1,2,4],advance=advance,validate=validate,output_directory=args.out/'schedule',
            workers=args.workers,executor='process',explore_slots=2,reduction=2,seed=174)
    result['catalog_scan']=dict(materials=DATA['scan_calculable'],wavelengths=len(DATA['full_wavelengths']),sites=4,
        shortlist_material_branches=12,paired_substitutions=1,seed_variants=3,continued_branches=len(make_branches()))
    save(args.out/'RESULT.json',result)
    print(json.dumps({k:result[k]for k in ('any_passed','allocated_work','equal_full_budget_work','seconds','catalog_scan')}),flush=True)

if __name__=='__main__':main()
