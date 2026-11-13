"""Actual-case native adaptive MF optimization and O(N) sampled span audit."""
import os,sys,json,time,math,hashlib,shutil,traceback
from pathlib import Path
from copy import deepcopy
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip
CODE=Path(__file__).resolve().parents[2];CASE=Path(__file__).resolve().parent
sys.path[:0]=[str(CODE),str(CASE)]
D=None;INPUTS=None;OUT=None;ARGS=None
import numpy as np,psutil
from scripts.ai_zemax.standalone_merit_graph import compact,MeritGraph,dependencies,CONTROL,MEASURE,write_native
from scripts.ai_zemax.standalone_merit_sphere import spherical_support
from scripts.ai_zemax.standalone_span import replace_pairwise_span
from scripts.ai_zemax.standalone_merit_exchange import run_merit_exchange
from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin
from contextlib import nullcontext
from problem import BASE,Design,GreenProblem,PINS
from audit import full_audit
from concurrent.futures import ProcessPoolExecutor
def load(path):
 path=Path(path)
 raw=path.read_bytes()if path.exists()else gzip.decompress(Path(str(path)+'.gz').read_bytes())
 return json.loads(raw)

save=lambda p,x:Path(p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cached(path):
 values=[float(line.split()[-1])for line in path.read_text(encoding='utf-16').splitlines()[1:]if line.strip()]
 a=np.array(values)
 if not np.isfinite(a).all():raise ValueError('Nonfinite native operands')
 return a


def main():
 global D,INPUTS,OUT,ARGS
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source-project',type=Path,required=True)
 parser.add_argument('--graphs',type=Path,required=True)
 parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--install',type=Path,required=True)
 parser.add_argument('--pythonnet-dir',type=Path)
 parser.add_argument('--workers',type=int,default=1)
 parser.add_argument('--cpu-count',type=int)
 ARGS=parser.parse_args();D=ARGS.source_project.resolve();INPUTS=ARGS.graphs.resolve();OUT=ARGS.out.resolve()
 if ARGS.pythonnet_dir:sys.path.insert(0,str(ARGS.pythonnet_dir.resolve()))
 if ARGS.cpu_count:psutil.Process().cpu_affinity(list(range(ARGS.cpu_count)))
 OUT.mkdir(exist_ok=False);project=OUT/'project';project.mkdir();shutil.copytree(D/'GLASSCAT',project/'GLASSCAT')
 sourcefile=D/'GREEN8_FULL_TARGETS_13CFG.ZMX';source_sha=sha(sourcefile)
 if source_sha!=PINS['native_ZMX_sha256']or sha(project/'GLASSCAT/OH9AD9D2.AGF')!=PINS['catalog_sha256']or sha(project/'GLASSCAT/FIXWIN.AGF')!=PINS['fixed_window_catalog_sha256']:raise ValueError('Frozen native prescription or catalogs differ')
 seed=project/'BASELINE.ZMX';shutil.copy2(sourcefile,seed)
 graph=load(INPUTS/'MF_FULL_GRAPH.json');geo=load(INPUTS/'MF_GEOMETRY_REFS.json');focus=load(INPUTS/'MF_FOCUS_REFS.json');wfe=load(INPUTS/'MF_WFE_REFS.json')
 span_graph,proof=replace_pairwise_span(graph,[r['object_full2D_shift_row']for r in focus['LCA_nodes']],
     [r['bound_rows']for r in focus['LCA_pair_bounds']],.0007,weight=graph[focus['LCA_pair_bounds'][0]['bound_rows'][0]-1]['weight'])
 removed=set(proof['deactivated_pair_bound_rows']);weighted={i for i,r in enumerate(graph,1)if r['weight']>0.}
 span_active=sorted((weighted-removed)|{proof['bound_row']})
 fullspan,spanreport=compact(span_graph,active_rows=span_active,audit_identity='original-green8-full-spec-independent-audit')
 write_native(project/'FULL_SPAN.MF',fullspan);save(OUT/'SPAN_PROOF.json',proof);save(OUT/'SPAN_REPORT.json',spanreport)
 support=spherical_support(graph,geo['geometry'],geo['domain'],BASE['explicit_nominal_solid_profiles']);protected=set(support['protected_rows'])
 for key in ('vertices','ports','cardinals','face_witnesses'):protected.update(i for r in geo[key]for i in r['bound_rows'])
 dref=load(INPUTS/'MF_OBJECTIVE_D_REFS.json')['value_row'];protected.update(i for i,r in enumerate(graph,1)if r['operand']in('OPGT','OPLT')and r['params'][0]==dref and r['weight']>0.);groups=[];initial=[];classes={};wfe_roots=set()
 def group(sample,roots,kind,selected=False):
  roots=[i for i in roots if i in weighted]
  if not roots:return
  groups.append(dict(sample=sample,rows=roots));classes.update({i:kind for i in roots})
  if selected:initial.append(sample)
 fields=[0.,.625,1.25]
 close=lambda h:any(abs(h-f)<1e-12 for f in fields)
 for r in wfe['conditions']:
  if r['role']=='green_hard_cap':
   refs=[r[k]['upper_bound_row']for k in ('RWCE','RWRE')];wfe_roots.update(refs)
   group(dict(family='wfe',field=r['field_mm'],wave=r['wavelength_um']),refs,'wfe',close(r['field_mm']))
 for key,kind in [('first_order','fo'),('lateral_color','lc')]:
  for r in geo[key]:group(dict(family=kind,field=r['field_y_mm'],wave=r['wavelength_um']),r['bound_rows'],kind,close(r['field_y_mm']))
 for r in focus['FC595_fields']:
  h=r['field_radius_mm'];group(dict(family='fc',field=h),r['bound_rows'],'fc',close(h)or abs(h-.8833608046477396)<1e-10)
 for r in focus['LCA_pair_bounds']:
  group(dict(family='lca',i=r['i_nm'],j=r['j_nm']),r['bound_rows'],'lca',(r['i_nm'],r['j_nm'])==(570,620))
 covered=protected|{i for g in groups for i in g['rows']}|set(support['removed_rows'])
 if weighted-covered:raise ValueError('Unclassified constraints in native adaptive source')
 work_graph=deepcopy(graph)
 for i in wfe_roots:work_graph[i-1]['target']*=.98
 dynamic=[False]*(len(graph)+1)
 for i,r in enumerate(graph,1):
  op=r['operand']
  if op=='CVVA':dynamic[i]=int(r['params'][0])in(21,24)
  elif op in {'TTHI','INDX','CONS'}|CONTROL:dynamic[i]=False
  elif op in MEASURE:dynamic[i]=True
  else:dynamic[i]=any(dynamic[j]for j in dependencies(r))
 app=None;started=time.perf_counter();result={'native_worker_sha256':sha(Path(__file__)),'source_ZMX_sha256':source_sha,'optimizer_started':True,'physical_source_modified':False,'scope_sha256':wfe['scope_sha256'],'native_instances':1}
 def update(**state):print(json.dumps(state),flush=True);save(OUT/'STATE.json',dict(state,seconds=time.perf_counter()-started))
 try:
  with nullcontext():
   from pythonnet import load as netload;netload('netfx');import clr
   install=ARGS.install.resolve();clr.AddReference(str(install/'ZOSAPI_NetHelper.dll'));import ZOSAPI_NetHelper
   if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(str(install)):raise RuntimeError('Native API initialization failed')
   clr.AddReference(str(install/'ZOSAPI_Interfaces.dll'));clr.AddReference(str(install/'ZOSAPI.dll'));import ZOSAPI as api
   before={p.pid for p in psutil.process_iter()};conn=api.ZOSAPI_Connection();conn.ConnectionTimeoutSeconds=45;app=conn.CreateNewApplication()
   if app is None or not app.IsValidLicenseForAPI:raise RuntimeError('Native API license')
   for process in psutil.process_iter(['name']):
    if process.pid not in before and 'opticstudio'in(process.info['name']or'').lower():
     if ARGS.cpu_count:process.cpu_affinity(list(range(ARGS.cpu_count)))
   system=app.PrimarySystem
   if not system.LoadFile(str(seed),False):raise RuntimeError('Load native copied source')
   if not system.IsProjectDirectory and not system.ConvertToProjectDirectory(str(project)):raise RuntimeError('Activate copied private project')
   system.UpdateFileLists()
   if system.MCE.NumberOfConfigurations!=13:raise ValueError('Native source configuration count changed')
   radii=[float(system.LDE.GetSurfaceAt(i).Radius)for i in (21,24)];curvatures=1/np.array(radii)
   guard_design=Design(scope='tube');guard=GreenProblem([0.,.3125,.625,.9375,1.25],design=guard_design)
   def independent_q(q):
    full=np.zeros(guard_design.n)
    for j,surface in enumerate((21,24)):
     k=guard_design.curved.index(surface-1);full[k]=curvatures[j]*1e-4*q[j]/guard_design.cs[k]
    return full
   def set_q(q):
    system.MCE.SetCurrentConfiguration(1)
    for i,c,x in zip((21,24),curvatures,q):system.LDE.GetSurfaceAt(i).Radius=float(1/(c*(1+x*1e-4)))
   def calculate(path,tag):
    system.MFE.LoadMeritFunction(str(path));value=float(system.MFE.CalculateMeritFunction())
    if not math.isfinite(value):raise ValueError('Nonfinite native merit')
    saved=OUT/(tag+'.MF');system.MFE.SaveMeritFunction(str(saved));return cached(saved),value
   original_path=D/'GREEN8_MANUFACTURING_FOCUS_13CFG.MF';parity=[]
   set_q([0.,0.]);frozen_values=cached(original_path);measured,_=calculate(original_path,'FROZEN_BINDING_CHECK')
   mask=np.array([r['operand']not in CONTROL for r in graph])
   if not np.allclose(frozen_values[mask],measured[mask],rtol=2e-9,atol=1e-8):raise ValueError('Native source differs from the published frozen evaluated operands')

   for scale in (0.,.1,-.1):
    set_q([scale,0.]);original,_=calculate(original_path,'SOURCE_PARITY');new,_=calculate(project/'FULL_SPAN.MF','SPAN_PARITY')
    pairs=[(i,j)for i,j in enumerate(spanreport['row_map_1based'][1:])if j is not None and i<len(graph)and graph[i]['operand']not in CONTROL]
    a=np.array([original[i]for i,j in pairs]);b=np.array([new[j-1]for i,j in pairs])
    if not np.allclose(a,b,rtol=2e-10,atol=1e-9):raise ValueError('Native span source-row parity')
    values=np.array([original[r['object_full2D_shift_row']-1]for r in focus['LCA_nodes']]);native_span=new[spanreport['row_map_1based'][proof['span_row']]-1]
    if abs(native_span-np.ptp(values))>1e-11:raise ValueError('Native MAXX/MINN span semantics')
    parity.append(dict(q=[scale,0.],mapped_rows=len(pairs),maximum_error=float(max(abs(a-b))),PV_mm=float(native_span)))
   set_q([0.,0.]);result['span']=dict(original_rows=len(graph),compiled_rows=len(fullspan),source_pairs=proof['source_pairs'],epigraph_bounds=proof['epigraph_inequalities'],parity=parity)
   save(OUT/'SPAN_VALIDATION.json',result['span']);update(status='span_native_parity_passed',rows=len(fullspan))
   latest={'path':None,'report':None,'compiled':None};audit_cache={};last_values={};evaluations=0
   class NativeProblem(EfficientSolveMixin):
    bounds=np.array([[-3.,3.],[-3.,3.]])
    def __init__(self,compiled,report,path):self.rows=compiled;self.report=report;self.path=path;self.loaded=False;self.cache={};self.roots=[]
    def prepare(self):
     for old,new in enumerate(self.report['row_map_1based']):
      if old==0 or new is None or old not in weighted or old in self.report['deactivated_weighted_rows']or not dynamic[old]:continue
      kind='geometry'if old in protected else classes[old]
      self.roots.append((old,new,kind))
    def evaluate(self,q):
     nonlocal evaluations
     key=np.asarray(q).tobytes()
     if key in self.cache:return self.cache[key]
     full=independent_q(q)
     if guard.batch([full])[0]is None:return None
     set_q(q)
     if not self.loaded:system.MFE.LoadMeritFunction(str(self.path));self.loaded=True
     value=float(system.MFE.CalculateMeritFunction())
     if not math.isfinite(value):return None
     system.MFE.SaveMeritFunction(str(OUT/'CURRENT_WORKING.MF'));values=cached(OUT/'CURRENT_WORKING.MF');gates={}
     for old,new,kind in self.roots:
      row=self.rows[new-1];v=values[int(row['params'][0])-1]
      if kind=='wfe'and v<=0.:return None
      slack=(v-row['target'])if row['operand']=='OPGT'else(row['target']-v)
      gates.setdefault(kind,[]).append(slack*math.sqrt(row['weight']))
     row=dict(gates={k:np.array(v)for k,v in gates.items()});self.cache[key]=row;evaluations+=1
     return row
    def batch(self,qs):return[self.evaluate(q)for q in qs]
    @staticmethod
    def slacks(row):return row['gates']
    def model(self,q):return guard_design.model(independent_q(q))
   def solve_compiled(q,compiled,report,iteration):
    path=OUT/f'ROUND_{iteration:02d}.MF';write_native(path,compiled)
    p=NativeProblem(compiled,report,path);p.prepare();latest.update(path=path,compiled=compiled,report=report)
    update(status='optimizing_native_working_set',round=iteration,rows=len(compiled),active_samples=len([g for g in groups if any(i in report['protected_rows']or(i not in report['deactivated_weighted_rows'])for i in g['rows'])]))
    solve=p.solve(OUT/f'solve-{iteration:02d}',iters=6,start_trust=.4,variant=.2,damping=1e-6,
       phase2_iterations=3,margin_keys=['wfe'],margin_scales={'wfe':1.},hard_gate_keys=['geometry'],
       initial_q=q,subproblem_backend='clarabel',difference_step=.002,verbose=False)
    return solve['q'],solve
   def audit(q):
    set_q(q);values,_=calculate(project/'FULL_SPAN.MF','CURRENT_AUDIT');latest['path']=None
    # Native loading changed; every following solve creates a new adapter and reloads.
    refs=spanreport['row_map_1based'];focal={r['wavelength_nm']:values[refs[r['object_full2D_shift_row']]-1]for r in focus['LCA_nodes']}
    physical=[]
    for old in protected:
     if not dynamic[old]:continue
     root=graph[old-1];v=values[refs[int(root['params'][0])]-1];g=(v-root['target'])if root['operand']=='OPGT'else(root['target']-v)
     physical.append(g*math.sqrt(root['weight']))
    physical_min=min(physical)if physical else 1.
    records=[]
    for group0 in groups:
     sample=group0['sample'];slacks=[]
     if sample['family']=='lca':
      d=focal[sample['i']]-focal[sample['j']];slacks=[(.0007-d)*math.sqrt(graph[group0['rows'][0]-1]['weight']),(.0007+d)*math.sqrt(graph[group0['rows'][0]-1]['weight'])]
     else:
      for old in group0['rows']:
       root=graph[old-1];new=refs[int(root['params'][0])]
       if new is None:raise ValueError('Full span audit lost a retained optical measurement')
       v=values[new-1];g=(v-root['target'])if root['operand']=='OPGT'else(root['target']-v);slacks.append(g*math.sqrt(root['weight']))
     records.append(dict(sample=sample,gates={sample['family']:np.array(slacks),'protected':physical_min}))
    audit_cache[np.asarray(q).tobytes()]=dict(native_PV_um=float(np.ptp(list(focal.values()))*1000),protected_min=physical_min)
    return records
   def final(q):
    model=guard_design.model(independent_q(q));save(OUT/'CANDIDATE_MODEL.json',model)
    with ProcessPoolExecutor(max_workers=ARGS.workers)as pool:checked=full_audit(model,executor=pool)
    save(OUT/'INDEPENDENT_FINAL_AUDIT.json',checked);return{k:v for k,v in checked.items()if k!='per_wave'}
   q0=np.array([(1/(1-.0001)-1)/.0001,0.])
   exchange=run_merit_exchange(work_graph,groups,initial,q0,protected_rows=sorted(protected),
      audit_identity='green8:'+wfe['source_model_file_sha256']+':'+wfe['scope_sha256'],
      solve_compiled=solve_compiled,audit=audit,strict_validator=final,output_directory=OUT/'exchange',
      budget={'rows':20000,'wavefront_queries':160},max_rounds=4,add_per_round=8,
      protected_samples=[s for s in initial if s['family']=='wfe'and s['field']==0.],
      promote_worst=True,drop_after=2,drop_slack=.4,max_active=150)
   q=np.array(exchange['q']);set_q(q)
   if latest['compiled']is not None:
    path=project/'ADAPTIVE_WORKING.MF';write_native(path,latest['compiled']);system.MFE.LoadMeritFunction(str(path));system.MFE.CalculateMeritFunction();system.MCE.SetCurrentConfiguration(1);system.SaveAs(str(project/'ADAPTIVE_WORKING.ZMX'));system.MFE.SaveMeritFunction(str(path))
   system.MFE.LoadMeritFunction(str(project/'FULL_SPAN.MF'));system.MFE.CalculateMeritFunction();system.MCE.SetCurrentConfiguration(1);system.SaveAs(str(project/'ADAPTIVE_FULL_SPAN.ZMX'));system.MFE.SaveMeritFunction(str(project/'FULL_SPAN.MF'))
   reload_checks=[]
   for stem,mf,compiled in [('ADAPTIVE_WORKING','ADAPTIVE_WORKING',latest['compiled']),('ADAPTIVE_FULL_SPAN','FULL_SPAN',fullspan)]:
    saved=cached(project/(mf+'.MF'))
    if not system.LoadFile(str(project/(stem+'.ZMX')),False):raise RuntimeError('Reload saved native case failed')
    system.MCE.SetCurrentConfiguration(1)
    if system.MCE.NumberOfConfigurations!=13:raise ValueError('Reload lost configurations')
    for surface,c,x in zip((21,24),curvatures,q):
     if abs(float(system.LDE.GetSurfaceAt(surface).Radius)-1/(c*(1+x*1e-4)))>1e-9:raise ValueError('Reload changed optimized curvature')
    system.MFE.LoadMeritFunction(str(project/(mf+'.MF')));system.MFE.CalculateMeritFunction();system.MFE.SaveMeritFunction(str(OUT/'RELOAD.MF'))
    measured=cached(OUT/'RELOAD.MF');mask=np.array([r['operand']not in CONTROL for r in compiled])
    if not np.allclose(saved[mask],measured[mask],rtol=2e-10,atol=1e-9):raise ValueError('Saved/reloaded native MF values differ')
    reload_checks.append(dict(file=stem+'.ZMX',rows=len(compiled),mapped_noncontrol_rows=int(mask.sum()),
        maximum_error=float(np.max(np.abs(saved[mask]-measured[mask]))),passed=True))
   save(OUT/'RELOAD_VALIDATION.json',reload_checks)
   result.update(exchange=exchange,native_evaluations=evaluations,working_WFE_guard=.98,q_initial=q0.tolist(),q_final=q.tolist(),
       source_prescription_unchanged=sha(sourcefile)==source_sha,seconds=time.perf_counter()-started,
       manufacturing_release=False,saved_reload_validation=reload_checks,passed=exchange['passed'])
   save(OUT/'RESULT.json',result);update(status='complete',passed=result['passed'],q=q.tolist(),evaluations=evaluations)
 finally:
  if app is not None:app.CloseApplication()

if __name__=='__main__':
 try:main()
 except Exception:
  if OUT is not None and OUT.exists():save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc()))
  raise
