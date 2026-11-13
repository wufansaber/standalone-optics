"""One owned native process, copies only; no optimization or optical acceptance."""
import os,sys,json,time,math,shutil,hashlib,traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,gzip
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-project',type=Path,required=True)
parser.add_argument('--compiled',type=Path,required=True)
parser.add_argument('--graphs',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--install',type=Path,required=True)
parser.add_argument('--pythonnet-dir',type=Path)
parser.add_argument('--cpu-count',type=int)
args=parser.parse_args()
CODE=Path(__file__).resolve().parents[2]
D=args.source_project.resolve();OUT=args.out.resolve();COMPILED=args.compiled.resolve()
OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(CODE))
if args.pythonnet_dir:sys.path.insert(0,str(args.pythonnet_dir.resolve()))
import numpy as np,psutil
from contextlib import nullcontext
from scripts.ai_zemax.standalone_merit_graph import normalize
load=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')

def cached(path):
 lines=path.read_text(encoding='utf-16').splitlines()[1:]
 result=[]
 for line in lines:
  items=line.split()
  if not items:continue
  if len(items)!=12:raise ValueError('Unexpected native MF record')
  result.append(float(items[-1]))
 result=np.array(result)
 if not np.isfinite(result).all():raise ValueError('Invalid cached operands')
 return result

def main():
 start=time.monotonic();project=OUT/'project';project.mkdir(exist_ok=False)
 shutil.copytree(D/'GLASSCAT',project/'GLASSCAT')
 source_zmx=D/'GREEN8_FULL_TARGETS_13CFG.ZMX';frozen_sha=sha(source_zmx)
 for name in ['OH9AD9D2.AGF','FIXWIN.AGF']:
  assert sha(project/'GLASSCAT'/name)==sha(D/'GLASSCAT'/name)
 assert sha(project/'GLASSCAT/OH9AD9D2.AGF')=='9ad9d23da983ef105ec02dd6c9ad55f124163ad604a074e53f753dc351c49e86'
 copied=project/'BASELINE.ZMX';shutil.copy2(source_zmx,copied)
 def graph(kind):
  path=args.graphs/('MF_'+kind+'_GRAPH.json');data=path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes());return json.loads(data)
 graphs={kind:graph(kind) for kind in ['CORE','FULL']}
 original_files={'CORE':D/'GREEN8_CORE_GEOMETRY.MF','FULL':D/'GREEN8_MANUFACTURING_FOCUS_13CFG.MF'}
 names=['CORE_EQUIVALENT','FULL_EQUIVALENT','WORKING_CORE','WORKING']
 reports={n:load(COMPILED/(n+'_REPORT.json'))for n in names};app=None
 result=dict(native_version='OpticStudio 2022 R2.02',source_ZMX_sha256=frozen_sha,
   source_model_sha256=load(COMPILED/'SUMMARY.json')['source_model_sha256'],
   optimization_started=False,new_optical_design_accepted=False,manufacturing_release=False,
   physical_prescription_unchanged=True,verifier_sha256=sha(Path(__file__)),points=[],timings={},reloads=[])
 def update(**kw):print(json.dumps(kw),flush=True);save(OUT/'NATIVE_STATE.json',dict(kw,seconds=time.monotonic()-start))
 try:
  with nullcontext():
   from pythonnet import load as netload;netload('netfx');import clr
   install=args.install.resolve()
   clr.AddReference(str(install/'ZOSAPI_NetHelper.dll'));import ZOSAPI_NetHelper
   assert ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(str(install))
   clr.AddReference(str(install/'ZOSAPI_Interfaces.dll'));clr.AddReference(str(install/'ZOSAPI.dll'));import ZOSAPI as api
   before={p.pid for p in psutil.process_iter()};conn=api.ZOSAPI_Connection();conn.ConnectionTimeoutSeconds=45;app=conn.CreateNewApplication()
   if app is None or not app.IsValidLicenseForAPI:raise RuntimeError('Native API license')
   for p in psutil.process_iter(['name']):
    if p.pid not in before and 'opticstudio'in(p.info['name']or'').lower():
     if args.cpu_count is not None:p.cpu_affinity(list(range(args.cpu_count)))
   system=app.PrimarySystem
   if not system.LoadFile(str(copied),False):raise ValueError('Load copied original')
   if not system.IsProjectDirectory and not system.ConvertToProjectDirectory(str(project)):raise ValueError('Private project directory')
   system.UpdateFileLists()
   if not Path(str(system.SystemFile)).resolve().is_relative_to(project):raise ValueError('Outside copied project')
   if system.MCE.NumberOfConfigurations!=13:raise ValueError('Require all 13 configurations')
   def fingerprint():
    system.MCE.SetCurrentConfiguration(1);system.UpdateStatus()
    return [(float(s.Radius),float(s.Thickness),str(s.Material),float(s.Conic),float(s.SemiDiameter))
            for s in [system.LDE.GetSurfaceAt(i)for i in range(system.LDE.NumberOfSurfaces)]]
   original_fingerprint=fingerprint()
   def native(path,tag,save_values=True):
    then=time.perf_counter();system.MFE.LoadMeritFunction(str(path));loading=time.perf_counter()-then
    then=time.perf_counter();value=float(system.MFE.CalculateMeritFunction());seconds=time.perf_counter()-then
    if not math.isfinite(value):raise ValueError('Nonfinite merit value')
    vals=None
    if save_values:
     saved=OUT/(tag+'_EVALUATED.MF');system.MFE.SaveMeritFunction(str(saved));vals=cached(saved)
     if len(vals)!=system.MFE.NumberOfOperands:raise ValueError('Cached value count')
    return vals,dict(load_seconds=loading,evaluation_seconds=seconds,merit=value,rows=int(system.MFE.NumberOfOperands))
   radius=float(system.LDE.GetSurfaceAt(21).Radius)
   for point,scale in [('baseline',1.),('radius_plus',1.00001),('radius_minus',.99999)]:
    system.MCE.SetCurrentConfiguration(1);system.LDE.GetSurfaceAt(21).Radius=radius*scale
    source_values={};point_report=dict(point=point,radius_scale=scale,comparisons=[])
    for kind in ['CORE','FULL']:
     update(status='evaluating',point=point,graph=kind)
     values,timing=native(original_files[kind],point+'_'+kind);source_values[kind]=values
     if point=='baseline':
      published=cached(original_files[kind]);mask=np.array([r['operand']not in {'CONF','PRIM'}for r in graphs[kind]])
      error=float(np.max(np.abs(values[mask]-published[mask])))
      if not np.allclose(values[mask],published[mask],rtol=2e-9,atol=1e-8):raise ValueError('Copied native system differs from frozen evaluated source')
      point_report[kind+'_published_cache_max_error']=error
     for name in [n for n in names if ('CORE' in n)==(kind=='CORE')]:
      update(status='evaluating_compact',point=point,graph=name)
      new,newtime=native(COMPILED/(name+'.MF'),point+'_'+name);rep=reports[name]
      pairs=[(i,j)for i,j in enumerate(rep['row_map_1based'][1:])if j is not None and graphs[kind][i]['operand']not in {'CONF','PRIM'}]
      a=np.array([values[i]for i,j in pairs]);b=np.array([new[j-1]for i,j in pairs]);err=float(np.max(np.abs(a-b)))
      if not np.allclose(a,b,rtol=2e-10,atol=1e-9):raise ValueError('Native mapped-row parity failed: '+name)
      if rep['full_objective_preserved'] and abs(timing['merit']-newtime['merit'])>1e-11:raise ValueError('Equivalent MF objective changed')
      roots=[i for i,j in pairs if graphs[kind][i]['weight']>0 and i+1 not in rep['deactivated_weighted_rows']]
      row=dict(name=name,mapped_noncontrol_rows=len(pairs),weighted_roots_checked=len(roots),maximum_absolute_error=err,source=timing,compiled=newtime)
      point_report['comparisons'].append(row)
      if point=='baseline':
       # Save a complete native file with its compiled MF, then restore source
       # geometry before producing any other case artifact.
       system.MCE.SetCurrentConfiguration(1);system.SaveAs(str(project/(name+'.ZMX')))
       system.MFE.SaveMeritFunction(str(project/(name+'.MF')))
    result['points'].append(point_report)
    save(OUT/'NATIVE_VALIDATION.partial.json',result)
   system.MCE.SetCurrentConfiguration(1);system.LDE.GetSurfaceAt(21).Radius=radius
   if fingerprint()!=original_fingerprint:raise ValueError('Physical prescription changed after restoration')
   # Repeated native Calculate calls, excluding load, save and Python parsing.
   for kind,name in [('CORE','WORKING_CORE'),('FULL','WORKING')]:
    pairs=[]
    for repeat in range(3):
     order=[('original',original_files[kind]),('working',COMPILED/(name+'.MF'))]
     if repeat%2:order.reverse()
     rec={}
     for label,path in order:
      _,timing=native(path,'timing',False);rec[label]=timing['evaluation_seconds']
     pairs.append(rec)
    original=float(np.median([r['original']for r in pairs]));working=float(np.median([r['working']for r in pairs]))
    result['timings'][kind]=dict(paired_repetitions=pairs,original_median_seconds=original,
       working_median_seconds=working,speed_ratio=original/working,scope='CalculateMeritFunction only; baseline, same native instance; no Jacobian measured')
   for name in names:
    path=project/(name+'.ZMX')
    if not system.LoadFile(str(path),False):raise ValueError('Reload saved native artifact')
    if fingerprint()!=original_fingerprint or system.MCE.NumberOfConfigurations!=13:raise ValueError('Reload geometry/configuration changed')
    if system.MFE.NumberOfOperands!=reports[name]['output']['counts']['rows']:raise ValueError('Reload MF count changed')
    value=float(system.MFE.CalculateMeritFunction())
    if not math.isfinite(value):raise ValueError('Invalid reloaded merit')
    result['reloads'].append(dict(name=name,rows=int(system.MFE.NumberOfOperands),configurations=13,merit=value,sha256=sha(path),passed=True))
   if sha(source_zmx)!=frozen_sha:raise ValueError('Original source file was modified')
   result.update(passed=True,source_files_unchanged=True,total_seconds=time.monotonic()-start)
   save(OUT/'NATIVE_VALIDATION.json',result);update(status='passed',timings=result['timings'])
 finally:
  if app is not None:app.CloseApplication()

if __name__=='__main__':
 try:main()
 except Exception:
  save(OUT/'NATIVE_FAILURE.json',dict(traceback=traceback.format_exc()));raise
