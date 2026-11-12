"""Publish only whitelisted dimensionless metrics from a verified private run."""
import argparse,json,math,hashlib
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def summary(checkpoints, native_gate):
    gate = read(native_gate)
    if not (gate.get('status') == 'complete'
            and gate.get('all_original_standalone_and_native_goals_passed') is True
            and gate.get('owned_application_closed') is True):
        raise ValueError('Complete closed final native verification is required')
    for name in ('NATIVE_FORMAL_FOCUS.json','NATIVE_RWRE32_GREEN17.json','prepared/NATIVE_RESULT.json','prepared/SOURCE_HASHES.json'):
        actual=hashlib.sha256((Path(native_gate).parent/name).read_bytes()).hexdigest()
        if gate['final_evidence_sha256'].get(name)!=actual:
            raise ValueError('Final native evidence hash mismatch')
    rows=[]
    for index, path in enumerate(checkpoints):
        cp=read(path)
        values={key:float(cp['joint_components'][key]) for key in ('WFE','LCA','FC')}
        if not all(math.isfinite(v) and v>=0 for v in values.values()):
            raise ValueError('Nonfinite or negative normalized metric')
        physical=bool(cp['high_precision_body']['passed']
                      and cp['independent_declared_body_certificate']['passed']
                      and cp['QA65']['ordinary65_FO_passed']
                      and cp['QA65']['objective_D_passed']
                      and cp['QA65']['physical_all15']
                      and cp['QA65']['no_pupil_or_field_drop']
                      and cp['QA65']['negative_segment_rays']==0
                      and not cp['QA65']['failures'])
        if not physical:
            raise ValueError('A supplied checkpoint did not pass original physical checks')
        rows.append(dict(checkpoint_index=index,normalized=values,
                         Phi=max(values.values()),original_physical_checks_passed=True))
    if not rows or rows[-1]['Phi']>1 or read(checkpoints[-1])['model_sha256']!=gate['source']['model_sha256']:
        raise ValueError('Final point must meet all targets and match native verification')
    nf=read(Path(native_gate).parent/'NATIVE_FORMAL_FOCUS.json')
    if not(nf['model_sha256']==gate['source']['model_sha256']
           and nf['full_original_focus_passed']and nf['standalone_parity_passed']
           and nf['statistics']['invalid_rays']==0 and nf['statistics']['vignetted_rays']==0):
        raise ValueError('Native focus evidence is incomplete or belongs to another model')
    standard=read(Path(native_gate).parent/'prepared/NATIVE_RESULT.json')
    rectangular=read(Path(native_gate).parent/'NATIVE_RWRE32_GREEN17.json')
    if not(standard['model_sha256']==gate['source']['model_sha256']
           and standard['same_prescription_numerics_passed']
           and standard['native_green_RWCE20_limits_passed']
           and standard['complete15_physical_native_batch_passed']
           and all(r['passed']and 0.<=r['RWRE32_waves']<=r['cap_waves']for r in rectangular)):
        raise ValueError('Native wavefront or physical validation is incomplete')
    native=dict(original_same_prescription_verification_passed=True,
                focus_sampling=nf['sampling'],spectral_nodes=len(nf['wavelengths_nm']),
                FC_grid_fields=nf['FC_grid_fields'],local_extrema_searched=True,
                no_failed_or_vignetted_rays=True,
                normalized_WFE_RWCE20_max=max(r['native_RWCE20']/r['ceiling_waves']for r in standard['RWCE20']),
                normalized_WFE_RWRE32_max=max(r['RWRE32_waves']/r['cap_waves']for r in rectangular),
                native_vs_standalone_focus_max_um=nf['standalone_focus_parity_max_um'],
                actual_magnification_max_absolute_error=nf['magnification_parity_max'])
    return dict(schema='anonymous_joint_optical_minimax_case_v1',
                final_all_declared_goals_passed=True,history=rows,native_validation=native,
                contains_prescription_or_catalog=False,global_optimum_claimed=False,
                hardware_performance_claimed=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--native-gate',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=summary(args.checkpoints,args.native_gate)
    if args.out.exists():raise FileExistsError('Fresh sanitized output required')
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
