"""Evaluate an optical model without Zemax and save the sampled results as JSON."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from .standalone import disk_quadrature
from .standalone_model import load_zmx
from .standalone_system import OpticalSystem, rectangular_pupil


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_model(model, sampling, *, allow_signed_segments=False, use_primary_field=False):
    system = OpticalSystem(model,allow_signed_segments=allow_signed_segments)
    pupil,weights = disk_quadrature(20,64) if sampling=='gq20' else rectangular_pupil(int(sampling[4:]))
    rows = []
    for field_index,field in enumerate(system.fields,1):
        for wave,wavelength in enumerate(system.wavelengths):
            row = dict(field_index=field_index,field_xy=field.tolist(),wave_index=wave+1,wavelength_um=float(wavelength))
            try:
                value,_,_ = system.evaluate(field,wave,pupil,weights,use_primary_field=use_primary_field)
                row.update(value,status='calculated')
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                row.update(status='failed',error=str(error))
            rows.append(row)
    calculated = [r for r in rows if r['status']=='calculated']
    negative = [r for r in calculated if r['rays_with_negative_segments']]
    failures = len(rows)-len(calculated)
    status = 'incomplete' if failures else ('signed_comparison_only' if negative else 'calculated')
    return dict(status=status,source_sha256=model.get('source_sha256'),field_type=model['field_type'],
        field_reference='primary' if use_primary_field else 'measurement',image_z_mm=float(system.image_z_mm),
        sampling=sampling,pupil_samples_per_group=len(pupil),failed_groups=failures,
        groups_with_negative_segments=len(negative),allow_signed_segments=allow_signed_segments,
        catalogs_resolved_outside_header=model.get('catalogs_resolved_outside_header',[]),
        worst_sampled_wfe_waves=max((r['wavefront_rms_waves']for r in calculated),default=None),
        worst_sampled_rms_spot_um=max((r['spot_rms_um']for r in calculated),default=None),rows=rows)


def evaluate_optional(model, report, args):
    system = OpticalSystem(model, allow_signed_segments=args.allow_signed_segments)
    if args.aberrations or args.plot_dir:
        from .standalone_aberrations import analyze_aberrations

        pupil, weights = (disk_quadrature(20, 64) if args.sampling == 'gq20'
                          else rectangular_pupil(int(args.sampling[4:])))
        try:
            report['aberrations'] = analyze_aberrations(system, pupil=pupil, weights=weights,
                                                        fan_samples=args.fan_samples)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            report['aberrations'] = {'status': 'failed', 'error': str(error)}
    if args.mtf:
        from .standalone_mtf_workflow import calculate_mtf

        mtf_rows = []
        spectra = [None, *range(len(system.wavelengths))] if args.mtf_all_waves else [None]
        for i, field in enumerate(system.fields, 1):
            for wave in spectra:
                entry = {'field_index': i, 'field_xy': field.tolist(),
                         'wave_index': None if wave is None else wave+1}
                try:
                    entry.update(calculate_mtf(system, field, samples=args.mtf_samples, wave=wave,
                                               method=args.mtf_method,
                                               maximum_frequency_lpmm=args.mtf_max_frequency),
                                 status='calculated')
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    entry.update(status='failed', error=str(error))
                mtf_rows.append(entry)
        report['mtf'] = mtf_rows
    aberrations = report.get('aberrations', {})
    if args.plot_dir:
        write_field_plots(system, aberrations, report, args)
    summary = aberrations.get('summary', {})
    mtf = report.get('mtf', [])
    failed = (aberrations.get('status') == 'failed' or summary.get('fan_failures', 0)
              or summary.get('focus_failures', 0) or any(row['status'] == 'failed' for row in mtf)
              or any(row['status'] == 'incomplete' for row in report.get('field_plots', [])))
    signed = (summary.get('signed_comparison_only')
              or any('signed_comparison_only' in row.get('warnings', []) for row in mtf)
              or any(row['status'] == 'signed_comparison_only' for row in report.get('field_plots', [])))
    if failed:
        report['status'] = 'incomplete'
    elif signed and report['status'] == 'calculated':
        report['status'] = 'signed_comparison_only'


def write_field_plots(system, aberrations, report, args):
    from .standalone_plots import write_ray_fan_svg, write_spot_svg
    from .standalone_spot import spot_diagram

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, field in enumerate(system.fields, 1):
        entry = {'field_index': i, 'field_xy': field.tolist(), 'status': 'calculated'}
        try:
            spot = spot_diagram(system, field, samples=args.spot_samples)
            path = args.plot_dir / f'field_{i:03d}_spot.svg'
            write_spot_svg(path, spot, field_index=i)
            entry.update(spot_svg=str(path), spot=spot, status=spot['status'])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            entry.update(status='incomplete', spot_error=str(error))
        if aberrations.get('status') != 'failed':
            path = args.plot_dir / f'field_{i:03d}_ray_fan.svg'
            write_ray_fan_svg(path, aberrations, field_index=i)
            entry['ray_fan_svg'] = str(path)
        entries.append(entry)
    report['field_plots'] = entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model',type=Path,help='Text ZMX or normalized prescription JSON')
    source.add_argument('--demo',action='store_true',help='Run the self-contained synthetic singlet')
    parser.add_argument('--glass-dir',type=Path,help='Directory of authorized AGF files for a text ZMX')
    parser.add_argument('--sampling',choices=('gq20','rect32','rect64'),default='gq20')
    parser.add_argument('--field-reference',choices=('measurement','primary'),default='measurement')
    parser.add_argument('--allow-signed-segments',action='store_true',help='Replay signed paths for numerical comparison and flag geometry')
    parser.add_argument('--optimize',action='store_true',help='Refine the synthetic demo using optional SciPy')
    parser.add_argument('--aberrations', action='store_true', help='Evaluate axial and lateral color, ray fans, field curvature, astigmatism, and distortion')
    parser.add_argument('--fan-samples', type=int, default=101, help='Odd number of samples per transverse ray fan')
    parser.add_argument('--mtf', action='store_true', help='Calculate image-space MTF curves')
    parser.add_argument('--mtf-method', choices=('auto', 'fft', 'geometric'), default='auto',
                        help='Automatically select a method, or request scalar FFT or geometric MTF')
    parser.add_argument('--mtf-max-frequency', type=float, help='Override the maximum image frequency in lp/mm')
    parser.add_argument('--mtf-samples', type=int, default=256, help='Even pupil grid size per MTF axis')
    parser.add_argument('--mtf-all-waves', action='store_true', help='Include every monochromatic MTF curve as well as the polychromatic curve')
    parser.add_argument('--plot-dir', type=Path, help='Write spot and ray fan SVGs for each saved field')
    parser.add_argument('--spot-samples', type=int, choices=(32, 64), default=32,
                        help='RECT sampling parameter for the spot diagram (2n by 2n grid)')
    parser.add_argument('--out',type=Path,help='Write the evaluation report')
    parser.add_argument('--neutral-out',type=Path,help='Explicitly export a private normalized model JSON')
    parser.add_argument('--manufacturing-policy', type=Path,
                        help='Apply a project nominal-geometry policy before optical evaluation')
    parser.add_argument('--manufacturing-only', action='store_true',
                        help='Write the nominal-geometry audit without tracing optical performance')
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in (args.model,args.out,args.neutral_out,args.manufacturing_policy)
             if path is not None]
    if len(paths)!=len(set(paths)):
        parser.error('Input model, policy, report, and model export must be different files')
    if args.optimize and not args.demo:
        parser.error('--optimize is for the synthetic demo')
    if args.demo and args.neutral_out:
        parser.error('--neutral-out requires an input model')
    if args.demo and (args.aberrations or args.mtf or args.plot_dir):
        parser.error('--aberrations, --mtf, and --plot-dir require an input model')
    if args.demo and (args.manufacturing_only or args.manufacturing_policy):
        parser.error('Manufacturing audit requires an input model')
    if args.manufacturing_only and (args.aberrations or args.mtf or args.plot_dir or args.allow_signed_segments):
        parser.error('--manufacturing-only cannot be combined with optical analyses or signed-path replay')
    if not args.mtf and (args.mtf_all_waves or args.mtf_method != 'auto' or args.mtf_max_frequency is not None):
        parser.error('MTF options require --mtf')
    if (args.aberrations or args.plot_dir) and (args.fan_samples < 3 or args.fan_samples % 2 != 1):
        parser.error('--fan-samples must be an odd number of at least 3')
    if args.mtf and (not 16 <= args.mtf_samples <= 1024 or args.mtf_samples % 2):
        parser.error('--mtf-samples must be an even number from 16 to 1024')
    try:
        if args.demo:
            from .standalone_demo import run
            report = dict(run(args.optimize),status='calculated',failed_groups=0)
        else:
            extension = args.model.suffix.lower()
            if extension=='.zmx':
                if args.glass_dir is None:
                    parser.error('--glass-dir is required for a text ZMX')
                model = load_zmx(args.model,args.glass_dir)
            elif extension=='.json':
                raw = args.model.read_bytes()
                model = json.loads(raw.decode('utf-8-sig'))
                model.setdefault('source_sha256',hashlib.sha256(raw).hexdigest())
            else:
                parser.error('Use a text ZMX or normalized JSON file')
            manufacturing = None
            policy_raw = None
            if args.manufacturing_only or args.manufacturing_policy:
                from .standalone_manufacturing import audit_manufacturing, audit_ray_envelope

                policy_raw = args.manufacturing_policy.read_bytes() if args.manufacturing_policy else None
                policy = json.loads(policy_raw.decode('utf-8-sig')) if policy_raw is not None else None
                manufacturing = audit_manufacturing(model, policy)
                if not args.manufacturing_only and manufacturing['passed']:
                    pupil, _ = (disk_quadrature(20, 64) if args.sampling == 'gq20'
                                else rectangular_pupil(int(args.sampling[4:])))
                    manufacturing = audit_ray_envelope(OpticalSystem(model), manufacturing, pupil,
                                                       use_primary_field=args.field_reference == 'primary')
            if args.manufacturing_only:
                report = {'status': manufacturing['status'], 'manufacturing': manufacturing}
            elif manufacturing is not None and not manufacturing['passed']:
                report = {'status': 'manufacturing_rejected', 'manufacturing': manufacturing}
            else:
                report = evaluate_model(model,args.sampling,allow_signed_segments=args.allow_signed_segments,
                                        use_primary_field=args.field_reference=='primary')
                if manufacturing is not None:
                    report['manufacturing'] = manufacturing
                if args.aberrations or args.mtf or args.plot_dir:
                    evaluate_optional(model, report, args)
            report['input_file_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            if policy_raw is not None:
                report['manufacturing_policy_file_sha256'] = hashlib.sha256(policy_raw).hexdigest()
            if args.neutral_out:
                write_json(args.neutral_out,model)
        if args.out:
            write_json(args.out,report)
            print(f"{report['status']} | {args.out}")
        else:
            print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
        return 2 if report['status'] in ('incomplete', 'invalid', 'manufacturing_rejected') else 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError) as error:
        print(f'Optical calculation error: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
