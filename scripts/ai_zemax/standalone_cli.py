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
    if args.aberrations:
        from .standalone_aberrations import analyze_aberrations

        pupil, weights = (disk_quadrature(20, 64) if args.sampling == 'gq20'
                          else rectangular_pupil(int(args.sampling[4:])))
        try:
            report['aberrations'] = analyze_aberrations(system, pupil=pupil, weights=weights,
                                                        fan_samples=args.fan_samples)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            report['aberrations'] = {'status': 'failed', 'error': str(error)}
    if args.mtf:
        from .standalone_mtf import diffraction_mtf

        mtf_rows = []
        spectra = [None, *range(len(system.wavelengths))] if args.mtf_all_waves else [None]
        for i, field in enumerate(system.fields, 1):
            for wave in spectra:
                entry = {'field_index': i, 'field_xy': field.tolist(),
                         'wave_index': None if wave is None else wave+1}
                try:
                    entry.update(diffraction_mtf(system, field, samples=args.mtf_samples, wave=wave),
                                 status='calculated')
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    entry.update(status='failed', error=str(error))
                mtf_rows.append(entry)
        report['mtf'] = mtf_rows
    aberrations = report.get('aberrations', {})
    summary = aberrations.get('summary', {})
    mtf = report.get('mtf', [])
    failed = (aberrations.get('status') == 'failed' or summary.get('fan_failures', 0)
              or summary.get('focus_failures', 0) or any(row['status'] == 'failed' for row in mtf))
    signed = (summary.get('signed_comparison_only')
              or any('signed_comparison_only' in row.get('warnings', []) for row in mtf))
    if failed:
        report['status'] = 'incomplete'
    elif signed and report['status'] == 'calculated':
        report['status'] = 'signed_comparison_only'


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
    parser.add_argument('--mtf', action='store_true', help='Calculate scalar FFT MTF curves')
    parser.add_argument('--mtf-samples', type=int, default=256, help='Even pupil grid size per MTF axis')
    parser.add_argument('--mtf-all-waves', action='store_true', help='Include every monochromatic MTF curve as well as the polychromatic curve')
    parser.add_argument('--out',type=Path,help='Write the evaluation report')
    parser.add_argument('--neutral-out',type=Path,help='Explicitly export a private normalized model JSON')
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in (args.model,args.out,args.neutral_out) if path is not None]
    if len(paths)!=len(set(paths)):
        parser.error('Input model, report, and model export must be different files')
    if args.optimize and not args.demo:
        parser.error('--optimize is for the synthetic demo')
    if args.demo and args.neutral_out:
        parser.error('--neutral-out requires an input model')
    if args.demo and (args.aberrations or args.mtf):
        parser.error('--aberrations and --mtf require an input model')
    if args.mtf_all_waves and not args.mtf:
        parser.error('--mtf-all-waves requires --mtf')
    if args.aberrations and (args.fan_samples < 3 or args.fan_samples % 2 != 1):
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
            report = evaluate_model(model,args.sampling,allow_signed_segments=args.allow_signed_segments,
                                    use_primary_field=args.field_reference=='primary')
            if args.aberrations or args.mtf:
                evaluate_optional(model, report, args)
            report['input_file_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            if args.neutral_out:
                write_json(args.neutral_out,model)
        if args.out:
            write_json(args.out,report)
            print(f"{report['status']} | {args.out}")
        else:
            print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
        return 2 if report['status'] == 'incomplete' else 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError) as error:
        print(f'Optical calculation error: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
