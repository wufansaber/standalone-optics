"""Same-start real-ray SLSQP/SOCP comparison; includes dense and physical gates."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import clarabel

spec = importlib.util.spec_from_file_location('public_synthetic_optical_case',
    ROOT/'examples/local-optimizer-20261109/run.py')
optics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(case='public_synthetic_singlet', environment=dict(
        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
        clarabel=clarabel.__version__, platform=platform.platform()),
        acceptance=dict(wfe_waves=.25, spot_rms_mm=.018, chief_height_mm=.08,
            scenarios=optics.SCENARIOS, final_fields=11, pupil_rules=['GQ12x48', 'GQ20x48']),
        controls=dict(same_start=True, same_derivatives='analytic', same_iteration_budget=35,
            same_trust=.5, same_margin_iterations=6, same_independent_audit=True,
            comparison_includes='trace, derivatives, model building, solve, dense and physical acceptance'), runs=[])
    for repeat in range(args.repeats):
        order = ('slsqp', 'clarabel') if repeat % 2 == 0 else ('clarabel', 'slsqp')
        for backend in order:
            result = optics.run_one('analytic', args.out/f'{backend}-{repeat}', subproblem_backend=backend)
            report['runs'].append(dict(repeat=repeat, solver=backend, passed=result['passed'],
                seconds=result['seconds'], q=result['q'], final_optics=result['final_optics'],
                counts=result['counts'], exchange_rounds=len(result['history']),
                final_geometry_passed=all(r['passed'] for r in result['scenario_manufacturing'])))
            print(json.dumps(report['runs'][-1]), flush=True)
    report['all_passed'] = all(r['passed'] for r in report['runs'])
    report['median_wall_seconds'] = {backend: float(np.median([
        r['seconds'] for r in report['runs'] if r['solver'] == backend])) for backend in ('slsqp', 'clarabel')}
    report['speedup_scope'] = 'This synthetic case and environment only; no high-NA or global speedup claim'
    (args.out/'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if not report['all_passed']:
        raise SystemExit('At least one independent dense/physical audit failed')
    print(json.dumps({k: report[k] for k in ('all_passed', 'median_wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
