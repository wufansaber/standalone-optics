"""Real-ray multi-scenario design with one bounded image-plane compensator.

All fields/wavelengths of each disturbed assembly share its one focus control.
Nominal assembly has no compensation. Deterministic scenarios are not yield.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.ai_zemax.standalone_scenarios import ScenarioProblem

spec = importlib.util.spec_from_file_location('scenario_optics', ROOT/'examples/local-optimizer-20261109/run.py')
optics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optics)
SCENARIOS = {f'assembly-{i}': {'error': error} for i, error in enumerate(optics.SCENARIOS)}
CONTROL_BOUNDS = {key: ([] if i == 0 else [[-.02, .02]]) for i, key in enumerate(SCENARIOS)}


def make_problem(samples):
    problems = {key: optics.Singlet(samples, scenarios=(payload['error'],)) for key, payload in SCENARIOS.items()}

    def effective(design, controls):
        q = design.copy()
        if len(controls):
            q[3] += controls[0]  # Physical image plane translation in millimetres.
        return q

    def evaluate(design, payload, controls):
        key = next(key for key, value in SCENARIOS.items() if value == payload)
        p = problems[key]; q = effective(design, controls)
        row = p.evaluate(q)
        if row is None or not all(r['passed'] for r in optics.mechanical_audits(q, (payload['error'],))):
            return None
        return dict(gates=p.slacks(row), vectors={name: row[name] for name in ('wave_vector', 'spot_vector')}, raw=row)

    def derivative(design, payload, controls, row):
        key = next(key for key, value in SCENARIOS.items() if value == payload)
        p = problems[key]; gates = row['gates']; keys = tuple(gates)
        local = p.analytic_jacobian(effective(design, controls), row['raw'], keys)
        def extend(j):
            return np.concatenate((j, j[..., 3:4]), axis=-1) if len(controls) else j
        blocks, start = {}, 0
        for name, value in gates.items():
            blocks[name] = extend(local['slacks'][start:start+value.size].reshape(value.shape+(4,)))
            start += value.size
        return dict(gates=blocks, vectors={name: extend(j) for name, j in local['vectors'].items()})

    return ScenarioProblem(optics.Singlet.bounds, SCENARIOS, evaluate,
        optics.Singlet(samples).model, compensator_bounds=CONTROL_BOUNDS,
        local_jacobian=derivative, local_norm_specs=optics.Singlet(samples).local_norm_specs)


def audit(saved):
    model = saved['design']; front, rear = model['surface_rows'][1:3]
    q = np.array([(front['curvature_mm_inv']-1./30.)/.005,
        (rear['curvature_mm_inv']+1./45.)/.005, front['even_coefficients'][1]/1e-5,
        model['vertices_mm'][-1]-optics.BASE_IMAGE])
    records = []
    for item in saved['scenarios']:
        adjusted = q.copy()
        if item['controls']:
            adjusted[3] += item['controls'][0]
        error = SCENARIOS[item['id']]['error']
        rows = optics.audit_records(adjusted, scenarios=(error,))
        records.append(dict(id=item['id'], controls=item['controls'],
            passed=all(r.get('valid', True) and min(r['gates'].values()) >= 0. for r in rows),
            gate_minima={name: min(r['gates'][name] for r in rows) for name in rows[0]['gates']}))
    return dict(passed=all(r['passed'] for r in records), deterministic_scenarios=records,
                statistical_yield_estimated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    p = make_problem([dict(field_mm=float(h), pupil_rule='gq4') for h in (0., .5, 1.)])
    started = perf_counter()
    result = p.solve(args.out/'solve', 35, .5, .2, 1e-5, phase2_iterations=6,
        jacobian_provider=p.jacobian, subproblem_backend='clarabel', strict_validator=audit, verbose=False)
    saved = p.model(np.array(result['q']))
    report = dict(seconds=perf_counter()-started, solver=result,
        physical_design=saved, controls_definition='One image-plane axial shift per disturbed assembly, ±0.02mm; none for nominal',
        final_audit=result['final_acceptance'])
    (args.out/'scenario-compensation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=result['final_acceptance']['passed'], seconds=report['seconds'],
        controls=saved['scenarios'])), flush=True)
    if not result['final_acceptance']['passed']:
        raise SystemExit('Independent compensated-scenario audit failed')


if __name__ == '__main__':
    main()
