"""Reproduce a public synthetic singlet optimization without Zemax or CUDA.

The four design variables are front/rear curvature, front A4 and image focus.
Launch rays target the fixed z=64 mm stop, so their aiming is independent of
those variables. The moving paraxial exit-pupil reference and direction-based
tilt fit are differentiated. Thresholds are this demonstration's allocations,
not the unrelated UV355 objective's acceptance specification.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from ai_zemax.standalone import (Surface, disk_quadrature, paraxial_transfer,
                                 reference_opd_mm, trace, wavefront_rms)
from ai_zemax.standalone_derivatives import (RayParameter, paraxial_transfer_with_jacobian,
    reference_opd_with_jacobian, trace_with_jacobian, wavefront_rms_with_jacobian)
from ai_zemax.standalone_constraint_exchange import run_constraint_exchange
from ai_zemax.standalone_local_solver import EfficientSolveMixin
from ai_zemax.standalone_manufacturing import audit_manufacturing, audit_ray_envelope
from ai_zemax.standalone_system import OpticalSystem


POLICY = dict(schema_version=1, policy_id='SYNTHETIC_SINGLET_DEMONSTRATION',
    sources=['Illustrative engineering allocations; not supplier capability guarantees'],
    radial_samples=257, all_spherical=False, ambient_index=1.,
    defaults=dict(minimum_center_thickness_mm=2., minimum_radial_thickness_mm=1.5,
                  minimum_air_gap_mm=.3, minimum_rim_gap_mm=.3,
                  minimum_clear_to_mechanical_radial_margin_mm=.5,
                  maximum_diameter_mm=15., maximum_surface_slope=1.,
                  center_thickness_budget_mm=.02, radial_thickness_budget_mm=.02,
                  air_gap_budget_mm=.02, rim_gap_budget_mm=.02),
    surfaces={'2': dict(clear_radius_mm=5., mechanical_radius_mm=6.),
              '3': dict(clear_radius_mm=5., mechanical_radius_mm=6.)},
    lenses=[dict(front_surface=2, rear_surface=3)])
PARAMETERS = (RayParameter(1, 'curvature_mm_inv', scale=.005),
              RayParameter(2, 'curvature_mm_inv', scale=.005),
              RayParameter(1, 'even_coefficient', 1, 1e-5),
              RayParameter(3, 'vertex_mm', scale=1.))
WAVES = (.54, .55, .56)
INDICES = (1.5175, 1.5168, 1.5162)
BASE_IMAGE = 143.98788423757497
SCENARIOS = ((0., 0.), (2e-5, 0.), (-2e-5, 0.), (0., .005), (0., -.005))


class Singlet(EfficientSolveMixin):
    bounds = np.array([[-2., 2.], [-2., 2.], [-2., 2.], [-5., 5.]])
    # A deliberately useful but modest demonstration target for a singlet.
    wfe_target = .25
    spot_target = .018
    condition_scale = 0.

    def __init__(self, samples, *, scenarios=((0., 0.),), guard=.995):
        self.samples = samples
        self.scenarios = scenarios
        self.guard = guard
        self.rules = {name: disk_quadrature(int(name[2:]), 24 if name == 'gq4' else 48)
                      for name in {sample['pupil_rule'] for sample in samples}}
        self.derivative_evaluations = 0
        self.candidate_evaluations = 0
        self.local_norm_specs = {
            'wfe': dict(vector='wave_vector', gain=1./(self.wfe_target*self.guard), offset=1.),
            'spot': dict(vector='spot_vector', gain=1./(self.spot_target*self.guard), offset=1.)}

    @staticmethod
    def surfaces(q, glass=INDICES[1], scenario=(0., 0.)):
        dc, dz = scenario
        return (Surface(64., aperture_radius_mm=4.),
                Surface(65., glass, 1./30.+.005*q[0]+dc,
                        even_coefficients=(0., 1e-5*q[2]), aperture_radius_mm=6.),
                Surface(69., 1., -1./45.+.005*q[1], aperture_radius_mm=6.),
                Surface(BASE_IMAGE+q[3]+dz))

    def model(self, q):
        sf = self.surfaces(q)
        return dict(vertices_mm=[s.vertex_mm for s in sf],
            surface_rows=[dict(curvature_mm_inv=s.curvature_mm_inv, conic=s.conic,
                even_coefficients=list(s.even_coefficients),
                aperture_radius_mm=None if not np.isfinite(s.aperture_radius_mm) else s.aperture_radius_mm)
                for s in sf],
            wavelengths_um=list(WAVES), indices_by_wave=[[1., 1., n, 1., 1.] for n in INDICES],
            primary_wave=1, stop_surface=1, object_at_infinity=False,
            aperture_type='enpd', aperture_value=8., field_type='object_height',
            fields=[[0., float(h)] for h in np.linspace(0., 1., 11)],
            material_names=['', '', 'DEMO_FIXED_INDEX_GLASS', '', ''])

    def slacks(self, row):
        return dict(wfe=1.-row['wfe']/(self.wfe_target*self.guard),
                    spot=1.-row['spot']/(self.spot_target*self.guard),
                    height_positive=1.-row['height']/(.08*self.guard),
                    height_negative=1.+row['height']/(.08*self.guard))

    def evaluate(self, q, *, derivatives=False, geometry=True):
        max_pupil = max(len(value[0]) for value in self.rules.values())
        fields = len(self.samples)
        wave_rows, spot_rows, heights, wave_vectors, spot_vectors = [], [], [], [], []
        wave_jac, spot_jac, height_jac, wave_vjac, spot_vjac = [], [], [], [], []
        origins, raw, splits = [], [], []
        for sample in self.samples:
            pupil, weights = self.rules[sample['pupil_rule']]
            field = float(sample['field_mm'])
            all_pupil = np.vstack((np.zeros((1, 2)), pupil))
            origin = np.array([0., field, 0.])
            directions = np.column_stack(((4.*all_pupil-origin[:2])/64., np.ones(len(all_pupil))))
            origins.extend(np.broadcast_to(origin, directions.shape))
            raw.extend(directions)
            splits.append((len(directions), weights))
        origins, raw = np.asarray(origins), np.asarray(raw)
        for scenario in self.scenarios:
            for wavelength, glass in zip(WAVES, INDICES):
                surfaces = self.surfaces(q, glass, scenario)
                if derivatives:
                    whole = trace_with_jacobian(surfaces, origins, raw, PARAMETERS)
                    if not whole.differentiable.all():
                        return None
                    matrix, dm = paraxial_transfer_with_jacobian(surfaces, 64., PARAMETERS)
                    reference = matrix[0, 1]/matrix[1, 1]
                    dr = (dm[0, 1]*matrix[1, 1]-matrix[0, 1]*dm[1, 1])/matrix[1, 1]**2
                    rays = whole.trace
                else:
                    rays = trace(surfaces, origins, raw)
                    matrix = paraxial_transfer(surfaces, 64.)
                    reference = matrix[0, 1]/matrix[1, 1]
                if not rays.valid.all():
                    return None
                waves, spots, height, vwave, vspot = [], [], [], [], []
                jwaves, jspots, jheight, jvwave, jvspot = [], [], [], [], []
                start = 0
                for sample, (count, weights) in zip(self.samples, splits):
                    def extract(value, part):
                        result = replace(value, points=value.points[part], directions=value.directions[part],
                            opl_mm=value.opl_mm[part], valid=value.valid[part],
                            failed_surface=value.failed_surface[part], reason=value.reason[part])
                        result.surface_points = None
                        return result
                    chief_part, bundle_part = slice(start, start+1), slice(start+1, start+count)
                    chief, bundle = extract(rays, chief_part), extract(rays, bundle_part)
                    if derivatives:
                        def extract_derivative(part):
                            return replace(whole, trace=extract(rays, part),
                                points_jacobian=whole.points_jacobian[part],
                                directions_jacobian=whole.directions_jacobian[part],
                                opl_jacobian=whole.opl_jacobian[part],
                                differentiable=whole.differentiable[part],
                                derivative_reason=whole.derivative_reason[part])
                        bc, cc = extract_derivative(bundle_part), extract_derivative(chief_part)
                        opd, dj = reference_opd_with_jacobian(bc, cc, reference,
                                                               pupil_distance_jacobian=dr)
                        wave = wavefront_rms_with_jacobian(opd, dj, weights, wavelength,
                            bundle.directions[:, :2], tilt_jacobian=bc.directions_jacobian[:, :2])
                        rms, phase = wave.rms_waves, wave.residual_waves
                    else:
                        opd = reference_opd_mm(bundle, chief, reference)
                        rms, phase = wavefront_rms(opd, weights, wavelength, bundle.directions[:, :2])
                    weights = weights/weights.sum()
                    xy = bundle.points[:, :2]
                    centered = xy-np.sum(xy*weights[:, None], axis=0)
                    spot = np.sqrt(np.sum(weights[:, None]*centered**2))
                    # Keep the baseline paraxial magnification approximately;
                    # this gate is checked in every configured perturbation.
                    baseline_height = -1.1530128065*float(sample['field_mm'])
                    height.append(chief.points[0, 1]-baseline_height)
                    waves.append(rms)
                    spots.append(spot)
                    vwave.append(np.pad(phase*np.sqrt(weights), (0, max_pupil-len(weights))))
                    vspot.append(np.pad((centered*np.sqrt(weights[:, None])).ravel(),
                                        (0, 2*(max_pupil-len(weights)))))
                    if derivatives:
                        dxy = bc.points_jacobian[:, :2].copy()
                        dxy -= np.einsum('n,nip->ip', weights, dxy)[None]
                        jwaves.append(wave.rms_jacobian)
                        jspots.append(np.einsum('n,ni,nip->p', weights, centered, dxy)/spot)
                        jheight.append(cc.points_jacobian[0, 1])
                        jvwave.append(np.pad(wave.residual_jacobian*np.sqrt(weights[:, None]),
                                             ((0, max_pupil-len(weights)), (0, 0))))
                        jvspot.append(np.pad((dxy*np.sqrt(weights[:, None, None])).reshape(-1, 4),
                                              ((0, 2*(max_pupil-len(weights))), (0, 0))))
                    start += count
                wave_rows.append(waves); spot_rows.append(spots); heights.append(height)
                wave_vectors.append(vwave); spot_vectors.append(vspot)
                if derivatives:
                    wave_jac.append(jwaves); spot_jac.append(jspots); height_jac.append(jheight)
                    wave_vjac.append(jvwave); spot_vjac.append(jvspot)
        row = dict(wfe=np.array(wave_rows), spot=np.array(spot_rows), height=np.array(heights),
                   wave_vector=np.array(wave_vectors), spot_vector=np.array(spot_vectors))
        if geometry:
            model = self.model(q)
            geometry_report = audit_manufacturing(model, POLICY)
            if not geometry_report['passed']:
                return None
            # Same fixed full-field/full-spectrum envelope for every candidate.
            geometry_report = audit_ray_envelope(OpticalSystem(model), geometry_report,
                                                   disk_quadrature(4, 16)[0], rim_samples=32)
            if not geometry_report['passed']:
                return None
        if derivatives:
            row['jacobians'] = dict(wfe=np.array(wave_jac), spot=np.array(spot_jac), height=np.array(height_jac),
                wave_vector=np.array(wave_vjac), spot_vector=np.array(spot_vjac))
        return row

    def batch(self, qs):
        rows = []
        for q in qs:
            self.candidate_evaluations += 1
            try:
                rows.append(self.evaluate(q))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                rows.append(None)
        return rows

    def analytic_jacobian(self, q, base, keys):
        self.derivative_evaluations += 1
        row = self.evaluate(q, derivatives=True, geometry=False)
        if row is None:
            raise ValueError('Analytic derivative crossed an optical boundary')
        j = row['jacobians']
        blocks = dict(wfe=-j['wfe']/(self.wfe_target*self.guard),
                      spot=-j['spot']/(self.spot_target*self.guard),
                      height_positive=-j['height']/(.08*self.guard),
                      height_negative=j['height']/(.08*self.guard))
        return dict(slacks=np.concatenate([blocks[key].reshape(-1, 4) for key in keys]),
                    vectors={key: j[key] for key in ('wave_vector', 'spot_vector')})


def audit_records(q, *, scenarios=SCENARIOS):
    samples = [dict(field_mm=float(h), pupil_rule=rule) for h in np.linspace(0., 1., 11)
               for rule in ('gq12', 'gq20')]
    problem = Singlet(samples, scenarios=scenarios, guard=1.)
    row = problem.evaluate(q)
    failed_gates = dict(wfe=-1., spot=-1., height_positive=-1., height_negative=-1.)
    if row is None:
        return [dict(sample=s, valid=False, gates=dict(failed_gates)) for s in samples]
    if not all(report['passed'] for report in mechanical_audits(q, scenarios)):
        return [dict(sample=s, valid=False, gates=dict(failed_gates)) for s in samples]
    gates = problem.slacks(row)
    return [dict(sample=s, gates={key: float(np.min(value[:, i])) for key, value in gates.items()})
            for i, s in enumerate(samples)]


def mechanical_audits(q, scenarios=SCENARIOS):
    problem = Singlet([dict(field_mm=0., pupil_rule='gq4')])
    reports = []
    for dc, dz in scenarios:
        model = problem.model(q)
        model['surface_rows'][1]['curvature_mm_inv'] += dc
        model['vertices_mm'][-1] += dz
        geometry = audit_manufacturing(model, POLICY)
        verified = audit_ray_envelope(OpticalSystem(model), geometry,
                                      disk_quadrature(12, 48)[0], rim_samples=96)
        reports.append(verified)
    return reports


def run_one(backend, output, *, phase2=6, subproblem_backend="slsqp"):
    overall_start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    counts, solve_reports = [], []
    def solve_round(q, samples, index):
        problem = Singlet(samples, scenarios=SCENARIOS)
        tick = perf_counter()
        report = problem.solve(output/f'round-{index}', iters=35, start_trust=.5, variant=.2,
            damping=1e-5, use_cones=True, tolerances=1e-9, phase2_iterations=phase2,
            subproblem_backend=subproblem_backend,
            initial_q=q, margin_keys=['wfe', 'spot', 'height_positive', 'height_negative'],
            jacobian_provider=problem.analytic_jacobian if backend == 'analytic' else None,
            strict_validator=lambda model: {'passed': all(min(r['gates'].values()) >= 0.
                and r.get('valid', True) for r in audit_records(np.array(report_q(model))))}, verbose=False)
        counts.append(dict(candidates=problem.candidate_evaluations,
                           analytic_evaluations=problem.derivative_evaluations,
                           wall_seconds=perf_counter()-tick, counters=report['counters']))
        solve_reports.append(report)
        return np.array(report['q']), dict(status=report['status'], counters=report['counters'],
                                           final_acceptance=report['final_acceptance'])
    def report_q(model):
        front, rear = model['surface_rows'][1:3]
        return [(front['curvature_mm_inv']-1./30.)/.005,
                (rear['curvature_mm_inv']+1./45.)/.005,
                front['even_coefficients'][1]/1e-5, model['vertices_mm'][-1]-BASE_IMAGE]
    result = run_constraint_exchange(solve_round, audit_records,
        [dict(field_mm=0., pupil_rule='gq4')], np.zeros(4), max_rounds=4, add_per_round=3)
    result.update(backend=backend, subproblem_backend=subproblem_backend, counts=counts, solve_reports=solve_reports)
    final_problem = Singlet([dict(field_mm=float(h), pupil_rule=rule)
        for h in np.linspace(0., 1., 11) for rule in ('gq12', 'gq20')], scenarios=SCENARIOS, guard=1.)
    final = final_problem.evaluate(np.array(result['q']))
    result['final_optics'] = None if final is None else dict(
        maximum_wfe_waves=float(final['wfe'].max()), maximum_spot_rms_mm=float(final['spot'].max()),
        maximum_height_error_mm=float(abs(final['height']).max()))
    result['scenario_manufacturing'] = mechanical_audits(np.array(result['q']))
    result['passed'] = bool(result['passed'] and final is not None
        and all(np.min(value) >= 0. for value in final_problem.slacks(final).values())
        and all(report['passed'] for report in result['scenario_manufacturing']))
    result['exchange_seconds'] = result['seconds']
    result['seconds'] = perf_counter()-overall_start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--solver', choices=['slsqp', 'clarabel', 'auto'], default='slsqp')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    initial = Singlet([dict(field_mm=float(h), pupil_rule='gq12') for h in (0., .5, 1.)], scenarios=SCENARIOS)
    base = initial.evaluate(np.zeros(4))
    dense_initial = Singlet([dict(field_mm=float(h), pupil_rule=rule)
        for h in np.linspace(0., 1., 11) for rule in ('gq12', 'gq20')], scenarios=SCENARIOS, guard=1.)
    dense_base = dense_initial.evaluate(np.zeros(4))
    report = dict(case='public_synthetic_singlet', uses_zemax=False, uses_cuda=False,
        environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                         platform=platform.platform(), processor=platform.processor()),
        parameters=['front_curvature_knob', 'rear_curvature_knob', 'front_A4_knob', 'focus_mm'],
        wavelengths_um=WAVES, index_after_front=INDICES, deterministic_scenarios=SCENARIOS,
        acceptance=dict(maximum_wfe_waves=Singlet.wfe_target, maximum_spot_rms_mm=Singlet.spot_target,
                        maximum_chief_height_error_mm=.08),
        initial=dict(maximum_wfe_waves=float(dense_base['wfe'].max()),
                     maximum_spot_rms_mm=float(dense_base['spot'].max()),
                     maximum_height_error_mm=float(abs(dense_base['height']).max())), runs=[])
    # Independent central differences validate the adapter's *complete slack*
    # derivative, rather than testing endpoint tangents alone.
    keys = tuple(initial.slacks(base))
    analytic = initial.analytic_jacobian(np.zeros(4), base, keys)['slacks']
    epsilon = 1e-4
    columns = []
    for i in range(4):
        plus, minus = np.eye(4)[i]*epsilon, -np.eye(4)[i]*epsilon
        a, b = initial.slacks(initial.evaluate(plus, geometry=False)), initial.slacks(initial.evaluate(minus, geometry=False))
        columns.append(np.concatenate([(a[k]-b[k]).ravel() for k in keys])/(2*epsilon))
    report['maximum_slack_jacobian_error'] = float(abs(analytic-np.column_stack(columns)).max())
    report['gradient_validation'] = dict(method='central_difference', dimensionless_step=epsilon,
        maximum_absolute_normalized_slack_error=report['maximum_slack_jacobian_error'],
        acceptance_limit=1e-5, passed=report['maximum_slack_jacobian_error'] <= 1e-5)
    if not report['gradient_validation']['passed']:
        raise SystemExit('Complete optical slack derivative disagrees with central differences')
    for repeat in range(args.repeats):
        order = ('finite_difference', 'analytic') if repeat % 2 == 0 else ('analytic', 'finite_difference')
        for backend in order:
            value = run_one(backend, args.out/f'{backend}-{repeat}', subproblem_backend=args.solver)
            report['runs'].append(value)
            print(json.dumps(dict(repeat=repeat, backend=backend, passed=value['passed'],
                seconds=value['seconds'], q=value['q'], counts=value['counts'])), flush=True)
    report['all_passed'] = all(value['passed'] for value in report['runs'])
    report['median_wall_seconds'] = {name: float(np.median([r['seconds'] for r in report['runs'] if r['backend'] == name]))
                                    for name in ('finite_difference', 'analytic')}
    (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('initial', 'maximum_slack_jacobian_error',
                     'all_passed', 'median_wall_seconds')}, indent=2))
    if not report['all_passed']:
        raise SystemExit('At least one independent dense audit failed; inspect summary.json')


if __name__ == '__main__':
    main()
