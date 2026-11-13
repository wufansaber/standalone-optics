"""Regressions for scoped process limits and fresh optical/physical envelopes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.ai_zemax.standalone import disk_quadrature
from scripts.ai_zemax.standalone_cli import main as analyze
from scripts.ai_zemax.standalone_gpu_screen import CudaCandidateScreen
from scripts.ai_zemax.standalone_gpu_trace import require_cupy
from scripts.ai_zemax.standalone_manufacturing import audit_manufacturing, audit_ray_envelope
from scripts.ai_zemax.standalone_parallel_screen import screen_candidates
from scripts.ai_zemax.standalone_system import OpticalSystem
from tests.test_manufacturing import singlet, policy


def pair():
    model = dict(vertices_mm=[10., 12., 12.2, 14.2, 100.],
                 indices_by_wave=[[1., 1.5, 1., 1.5, 1., 1.]],
                 surface_rows=[dict(aperture_radius_mm=1., mechanical_radius_mm=1.2)
                               for _ in range(4)]+[{}])
    limits = policy()
    limits['lenses'].append(dict(front_surface=3, rear_surface=4))
    return model, limits


def optical_escape():
    model = singlet()
    for row in model['surface_rows'][:2]:
        row['aperture_radius_mm'] = 2.
    model.update(fields=[[0., 0.], [0., .5]], field_type='object_height',
                 aperture_type='obna', aperture_value=.1, wavelengths_um=[.55, .65],
                 indices_by_wave=[[1., 1.5, 1., 1.], [1., 1.49, 1., 1.]],
                 stop_surface=1, primary_wave=0, object_at_infinity=False,
                 object_space_telecentric=True)
    limits = policy()
    limits['surfaces'] = {str(i): dict(clear_radius_mm=1., mechanical_radius_mm=1.2)
                         for i in (1, 2)}
    return model, limits


class ManufacturingAcceptanceTests(unittest.TestCase):
    def codes(self, report):
        json.dumps(report, allow_nan=False)
        return {row['code'] for row in report['violations']}

    def test_all_six_previously_ignored_lens_fields_are_enforced(self):
        for key, value, expected in (
                ('maximum_surface_slope', .001, 'surface_slope_above_limit'),
                ('minimum_clear_to_mechanical_radial_margin_mm', .5, 'radial_margin_below_limit'),
                ('minimum_air_gap_mm', .5, 'air_gap_below_limit'),
                ('air_gap_budget_mm', .2, 'air_gap_below_limit'),
                ('minimum_rim_gap_mm', .5, 'rim_gap_below_limit'),
                ('rim_gap_budget_mm', .2, 'rim_gap_below_limit')):
            with self.subTest(key=key):
                model, limits = pair() if 'gap' in key else (singlet(), policy())
                limits['lenses'][0][key] = value
                report = audit_manufacturing(model, limits)
                self.assertFalse(report['eligible'])
                self.assertIn(expected, self.codes(report))

    def test_gap_override_binds_both_sides_and_keeps_minimum_budget_pairs(self):
        model, limits = pair()
        # A lens applies to the air interface on either side, with the tighter
        # complete requirement. Do not combine one lens's minimum with another's budget.
        limits['lenses'][0].update(minimum_air_gap_mm=.19, air_gap_budget_mm=0.)
        limits['lenses'][1].update(minimum_air_gap_mm=.01, air_gap_budget_mm=.18)
        report = audit_manufacturing(model, limits)
        self.assertTrue(report['passed'])
        limits['lenses'][1]['air_gap_budget_mm'] = .20
        report = audit_manufacturing(model, limits)
        self.assertIn('air_gap_below_limit', self.codes(report))
        self.assertEqual(report['gaps'][0]['air_requirement_lens'], 2)
        self.assertEqual(report['gaps'][0]['effective_air_gap_budget_mm'], .20)

    def test_shared_face_satisfies_both_lens_process_limits(self):
        model = dict(vertices_mm=[10., 12., 14., 100.],
                     indices_by_wave=[[1., 1.5, 1.6, 1., 1.]],
                     surface_rows=[dict(aperture_radius_mm=1., mechanical_radius_mm=1.2),
                                   dict(curvature_mm_inv=.02, aperture_radius_mm=1., mechanical_radius_mm=1.2),
                                   dict(aperture_radius_mm=1., mechanical_radius_mm=1.2), {}])
        limits = policy()
        limits['lenses'] = [dict(front_surface=1, rear_surface=2, maximum_surface_slope=.1),
                            dict(front_surface=2, rear_surface=3, maximum_surface_slope=.001,
                                 minimum_clear_to_mechanical_radial_margin_mm=.3)]
        report = audit_manufacturing(model, limits)
        self.assertEqual(report['gaps'][0]['kind'], 'shared_cemented_interface')
        shared = next(row for row in report['surfaces'] if row['surface'] == 2)
        self.assertEqual(shared['effective_maximum_surface_slope'], .001)
        self.assertEqual(shared['effective_minimum_clear_to_mechanical_radial_margin_mm'], .3)
        self.assertIn('surface_slope_above_limit', self.codes(report))
        self.assertIn('radial_margin_below_limit', self.codes(report))

    def test_lens_override_can_relax_default_without_relaxing_other_lenses(self):
        limits = policy()
        limits['defaults']['maximum_surface_slope'] = .001
        limits['lenses'][0]['maximum_surface_slope'] = .1
        self.assertTrue(audit_manufacturing(singlet(), limits)['passed'])

    def test_fresh_envelope_rejects_ca_and_physical_escape_without_clipping(self):
        model, limits = optical_escape()
        original = deepcopy(model)
        geometry = audit_manufacturing(model, limits)
        self.assertTrue(geometry['passed'])
        system = OpticalSystem(model)
        origin, slopes = system.launch_field([0., .5])
        ray = system._shoot(0, [[0., 1.]], origin, slopes, record_path=True)
        self.assertTrue(ray.valid[0])
        self.assertGreater(np.linalg.norm(ray.surface_points[0, 0, :2]), 1.5)
        pupil, _ = disk_quadrature(2, 8)
        report = audit_ray_envelope(system, geometry, pupil)
        self.assertIn('ray_outside_clear_aperture', self.codes(report))
        self.assertIn('ray_outside_mechanical_radius', self.codes(report))
        self.assertFalse(report['passed'])
        self.assertTrue(geometry['passed'])  # no mutation of the nominal audit
        self.assertEqual(model, original)
        self.assertEqual(report['ray_envelope']['field_indices'], [0, 1])
        self.assertEqual(report['ray_envelope']['wave_indices'], [0, 1])

    def test_exact_rim_and_unselected_fields_cannot_evade_ranking_gate(self):
        model, limits = optical_escape()
        # Interior GQ1 nodes fit on axis; exact rim exceeds the tighter CA.
        limits['surfaces'] = {str(i): dict(clear_radius_mm=.9, mechanical_radius_mm=1.2)
                             for i in (1, 2)}
        result = screen_candidates(model, [dict(id='escape')], radial=1, azimuthal=4,
                                   field_indices=[0], wave_indices=[0], manufacturing_policy=limits)
        self.assertEqual(result['ranked'], [])
        row = result['results'][0]
        self.assertEqual(row['reasons'][0]['code'], 'manufacturing_ray_envelope_failed')
        self.assertEqual(row['manufacturing']['ray_envelope']['field_indices'], [0, 1])
        self.assertEqual(row['manufacturing']['ray_envelope']['wave_indices'], [0, 1])
        axis = deepcopy(model)
        axis['fields'] = [[0., 0.]]
        report = audit_ray_envelope(OpticalSystem(axis), audit_manufacturing(axis, limits),
                                    disk_quadrature(1, 4)[0])
        self.assertIn('ray_outside_clear_aperture', self.codes(report))

    def test_cli_runs_fresh_gate_but_manufacturing_only_remains_nominal(self):
        model, limits = optical_escape()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, settings, output = root/'lens.json', root/'policy.json', root/'report.json'
            source.write_text(json.dumps(model), encoding='utf-8')
            settings.write_text(json.dumps(limits), encoding='utf-8')
            args = ['--model', str(source), '--manufacturing-policy', str(settings), '--out', str(output)]
            self.assertEqual(analyze(args), 2)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'manufacturing_rejected')
            self.assertIn('ray_outside_mechanical_radius', self.codes(report['manufacturing']))
            self.assertNotIn('rows', report)
            self.assertEqual(analyze([*args, '--manufacturing-only']), 0)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertTrue(report['manufacturing']['passed'])
            self.assertNotIn('ray_envelope', report['manufacturing'])

    def test_cuda_and_cpu_use_same_fresh_envelope_gate(self):
        try:
            require_cupy()
        except RuntimeError as error:
            self.skipTest(str(error))
        model, limits = optical_escape()
        candidates = [dict(id='escape')]
        settings = dict(radial=2, azimuthal=8, manufacturing_policy=limits)
        cpu = screen_candidates(model, candidates, **settings)
        with CudaCandidateScreen(model, **settings) as cuda:
            gpu = cuda.screen(candidates)
        self.assertEqual(cpu['ranked'], gpu['ranked'])
        self.assertEqual(cpu['results'][0]['manufacturing'], gpu['results'][0]['manufacturing'])
        self.assertEqual(cpu['results'][0]['reasons'], gpu['results'][0]['reasons'])


if __name__ == '__main__':
    unittest.main()
