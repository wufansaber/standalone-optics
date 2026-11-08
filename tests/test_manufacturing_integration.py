"""Manufacturing gates run before scoring through the public CPU/CUDA APIs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.ai_zemax.standalone_cli import main as analyze
from scripts.ai_zemax.standalone_gpu_screen import CudaCandidateScreen
from scripts.ai_zemax.standalone_model import load_zmx
from scripts.ai_zemax.standalone_parallel_screen import screen_candidates
from scripts.ai_zemax.standalone_optimization_cli import evaluate_batch


def singlet():
    return {
        'vertices_mm': [100., 104., 325.],
        'surface_rows': [
            {'curvature_mm_inv': .03, 'aperture_radius_mm': 5., 'mechanical_radius_mm': 6.},
            {'curvature_mm_inv': -.03, 'aperture_radius_mm': 5., 'mechanical_radius_mm': 6.},
            {},
        ],
        'wavelengths_um': [.55],
        'indices_by_wave': [[1., 1.5, 1., 1.]],
        'primary_wave': 0, 'stop_surface': 1, 'object_at_infinity': False,
        'aperture_type': 'obna', 'aperture_value': .015,
        'field_type': 'object_height', 'fields': [[0., 0.]],
    }


def singlet_policy():
    return {
        'schema_version': 1,
        'defaults': {'minimum_center_thickness_mm': 1.,
                     'minimum_radial_thickness_mm': .5,
                     'minimum_clear_to_mechanical_radial_margin_mm': .1},
        'lenses': [{'front_surface': 1, 'rear_surface': 2}],
    }


class ManufacturingIntegrationTests(unittest.TestCase):
    def test_default_screen_is_exploratory_and_explicit_policy_is_required(self):
        model = singlet()
        baseline = deepcopy(model)
        result = screen_candidates(model, [{'id': 'baseline'}], radial=2, azimuthal=8)
        self.assertEqual(result['ranked'], ['baseline'])
        row = result['results'][0]
        self.assertFalse(row['manufacturing']['passed'])
        self.assertTrue(row['manufacturing']['eligible'])
        self.assertEqual(row['manufacturing']['status'], 'incomplete')
        self.assertEqual(model, baseline)

        result = screen_candidates(model, [{'id': 'baseline'}], manufacturing_policy={})
        self.assertEqual(result['ranked'], [])
        self.assertEqual(result['results'][0]['reasons'][0]['code'], 'manufacturing_incomplete')
        self.assertEqual(result['results'][0]['groups'], [])
        json.dumps(result, allow_nan=False)

    def test_explicit_policy_is_preserved_through_dense_finalist_review(self):
        options = SimpleNamespace(radial=2, azimuthal=8, workers=1, rounds=1,
                                  backend='cpu', cuda_device=None, dense_top=1, verify_serial=True)
        policy = singlet_policy()
        coarse, dense, _, _ = evaluate_batch(singlet(), [{'id': 'baseline'}], options,
                                             {'manufacturing_policy': policy})
        self.assertEqual(coarse['ranked'], ['baseline'])
        self.assertEqual(dense['ranked'], ['baseline'])
        self.assertEqual(coarse['settings']['manufacturing_policy'], policy)
        self.assertEqual(dense['settings']['manufacturing_policy'], policy)
        self.assertTrue(coarse['results'][0]['manufacturing']['passed'])
        self.assertTrue(dense['results'][0]['manufacturing']['passed'])

    def test_cpu_and_cuda_reject_impossible_solid_before_ray_scoring(self):
        model = singlet()
        model['surface_rows'][0]['curvature_mm_inv'] = .5
        result = screen_candidates(model, [{'id': 'impossible'}])
        row = result['results'][0]
        self.assertEqual(result['ranked'], [])
        self.assertEqual(row['reasons'][0]['code'], 'manufacturing_failed')
        self.assertEqual(row['groups'], [])

        # Preflight occurs before the CUDA backend schedules a device job.
        cuda = CudaCandidateScreen(model)
        gpu_row = cuda._score_chunk([{'id': 'impossible'}], 0)[0]
        self.assertEqual(gpu_row['manufacturing'], row['manufacturing'])
        self.assertEqual(gpu_row['reasons'], row['reasons'])
        self.assertEqual(gpu_row['groups'], [])

    def test_analysis_audit_is_explicit_and_writes_failure_without_optical_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = singlet()
            model['surface_rows'][0]['curvature_mm_inv'] = .5
            path, output = root/'lens.json', root/'audit.json'
            path.write_text(json.dumps(model), encoding='utf-8')
            self.assertEqual(analyze(['--model', str(path), '--manufacturing-only', '--out', str(output)]), 2)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertFalse(report['manufacturing']['eligible'])
            self.assertNotIn('rows', report)
            path.write_text(json.dumps(singlet()), encoding='utf-8')
            policy = root/'policy.json'
            policy.write_text(json.dumps(singlet_policy()), encoding='utf-8')
            self.assertEqual(analyze(['--model', str(path), '--manufacturing-policy', str(policy),
                                      '--out', str(output)]), 0)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertTrue(report['manufacturing']['passed'])
            self.assertEqual(report['manufacturing_policy_file_sha256'],
                             hashlib.sha256(policy.read_bytes()).hexdigest())
            self.assertEqual(report['rows'][0]['status'], 'calculated')

    def test_zmx_saved_radii_do_not_replace_explicit_ray_aperture(self):
        source = """MODE SEQ
UNIT MM
ROPD 2
FTYP 1 0 2 1 0
XFLN 0 0
YFLN 0 0.5
WAVM 1 0.55 1
ENPD 5
SURF 0
  TYPE STANDARD
  DISZ 100
SURF 1
  STOP
  TYPE STANDARD
  DISZ 100
  DIAM 5 0
  MEMA 6 1
  CLAP 0 4
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'radii.zmx'
            path.write_text(source, encoding='utf-8')
            model = load_zmx(path, Path(directory))
        row = model['surface_rows'][0]
        self.assertEqual(row['aperture_radius_mm'], 4.)
        self.assertEqual(row['clear_radius_mm'], 5.)
        self.assertEqual(row['mechanical_radius_mm'], 6.)
        self.assertEqual(row['clear_radius_source'], 'zmx_saved_automatic')
        self.assertEqual(row['mechanical_radius_source'], 'zmx_saved_declared')
        self.assertEqual(row['clear_radius_mode'], '0')
        self.assertEqual(row['mechanical_radius_mode'], '1')


if __name__ == '__main__':
    unittest.main()
