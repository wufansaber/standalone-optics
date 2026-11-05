"""Screening tests for spawn equivalence and candidate-level rejection."""
import json
import unittest

from scripts.ai_zemax.standalone_parallel_screen import CandidateScreen, screen_candidates


def model():
    return {
        'vertices_mm': [100., 325.],
        'surface_rows': [{'curvature_mm_inv': 1./30.}, {}],
        'wavelengths_um': [.486, .55, .656],
        'indices_by_wave': [[1., index, index] for index in (1.51, 1.5, 1.49)],
        'primary_wave': 1,
        'stop_surface': 1,
        'object_at_infinity': False,
        'aperture_type': 'obna',
        'aperture_value': .015,
        'field_type': 'object_height',
        'fields': [[0., 0.], [.5, 1.]],
    }


class ParallelScreenTests(unittest.TestCase):
    def test_spawn_matches_serial_and_reuses_pool(self):
        baseline = model()
        candidates = [
            {'id': 'first_baseline'},
            {'id': 'full_baseline', 'model': model()},
            {'id': 'curved', 'curvature_deltas': [[1, .001]]},
            {'id': 'invalid', 'curvature_deltas': [[2, .001]]},
        ]
        options = dict(field_indices=[0, 1], wave_indices=[1], radial=4, azimuthal=12)
        expected = screen_candidates(baseline, candidates, **options)
        with CandidateScreen(baseline, workers=2, **options) as pool:
            actual = pool.screen(candidates)
            replay = pool.screen(candidates[:1])
        self.assertEqual(actual['results'], expected['results'])
        self.assertEqual(actual['ranked'], expected['ranked'])
        self.assertEqual(replay['ranked'], ['first_baseline'])
        self.assertLess(actual['ranked'].index('first_baseline'), actual['ranked'].index('full_baseline'))
        self.assertEqual(actual['results'][-1]['reasons'][0]['code'], 'invalid_model')
        self.assertEqual(baseline['surface_rows'][0]['curvature_mm_inv'], 1./30.)
        json.dumps(actual, allow_nan=False)

    def test_candidate_rejections_are_not_ranks(self):
        baseline = model()
        clipped = model()
        clipped['surface_rows'][0]['aperture_radius_mm'] = .1
        report = screen_candidates(None, [{'id': 'limit', 'model': baseline},
                                         {'id': 'clipped', 'model': clipped}],
                                   spot_limit_um=.00001)
        self.assertEqual(report['status'], 'no_viable_candidates')
        self.assertEqual(report['ranked'], [])
        self.assertIsNone(report['results'][0]['rank'])
        self.assertEqual(report['results'][0]['reasons'][0]['code'], 'screen_limit_exceeded')
        self.assertEqual(report['results'][1]['reasons'][0]['code'], 'trace_failed')
        self.assertIn('aperture', report['results'][1]['reasons'][0]['detail'])


if __name__ == '__main__':
    unittest.main()
