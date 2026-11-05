"""Exercise the optional CUDA path on machines with an available device."""
import unittest

import numpy as np

from scripts.ai_zemax.standalone import Surface, trace
from scripts.ai_zemax.standalone_gpu_screen import CudaCandidateScreen
from scripts.ai_zemax.standalone_gpu_trace import require_cupy, trace_gpu
from scripts.ai_zemax.standalone_optimization_cli import verify_cuda_scores
from scripts.ai_zemax.standalone_parallel_screen import screen_candidates


def setUpModule():
    try:
        require_cupy()
    except RuntimeError as error:
        raise unittest.SkipTest(str(error)) from error


class CudaBackendTests(unittest.TestCase):
    def test_asphere_and_failed_ray_match_cpu(self):
        surfaces = [Surface(10., 1.52, .024, -.6, (0., 1e-6), aperture_radius_mm=1.),
                    Surface(13., 1., -.018), Surface(80.)]
        directions = np.array([[0., 0., 1.], [.05, 0., 1.], [.2, 0., 1.]])
        reference = trace(surfaces, [0., 0., 0.], directions, record_path=True)
        actual = trace_gpu(surfaces, [0., 0., 0.], directions, record_path=True)
        for key in ('valid', 'failed_surface', 'reason', 'negative_segment_count', 'shortest_segment_surface'):
            np.testing.assert_array_equal(getattr(actual,key),getattr(reference,key))
        self.assertEqual(reference.reason[-1],'aperture')
        for key in ('points', 'directions', 'surface_points', 'opl_mm', 'shortest_segment_mm'):
            np.testing.assert_allclose(getattr(actual,key),getattr(reference,key),atol=1e-10,rtol=0.)

    def test_score_ranking_and_aperture_replay(self):
        model = {
            'vertices_mm': [100., 325.],
            'surface_rows': [{'curvature_mm_inv': 1./30.}, {}],
            'wavelengths_um': [.486, .55, .656],
            'indices_by_wave': [[1., index, index] for index in (1.51, 1.5, 1.49)],
            'primary_wave': 1, 'stop_surface': 1, 'object_at_infinity': False,
            'aperture_type': 'obna', 'aperture_value': .015,
            'field_type': 'object_height', 'fields': [[0., 0.], [.5, 1.]],
        }
        clipped = dict(model,surface_rows=[dict(model['surface_rows'][0],aperture_radius_mm=.1),{}])
        candidates = [{'id':'baseline'}, {'id':'bent','curvature_deltas':[[1,.001]]},
                      {'id':'clipped','model':clipped}]
        settings = dict(radial=4,azimuthal=16)
        reference = screen_candidates(model,candidates,**settings)
        with CudaCandidateScreen(model,**settings) as cuda:
            measured = cuda.screen(candidates)
            replay = cuda.screen(candidates[:1])
        check = verify_cuda_scores(reference,measured)
        self.assertTrue(check['serial_cuda_rank_identical'])
        self.assertEqual(replay['ranked'],['baseline'])
        self.assertIn('aperture',measured['results'][-1]['reasons'][0]['detail'])


if __name__ == '__main__':
    unittest.main()
