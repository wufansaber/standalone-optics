"""Coupled curvature/gap seeds preserve shared conjugates and source geometry."""
from copy import deepcopy
import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone_coupled_group_search import search_coupled_group_gaps
from tests.test_standalone_group_search import full_matrices, model_with_spacer, physical_gaps


class CoupledGroupSearchTests(unittest.TestCase):
    def setUp(self):
        self.model = model_with_spacer()
        self.cuts = [2, 3, 5]
        self.gaps = physical_gaps(self.model, self.cuts)
        self.bounds = np.column_stack((self.gaps-2., self.gaps+2.))
        self.mode = {'name': 'front_bend', 'min': -1., 'max': 1.,
                     'curvature_per_unit_mm_inv': [[1, 3e-5], [2, -2e-5]]}

    def search(self, **kwargs):
        settings = dict(split_after_surfaces=self.cuts, gap_bounds_mm=self.bounds,
                        curvature_modes=[self.mode], mode_samples=5,
                        gap_samples=12, layouts_per_mode=3, count=9)
        return search_coupled_group_gaps(self.model, **dict(settings, **kwargs))

    def test_balanced_layouts_keep_shared_abcd_and_original_model(self):
        original = deepcopy(self.model)
        report = self.search()
        self.assertEqual(report, self.search())
        self.assertEqual(self.model, original)
        self.assertEqual(report['candidates'][0]['id'], 'baseline')
        self.assertEqual(report['candidates'][0]['model'], self.model)
        self.assertEqual(len(report['candidates']), 9)
        self.assertGreater(report['generated_candidate_count'], len(report['candidates']))
        self.assertTrue(any(row['curvature_deltas'] for row in report['candidates']))
        self.assertEqual(len(report['mode_reports']), 5)
        json.dumps(report, allow_nan=False)
        target = full_matrices(self.model)[1, 0]
        for row in report['candidates']:
            model = row['model']
            np.testing.assert_allclose(row['matrix_by_wave'], full_matrices(model), atol=1e-10)
            np.testing.assert_allclose(row['matrix_by_wave'][1][0], target, atol=1e-9)
            np.testing.assert_allclose(row['gap_mm'], physical_gaps(model, self.cuts), atol=1e-10)
            np.testing.assert_allclose(model['vertices_mm'][-1], self.model['vertices_mm'][-1], atol=1e-10)
            self.assertEqual(model['indices_by_wave'], self.model['indices_by_wave'])
            self.assertEqual(model['stop_surface'], self.model['stop_surface'])
            for before, after in zip(self.model['surface_rows'], model['surface_rows']):
                self.assertEqual(dict(before, curvature_mm_inv=after['curvature_mm_inv']), after)
            if row['id'] != 'baseline':
                self.assertNotIn('source_sha256', model)
                self.assertEqual(model['coupled_group_search']['parent_source_sha256'], 'example')

    def test_custom_target_is_shared_and_does_not_mislabel_baseline(self):
        target_model = deepcopy(self.model)
        target_model['vertices_mm'][-1] += .1
        matrix = full_matrices(target_model)[1, 0]
        report = self.search(track_mm=target_model['vertices_mm'][-1], magnification=matrix[0],
                             defocus_b_mm=matrix[1], mode_samples=3)
        self.assertTrue(report['candidates'])
        self.assertNotIn('baseline', [row['id'] for row in report['candidates']])
        for row in report['candidates']:
            np.testing.assert_allclose(row['matrix_by_wave'][1][0], matrix, atol=1e-9)
            self.assertAlmostEqual(row['model']['vertices_mm'][-1], target_model['vertices_mm'][-1])

    def test_duplicate_modes_and_invalid_bounds(self):
        alias = dict(self.mode, name='same_coefficients')
        report = self.search(curvature_modes=[self.mode, alias], mode_samples=5, count=5)
        self.assertEqual(report['rejections']['duplicate_curvatures'], 2)
        self.assertEqual(len({row['id'] for row in report['candidates']}), len(report['candidates']))
        with self.assertRaisesRegex(ValueError, 'contain zero'):
            self.search(curvature_modes=[dict(self.mode, min=.1)])
        with self.assertRaisesRegex(ValueError, 'before the image'):
            self.search(curvature_modes=[dict(self.mode, curvature_per_unit_mm_inv=[
                [len(self.model['surface_rows']), 1.]])])
        with self.assertRaisesRegex(ValueError, 'search budget'):
            self.search(mode_samples=256, gap_samples=4096)


if __name__ == '__main__':
    unittest.main()
