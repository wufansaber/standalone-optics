"""Regression checks for reconstruction and constrained thick-group gap searches."""
from copy import deepcopy
import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone import paraxial_transfer
from scripts.ai_zemax.standalone_group_search import search_group_gaps
from scripts.ai_zemax.standalone_seed_realize import realize_seed
from scripts.ai_zemax.standalone_system import OpticalSystem


def model_with_spacer():
    seed = {'gaps_mm': [50., 55., 60., 35.],
            'powers_mm_inv': [.024, -.01, .024],
            'matrix': [[-1.3, 0.], [.01, -1/1.3]],
            'stop_group': 2}
    model = realize_seed(seed, refractive_indices=[[1.54, 1.55, 1.53], [1.52, 1.53, 1.51]],
                         wavelengths_um=[.486, .5876], primary_wave=1)
    vertices = model['vertices_mm']
    # A plane aperture in the air interval before the second singlet.
    vertices.insert(2, (vertices[1]+vertices[2])/2.)
    model['surface_rows'].insert(2, dict(curvature_mm_inv=0., aperture_radius_mm=8.))
    for row in model['indices_by_wave']:
        row.insert(3, 1.)
    model['stop_surface'] = 3
    model['source_sha256'] = 'example'
    return model


def full_matrices(model):
    system = OpticalSystem(model)
    return np.asarray([paraxial_transfer(surfaces, 0., indices[0])
                       for surfaces, indices in zip(system.surfaces, system.indices)])


def physical_gaps(model, cuts):
    vertices = model['vertices_mm']
    return np.array([vertices[0], *(vertices[i]-vertices[i-1] for i in cuts), vertices[-1]-vertices[-2]])


class GroupSearchTests(unittest.TestCase):
    def test_preserves_real_groups_and_matches_direct_transfer(self):
        model = model_with_spacer()
        baseline = deepcopy(model)
        cuts = [2, 3, 5]
        gaps = physical_gaps(model, cuts)
        settings = dict(split_after_surfaces=cuts, gap_bounds_mm=np.column_stack((gaps-2, gaps+2)),
                        samples=40, count=7, random_seed=9)
        report = search_group_gaps(model, **settings)
        self.assertEqual(model, baseline)
        self.assertEqual(report, search_group_gaps(model, **settings))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report['group_surface_ranges'], [[1, 2], [3, 3], [4, 5], [6, 7]])
        self.assertLess(report['baseline']['max_abcd_entry_delta'], 1e-11)
        self.assertEqual(len(report['candidates']), 7)
        self.assertEqual(report['candidates'][0]['id'], 'baseline')
        self.assertEqual(report['candidates'][0]['model'], model)
        for row in report['candidates']:
            shifted = row['model']
            self.assertEqual(shifted['surface_rows'], model['surface_rows'])
            self.assertEqual(shifted['indices_by_wave'], model['indices_by_wave'])
            self.assertEqual(shifted['stop_surface'], model['stop_surface'])
            np.testing.assert_allclose(row['gap_mm'], physical_gaps(shifted, cuts), atol=1e-11)
            np.testing.assert_allclose(row['matrix_by_wave'], full_matrices(shifted), atol=1e-11)
            np.testing.assert_allclose(row['matrix_by_wave'][1][0], full_matrices(model)[1, 0], atol=1e-10)
            self.assertAlmostEqual(shifted['vertices_mm'][-1], model['vertices_mm'][-1])

    def test_fixed_outer_gap_modes_recover_a_known_target(self):
        model = model_with_spacer()
        cuts = [2, 5]
        gaps = physical_gaps(model, cuts)
        target_model = deepcopy(model)
        target_model['vertices_mm'][2:5] = [z+.5 for z in model['vertices_mm'][2:5]]
        matrix = full_matrices(target_model)[1]
        for outer_fixed in ([0], [-1], [0, -1]):
            with self.subTest(outer_fixed=outer_fixed):
                bounds = np.column_stack((gaps-1, gaps+1))
                bounds[outer_fixed] = gaps[outer_fixed, None]
                result = search_group_gaps(model, split_after_surfaces=cuts, gap_bounds_mm=bounds,
                                           magnification=matrix[0, 0], defocus_b_mm=matrix[0, 1],
                                           samples=32, count=16)
                self.assertTrue(result['candidates'])
                expected = physical_gaps(target_model, cuts)
                self.assertTrue(any(np.max(abs(row['gap_mm']-expected)) < 1e-8
                                    for row in result['candidates']))

    def test_rejects_glass_split_and_infeasible_track(self):
        model = model_with_spacer()
        gaps = physical_gaps(model, [2, 5])
        bounds = np.column_stack((gaps, gaps))
        with self.assertRaisesRegex(ValueError, 'must be in air'):
            search_group_gaps(model, split_after_surfaces=[1, 5], gap_bounds_mm=bounds)
        report = search_group_gaps(model, split_after_surfaces=[2, 5], gap_bounds_mm=bounds,
                                   track_mm=100.)
        self.assertEqual(report['status'], 'no_feasible_candidates')


if __name__ == '__main__':
    unittest.main()
