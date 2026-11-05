"""Regression coverage for selecting MTF by the optical regime."""
import unittest
from unittest.mock import patch

from scripts.ai_zemax import standalone_mtf_workflow as workflow


class MtfWorkflowTests(unittest.TestCase):
    def test_large_resolved_opd_keeps_diffraction(self):
        fft = {'method': 'scalar_fft_normalized_pupil', 'warnings': ['large_opd_span_scalar_fft']}
        with patch.object(workflow, 'diffraction_mtf', return_value=fft),              patch.object(workflow, 'geometric_mtf') as geometric:
            self.assertIs(workflow.calculate_mtf(None, [0, 0]), fft)
            geometric.assert_not_called()

    def test_severe_blur_uses_geometric_and_records_fft_warning(self):
        warning = 'large_transverse_ray_aberration_scalar_fft'
        fft = {'warnings': [warning]}
        geometric_result = {'method': 'geometric_ray_intercept'}
        with patch.object(workflow, 'diffraction_mtf', return_value=fft),              patch.object(workflow, 'geometric_mtf', return_value=geometric_result) as geometric:
            result = workflow.calculate_mtf(None, [0, 0], maximum_frequency_lpmm=.02)
        self.assertEqual(result['selection']['fft_diagnostics'], [warning])
        self.assertEqual(geometric.call_args.kwargs['frequency_lpmm'][-1], .02)

    def test_unresolved_wavefront_uses_geometric_unless_fft_requested(self):
        warnings = ['large_opd_span_scalar_fft', 'phase_grid_may_alias_increase_sampling']
        fft = {'warnings': warnings}
        geometric_result = {'method': 'geometric_ray_intercept'}
        with patch.object(workflow, 'diffraction_mtf', return_value=fft),              patch.object(workflow, 'geometric_mtf', return_value=geometric_result) as geometric:
            self.assertIs(workflow.calculate_mtf(None, [0, 0], method='fft'), fft)
            geometric.assert_not_called()
            self.assertIs(workflow.calculate_mtf(None, [0, 0]), geometric_result)


if __name__ == '__main__':
    unittest.main()
