"""Analytical formula fixtures; no proprietary vendor catalog is bundled."""

from fractions import Fraction
import math
from pathlib import Path
import tempfile
import unittest

from scripts.ai_zemax.standalone_model import _Glass, _read_catalog


class GlassDispersionTests(unittest.TestCase):
    def test_herzberger_independent_rational_fixture(self):
        # At lambda=1/2: lambda^2=1/4 and L=500/111 exactly.
        # The rational expected value is independent of binary float evaluation.
        glass = _Glass(3, (1.5, .01, .0001, -.001, .00001, -.00000001), (.3, 2.), 20.)
        expected = (Fraction(3, 2)+Fraction(5, 111)+Fraction(25, 12321)
                    -Fraction(1, 4000)+Fraction(1, 1600000)-Fraction(1, 6400000000))
        self.assertAlmostEqual(glass.index(.5), float(expected), places=14)
        # At lambda=1: powers of lambda are one; result is n, not sqrt(n).
        expected_one = (Fraction(3, 2)+Fraction(5, 486)+Fraction(25, 236196)
                        -Fraction(1, 1000)+Fraction(1, 100000)-Fraction(1, 100000000))
        self.assertAlmostEqual(glass.index(1.), float(expected_one), places=14)

    def test_herzberger_polynomial_and_pole_terms_individually(self):
        wavelength = .5
        expected = [1., 500./111., (500./111.)**2, 1./4., 1./16., 1./64.]
        for coefficient, target in enumerate(expected):
            values = [0.]*6
            values[coefficient] = 1.
            self.assertAlmostEqual(_Glass(3, tuple(values), (.3, 2.), 20.).index(wavelength), target,
                                   places=13)

    def test_no_extrapolation_pole_and_nonphysical_index(self):
        glass = _Glass(3, (1.5, .01, .0001, 0., 0., 0.), (.3, 2.), 20.)
        for wavelength in (.2999, 2.0001, 0., -1., float('nan'), float('inf')):
            with self.subTest(wavelength=wavelength), self.assertRaises(ValueError):
                glass.index(wavelength)
        self.assertTrue(math.isfinite(glass.index(.3)))
        self.assertTrue(math.isfinite(glass.index(2.)))
        with self.assertRaisesRegex(ValueError, 'pole'):
            _Glass(3, (1., 1., 0., 0., 0., 0.), (.1, .2), 20.).index(math.sqrt(.028))
        for coefficients in ((-1., 0., 0., 0., 0., 0.), (float('nan'), 0., 0., 0., 0., 0.),
                             (1e308, 1e308, 0., 0., 0., 0.)):
            with self.assertRaisesRegex(ValueError, 'invalid refractive index'):
                _Glass(3, coefficients, (.3, 2.), 20.).index(.55)

    def test_extra_coefficients_fail_closed_and_zero_padding_is_allowed(self):
        for formula, coefficients in ((1, (2.25, 0., 0., 0., 0., 0.)),
                                      (2, (.5, .01, 0., 0., 0., 0.)),
                                      (3, (1.5, .01, .0001, 0., 0., 0.))):
            glass = _Glass(formula, coefficients, (.3, 2.), 20.)
            padded = _Glass(formula, (*coefficients, 0., 0., 0., 0.), (.3, 2.), 20.)
            self.assertEqual(glass.index(.55), padded.index(.55))
            with self.assertRaisesRegex(ValueError, 'Nonzero extra'):
                _Glass(formula, (*coefficients, .1), (.3, 2.), 20.).index(.55)

    def test_existing_schott_and_sellmeier_remain_exact(self):
        self.assertEqual(_Glass(1, (2.25, 0., 0., 0., 0., 0.), (.3, 2.), 20.).index(.55), 1.5)
        expected = math.sqrt(1.+.5*.55**2/(.55**2-.01))
        self.assertAlmostEqual(_Glass(2, (.5, .01, 0., 0., 0., 0.), (.3, 2.), 20.).index(.55),
                               expected, places=15)

    def test_agf_formula_three_coefficients_use_catalog_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'SYNTHETIC.AGF'
            path.write_text('CC Synthetic fixture, not a vendor catalog\n'
                'NM DEMO 3 000000 1.5 50 0 0 0\n'
                'CD 1.5 .01 .0001 -.001 .00001 -.00000001 0 0 0 0\n'
                'LD .3 2\nTD 0 0 0 0 0 0 23\n', encoding='utf-16')
            glasses = _read_catalog(path, {'DEMO'})
            self.assertEqual(glasses['DEMO'].formula, 3)
            self.assertEqual(glasses['DEMO'].reference_temperature_c, 23.)
            self.assertAlmostEqual(glasses['DEMO'].index(.5),
                _Glass(3, (1.5, .01, .0001, -.001, .00001, -.00000001), (.3, 2.), 23.).index(.5),
                places=15)


if __name__ == '__main__':
    unittest.main()
