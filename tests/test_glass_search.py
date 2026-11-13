"""Glass-branch identity, frozen spectra, budgets and acceptance authority."""
import copy
import tempfile
from pathlib import Path
import unittest
from numpy.linalg import LinAlgError

from scripts.ai_zemax.standalone_glass_search import (
    freeze_combination, enumerate_allowed_combinations, run_combination_search)


class Index:
    def __init__(self, status=0, excluded=0):
        self.status, self.excluded, self.calls = status, excluded, []

    def lookup(self, name, **identity):
        self.calls.append((name, identity))
        return dict(vendor=identity["vendor"], name=name, catalog_id=name + "-current",
            snapshot_sha256="a" * 64, release={"latest_verified": True},
            status={"code": self.status}, exclude_substitution_code=self.excluded)

    def refractive_index(self, name, wave, **identity):
        return 1.5 + .001 / wave**2


class GlassSearchTests(unittest.TestCase):
    def test_exact_full_spectrum_and_explicit_current_identity(self):
        index = Index()
        frozen = freeze_combination(index, [{"vendor": "A", "name": "CROWN"}], [.486, .588, .656, .486])
        self.assertEqual(frozen["indices_wave_by_group"][0], frozen["indices_wave_by_group"][3])
        self.assertNotEqual(frozen["indices_wave_by_group"][0], frozen["indices_wave_by_group"][1])
        self.assertTrue(index.calls[0][1]["current_only"])
        with self.assertRaises(ValueError):
            freeze_combination(index, [{"name": "CROWN"}], [.588])

    def test_new_design_blocks_obsolete_and_excluded(self):
        for index in (Index(status=2), Index(excluded=1)):
            with self.assertRaisesRegex(ValueError, "rejects"):
                freeze_combination(index, [{"vendor": "A", "name": "OLD"}], [.588])
            self.assertTrue(freeze_combination(index, [{"vendor": "A", "name": "OLD"}], [.588],
                                               allow_historical=True)["allow_historical"])

    def test_allowlist_cartesian_product_is_bounded(self):
        choices = [[{"vendor": "A", "name": "X"}, {"vendor": "A", "name": "Y"}],
                   [{"vendor": "B", "name": "Z"}]]
        self.assertEqual(len(enumerate_allowed_combinations(choices)), 2)
        with self.assertRaises(ValueError):
            enumerate_allowed_combinations(choices, maximum_combinations=1)

    def run_search(self, optimizer, validator):
        target = {"wavelengths_um": [.486, .588, .656], "limit": .1}
        def realize(seed, glass, t):
            t["limit"] = 900.  # Callback mutations must not change the next callback's target.
            return {"indices_by_wave": glass["indices_wave_by_group"], "material_names": ["X"],
                    "material_catalogs": ["A"], "wavelengths_um": target["wavelengths_um"],
                    "wavelength_weights": [1., 1., 1.], "primary_wave": 1}
        with tempfile.TemporaryDirectory() as directory:
            return run_combination_search(index=Index(), combinations=[
                {"id": branch, "choices": [{"vendor": "A", "name": "X"}]} for branch in ("one", "two")],
                seeds=[{}], target=target, output_directory=Path(directory), realize=realize,
                screen=lambda model, t: 0. if t["limit"] == .1 else float("nan"),
                optimize=optimizer, validate=validator)

    def test_independent_failure_retained_for_every_branch(self):
        result = self.run_search(lambda model, target, directory: (model, {"status": "numerical_feasibility"}),
                                 lambda model, target: {"passed": False, "reason": "dense holdout"})
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual(result["passed_branches"], 0)
        self.assertEqual(result["branches"][0]["runs"][0]["validation"]["reason"], "dense holdout")

    def test_continuous_solver_cannot_change_spectrum_or_indices(self):
        def changed(model, target, directory):
            model["indices_by_wave"][0][0] += .01
            return model, {}
        result = self.run_search(changed, lambda model, target: {"passed": True})
        self.assertFalse(result["all_branches_passed"])
        self.assertIn("frozen", result["branches"][0]["runs"][0]["error"])

    def test_failed_seed_does_not_erase_other_material_branch(self):
        result = self.run_search(lambda model, target, directory: (model, {}),
                                 lambda model, target: {"passed": True, "wfe_worst_waves": .08})
        self.assertEqual(result["passed_branches"], 2)
        self.assertEqual(result["target"]["limit"], .1)

    def test_singular_local_seed_is_retained_as_failure(self):
        def singular(model, target, directory):
            raise LinAlgError("singular trial")
        result = self.run_search(singular, lambda model, target: {"passed": True})
        self.assertEqual(len(result["branches"]), 2)
        self.assertIn("singular", result["branches"][1]["runs"][0]["error"])

    def test_reject_duplicate_ids_bad_budget_and_nonempty_output_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            arguments = dict(index=Index(), combinations=[{"id": "same"}, {"id": "same"}], seeds=[{}],
                target={"wavelengths_um": [.588]}, output_directory=out,
                realize=lambda *a: {}, screen=lambda *a: 0., optimize=lambda *a: ({}, {}),
                validate=lambda *a: {"passed": True})
            with self.assertRaisesRegex(ValueError, "unique"):
                run_combination_search(**arguments)
            self.assertFalse(out.exists())
            arguments["combinations"] = [{"id": "one"}]
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_combination_search(**arguments, seeds_per_branch=0)
            self.assertFalse(out.exists())
            out.mkdir()
            (out / "evidence.json").write_text("keep")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run_combination_search(**arguments)
            self.assertEqual((out / "evidence.json").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
