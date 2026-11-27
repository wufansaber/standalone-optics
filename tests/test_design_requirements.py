"""Independent physical acceptance and requirement-driven search ordering."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ai_zemax.standalone_branch_budget import run_budgeted_branches
from scripts.ai_zemax.standalone_design_requirements import DesignRequirements
from scripts.ai_zemax.standalone_glass_search import run_combination_search


CASES = {
    "small_spot": {"spot": 1.7, "wavefront": .168},
    "low_wavefront": {"spot": 1.9, "wavefront": .145},
    "collision": {"spot": .1, "wavefront": .01},
}


class DesignRequirementTests(unittest.TestCase):
    def branches(self, priority, limits, cases=CASES, *, budgets=(1,), checker=None):
        def advance(branch, state, budget, target):
            name = branch["id"]
            return {"valid": True, "state": {"id": name}, "metrics": cases[name]}

        def validate(branch, state, target):
            name = state["id"]
            return {"passed": name != "collision", "metrics": cases[name]}

        with tempfile.TemporaryDirectory() as tmp:
            result = run_budgeted_branches([{"id": name} for name in cases],
                target={"designRequirements": {"priority": priority, "limits": limits}},
                rung_budgets=budgets, advance=advance, validate=checker or validate,
                output_directory=Path(tmp) / "run", workers=2, explore_slots=0)
            saved = json.loads((Path(tmp) / "run" / "RESULT.json").read_text())
            self.assertEqual(saved["selection"], result["selection"])
            return result

    def glass(self, priority, limits, *, count=3, checker=None, screening=None):
        target = {"wavelengths_um": [.588],
                  "designRequirements": {"priority": priority, "limits": limits}}
        def validate(model, target):
            name = model["id"]
            return {"passed": name != "collision", "metrics": CASES[name],
                    "wfe_worst_waves": CASES[name]["wavefront"]}

        with tempfile.TemporaryDirectory() as tmp, patch(
                "scripts.ai_zemax.standalone_glass_search.freeze_combination",
                return_value={"materials": []}):
            return run_combination_search(index=None, combinations=[{"id": "A", "choices": []}],
                seeds=[{"id": name} for name in CASES], target=target, output_directory=tmp,
                realize=lambda seed, glass, target: seed, seeds_per_branch=count,
                screen=screening or (lambda model, target: {"metrics": CASES[model["id"]]}),
                optimize=lambda model, target, folder: (model, {}),
                validate=checker or validate)

    def test_optical_priority_does_not_override_physical_veto(self):
        for metric, name in (("spot", "small_spot"), ("wavefront", "low_wavefront")):
            with self.subTest(metric=metric):
                result = self.branches(metric, {})
                self.assertEqual(result["selection"]["candidate"], name)
                self.assertEqual(result["selection"]["status"], "ranked")
                self.assertIsNone(result["selection"]["targetMet"])
                result = self.glass(metric, {})
                self.assertEqual(result["selection"]["candidate"], ["A", list(CASES).index(name)])
                self.assertEqual(result["selection"]["status"], "ranked")

    def test_hard_secondary_limit_survives_successive_halving(self):
        cases = {name: CASES[name] for name in ("small_spot", "low_wavefront")}
        result = self.branches("spot", {"wavefront": .15}, cases, budgets=(1, 2))
        self.assertEqual(result["history"][0]["selected"], ["low_wavefront"])
        self.assertTrue(result["selection"]["targetMet"])
        self.assertEqual(result["allocated_work"], 4)
        def screened(model, target):
            return {"metrics": CASES[model["id"]], "valid": model["id"] != "collision"}
        result = self.glass("spot", {"wavefront": .15}, count=1, screening=screened)
        self.assertEqual(result["selection"]["candidate"], ["A", 1])

    def test_target_status_comes_from_dense_validation(self):
        def dense(model, *args):
            if model["id"] == "collision":
                return {"passed": False}
            return {"passed": True, "metrics": {"spot": 2., "wavefront": .20}}
        for result in (self.branches("spot", {"wavefront": .15}, checker=dense),
                       self.glass("spot", {"wavefront": .15}, checker=dense)):
            self.assertEqual(result["selection"]["status"], "out_of_spec")
            self.assertAlmostEqual(result["selection"]["maxRelativeViolation"], 1 / 3)
            self.assertFalse(result["selection"]["targetMet"])
        self.assertIsNone(result["branches"][0]["selected_seed_index"])

    def test_missing_required_evidence_does_not_accept_any_candidate(self):
        def missing(model, *args):
            return {"passed": True, "metrics": {"spot": 1.}}
        for result in (self.branches("spot", {"wavefront": .15}, checker=missing),
                       self.glass("spot", {"wavefront": .15}, checker=missing)):
            self.assertEqual(result["selection"]["status"], "unverified")
            self.assertIsNone(result["selection"]["candidate"])
        result = self.glass("wavefront", {}, screening=lambda *args: {"metrics": {"spot": 1.}})
        self.assertEqual(result["branches"][0]["physical_seed_count"], 0)

    def test_only_requested_metrics_and_positive_limits(self):
        req = DesignRequirements.from_dict({"priority": "spot", "limits": {"spot": 2.}})
        self.assertTrue(req.assess({"spot": 2., "wavefront": None})["targetMet"])
        for value in (0, -1, True, float("inf")):
            with self.assertRaises(ValueError):
                DesignRequirements("spot", {"spot": value})
        with self.assertRaises(ValueError):
            req.assess({"spot": float("nan")})


if __name__ == "__main__":
    unittest.main()
