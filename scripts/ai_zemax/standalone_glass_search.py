"""Discrete real-glass branches around continuous optical optimization.

Geometry and merit functions are callbacks, so this orchestration is not tied
to singlets or to the example's acceptance limits. Every retained material
branch uses the same immutable target and independent final validator.
"""
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from numpy.linalg import LinAlgError


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def enumerate_allowed_combinations(group_choices, *, maximum_combinations=256):
    """Enumerate only explicitly allowed identities for each physical group."""
    if not group_choices or any(not choices for choices in group_choices):
        raise ValueError("Every physical material group needs a nonempty explicit allowlist")
    count = 1
    for choices in group_choices:
        count *= len(choices)
    if count > maximum_combinations:
        raise ValueError("Material product exceeds the declared branch budget")
    return [{"id": f"combination-{i:04d}", "choices": list(deepcopy(choices))}
            for i, choices in enumerate(product(*group_choices))]


def freeze_combination(index, choices, wavelengths_um, *, allow_historical=False):
    """Explicit current vendor identities and exact full-spectrum indices.

    Choices are ordered optical groups, each {'vendor': ..., 'name': ...}.
    No nd/Vd glass substitution or cross-vendor fallback is performed.
    """
    rows = []
    for choice in choices:
        if not choice.get("vendor") or not choice.get("name"):
            raise ValueError("Every glass choice requires explicit vendor and name")
        record = index.lookup(choice["name"], vendor=choice["vendor"], current_only=True)
        excluded = record.get("exclude_substitution_code") == 1
        obsolete = record.get("status", {}).get("code") == 2
        if (excluded or obsolete) and not allow_historical:
            raise ValueError("New-design selection rejects excluded/obsolete material " + record["name"])
        values = [index.refractive_index(record["name"], wavelength,
                  catalog_id=record["catalog_id"]) for wavelength in wavelengths_um]
        rows.append({key: deepcopy(record[key]) for key in
            ("vendor", "name", "catalog_id", "snapshot_sha256", "release")})
        rows[-1]["indices_by_wave"] = values
        rows[-1]["status"] = deepcopy(record.get("status"))
        rows[-1]["exclude_substitution_code"] = record.get("exclude_substitution_code")
        rows[-1]["nd"] = record.get("nd")
        rows[-1]["vd"] = record.get("vd")
    return {"materials": rows, "wavelengths_um": list(wavelengths_um),
        "indices_wave_by_group": [list(row) for row in zip(*(item["indices_by_wave"] for item in rows))],
        "selection": "explicit_vendor_and_current_registry", "allow_historical": bool(allow_historical),
        "procurement_availability": "not_verified_by_calculability_or_current_catalog_membership", "sha256": _hash(rows)}


def run_combination_search(*, index, combinations, seeds, target, output_directory,
                           realize, screen, optimize, validate, seeds_per_branch=2):
    """Run mixed discrete/continuous search; retain successes and failures.

    realize(seed, frozen, target) -> neutral model
    screen(model, target) -> finite ranking score (lower is better)
    optimize(model, target, directory) -> (candidate, optimization metadata)
    validate(candidate, target) -> dict with a boolean passed

    Independent validation alone can mark a branch passed. A numerical solver
    convergence flag has no acceptance authority. The target is deep-copied
    for every callback and checked against mutation after each branch.
    """
    if type(seeds_per_branch) is not int or seeds_per_branch < 1:
        raise ValueError("seeds_per_branch must be a positive integer")
    if not isinstance(combinations, (list, tuple)) or not combinations:
        raise ValueError("Provide a nonempty explicit material-combination list")
    ids = []
    for number, combo in enumerate(combinations):
        name = combo.get("id", f"branch-{number:02d}")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Branch id must be a safe alphanumeric identifier")
        ids.append(name)
    if len(set(ids)) != len(ids):
        raise ValueError("Material branch IDs must be unique")
    out = Path(output_directory)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a fresh empty output directory; existing search evidence is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    fixed_target = deepcopy(target)
    target_hash = _hash(fixed_target)
    started = perf_counter()
    branches = []
    for number, combo in enumerate(combinations):
        name = ids[number]
        folder = out / name
        folder.mkdir(exist_ok=True)
        branch = {"id": name, "target_sha256": target_hash, "passed": False,
                  "generated_seed_count": len(seeds), "seeds": [], "runs": []}
        tick = perf_counter()
        try:
            frozen = freeze_combination(index, combo["choices"], fixed_target["wavelengths_um"])
            branch["material_combination"] = frozen
            ranked = []
            for si, seed in enumerate(seeds):
                item = {"seed_index": si, "status": "rejected"}
                try:
                    model = realize(deepcopy(seed), deepcopy(frozen), deepcopy(fixed_target))
                    screening = screen(deepcopy(model), deepcopy(fixed_target))
                    score = float(screening["score"] if isinstance(screening, dict) else screening)
                    if not __import__("math").isfinite(score):
                        raise ValueError("Nonfinite screening score")
                    item.update(status="physical_seed", score=score)
                    if isinstance(screening, dict):
                        item["screening"] = screening
                    ranked.append((score, si, model))
                except (ValueError, RuntimeError, FloatingPointError, LinAlgError) as error:
                    item["error"] = str(error)
                branch["seeds"].append(item)
            branch["physical_seed_count"] = len(ranked)
            for _, si, model in sorted(ranked, key=lambda row: (row[0], row[1]))[:seeds_per_branch]:
                run_folder = folder / f"seed-{si:02d}"
                run_folder.mkdir(exist_ok=True)
                run = {"seed_index": si, "passed": False}
                try:
                    candidate, metadata = optimize(deepcopy(model), deepcopy(fixed_target), run_folder)
                    # Refractive indices must remain exactly frozen in local refinement.
                    frozen_keys = ("indices_by_wave", "material_names", "material_catalogs", "wavelengths_um",
                                   "wavelength_weights", "primary_wave")
                    if any(candidate.get(key) != model.get(key) for key in frozen_keys):
                        raise ValueError("Continuous optimizer changed frozen physical glass identity/spectrum/indices")
                    checked = validate(deepcopy(candidate), deepcopy(fixed_target))
                    if type(checked.get("passed")) is not bool:
                        raise ValueError("Independent validator requires a boolean passed")
                    run.update(optimizer=metadata, validation=checked, passed=checked["passed"])
                    (run_folder / "CANDIDATE.json").write_text(json.dumps(candidate, indent=2, allow_nan=False), encoding="utf-8")
                except (ValueError, RuntimeError, FloatingPointError, LinAlgError) as error:
                    run["error"] = str(error)
                branch["runs"].append(run)
                (run_folder / "RESULT.json").write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
            passed = [run for run in branch["runs"] if run["passed"]]
            branch["passed"] = bool(passed)
            branch["selected_seed_index"] = min(passed, key=lambda r: r["validation"].get("wfe_worst_waves", float("inf")))["seed_index"] if passed else None
        except (KeyError, ValueError, RuntimeError, LinAlgError) as error:
            branch["error"] = str(error)
        if _hash(fixed_target) != target_hash:
            raise AssertionError("The common declared target changed")
        branch["seconds"] = perf_counter() - tick
        branches.append(branch)
        (folder / "BRANCH.json").write_text(json.dumps(branch, indent=2, allow_nan=False), encoding="utf-8")
        print(name, "passed", branch["passed"], "physical seeds", branch.get("physical_seed_count", 0),
              "selected", branch.get("selected_seed_index"), "seconds", round(branch["seconds"], 3), flush=True)
    result = {"schema": "standalone_optics.glass_combination_search.v1", "target": fixed_target,
        "target_sha256": target_hash, "branches": branches, "branch_count": len(branches),
        "passed_branches": sum(branch["passed"] for branch in branches),
        "all_branches_passed": bool(branches) and all(branch["passed"] for branch in branches),
        "seconds": perf_counter() - started,
        "claim": "independently checked finite-sampled local solutions; no global optimum or universal seed-convergence guarantee"}
    (out / "SEARCH.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
