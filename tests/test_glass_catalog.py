"""Synthetic-only regressions for immutable offline material lookup."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from scripts.ai_zemax.standalone_glass_catalog import (
    GlassCatalogIndex, import_catalogs, main, prepare_indices, refractive_index)


def agf(name="TEST", *, formula=1, coefficients="2.25 0 0 0 0 0", bounds="LD 0.3 1.2",
        status=0, nd=1.5, vd=60.):
    return (f"NM {name} {formula} 500600 {nd} {vd} 0 {status} 2\n"
            "GC Synthetic material for software regression only\n"
            f"CD {coefficients}\n{bounds}\nTD 0 0 0 0 0 0 20\n")


class GlassCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, filename="SCHOTT.AGF", text=None, encoding="utf-8"):
        path = self.root/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text is not None else agf(), encoding=encoding)
        return path

    def index(self, sources=None, **options):
        return import_catalogs(sources or [self.source()], self.root/"cache/index.json", **options)

    def test_import_query_metadata_and_exact_formula(self):
        index = self.index([self.source(text=agf()+agf("OLD", status=2, nd=1.6, vd=45.))])
        record = index.lookup("test", vendor="schott")
        self.assertEqual(record["status"]["label"], "standard")
        self.assertEqual(record["melt_frequency_code"], 2)
        self.assertEqual(record["release"]["status"], "local_unverified_release")
        self.assertIsNone(record["release"]["version_label"])
        self.assertTrue(record["dispersion"]["ready_for_current_tracer"])
        self.assertEqual(index.refractive_index("TEST", .355, vendor="SCHOTT"), 1.5)
        self.assertEqual([r["name"] for r in index.search(nd_min=1.55, vd_max=50.)], ["OLD"])
        old = index.search(status="obsolete")[0]
        self.assertTrue(old["status"]["historical_or_obsolete"])
        self.assertEqual(old["status"]["procurement_availability"], "unknown")

    def test_filename_is_not_a_verified_release(self):
        path = self.source("latest_20991231.AGF")
        with self.assertRaisesRegex(ValueError, "Explicit vendor"):
            self.index([path])
        index = self.index([path], vendor="CDGM")
        release = index.lookup("TEST")["release"]
        self.assertIsNone(release["release_date"])
        self.assertFalse(release["latest_verified"])

    def test_utf16_and_exact_sellmeier(self):
        nd = math.sqrt(1.+.5*.5875618**2/(.5875618**2-.01))
        nf = math.sqrt(1.+.5*.4861327**2/(.4861327**2-.01))
        nc = math.sqrt(1.+.5*.6562725**2/(.6562725**2-.01))
        path = self.source(text=agf(formula=2, coefficients="0.5 0.01 0 0 0 0",
                                   nd=nd, vd=(nd-1.)/(nf-nc)), encoding="utf-16")
        index = self.index([path])
        value = index.refractive_index("TEST", .55)
        self.assertAlmostEqual(value, math.sqrt(1.+.5*.55**2/(.55**2-.01)), places=14)
        self.assertEqual(value, refractive_index(index.path, "TEST", .55, vendor="SCHOTT"))

    def test_unsupported_missing_range_and_extra_coefficients_block(self):
        data = (agf("UNKNOWN", formula=999)+agf("NO_LD", bounds="")+
                agf("EXTRA", coefficients="2.25 0 0 0 0 0 0.1"))
        index = self.index([self.source(text=data)])
        self.assertEqual(index.search(ready_only=True), [])
        for name, reason in (("UNKNOWN", "unsupported_dispersion_formula"),
                             ("NO_LD", "missing_wavelength_validity"),
                             ("EXTRA", "nonzero_extra_coefficients")):
            self.assertIn(reason, " ".join(index.lookup(name)["dispersion"]["blocked_reasons"]))
            with self.assertRaisesRegex(ValueError, "not ready"):
                index.refractive_index(name, .55)

    def test_wavelength_no_extrapolation_or_nonfinite_inputs(self):
        index = self.index()
        for wavelength in (.299, 1.201, 0., -1., float("nan"), float("inf")):
            with self.subTest(wavelength=wavelength), self.assertRaises(ValueError):
                index.refractive_index("TEST", wavelength)
        self.assertEqual(index.refractive_index("TEST", .3), 1.5)
        self.assertEqual(index.refractive_index("TEST", 1.2), 1.5)
        self.assertEqual(index.search(wavelength_um=.2), [])

    def test_exact_herzberger_in_ir_keeps_visible_placeholder_metadata_out_of_checks(self):
        data = agf("IR", formula=3, coefficients="2.5 0.01 0.001 0.0001 0 0",
                   bounds="LD 2 14", nd=1., vd=0.)
        index = self.index([self.source(text=data)])
        record = index.lookup("IR")
        self.assertTrue(record["dispersion"]["ready_for_current_tracer"])
        self.assertFalse(record["data_consistency"]["checked"])
        inverse = 1./(4.-.028)
        self.assertAlmostEqual(index.refractive_index("IR", 2.),
                               2.5+.01*inverse+.001*inverse**2+.0001*4., places=14)
        with self.assertRaises(ValueError):
            index.refractive_index("IR", .355)

    def test_cross_vendor_homonyms_require_identity_even_if_equal(self):
        index = self.index([self.source("SCHOTT.AGF"), self.source("OHARA.AGF")])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            index.lookup("TEST")
        self.assertEqual(index.refractive_index("TEST", .55, vendor="OHARA"), 1.5)

    def test_multiple_releases_require_catalog_id(self):
        paths = [self.source("old.agf"), self.source("new.agf", agf(nd=1.6, coefficients="2.56 0 0 0 0 0"))]
        metadata = {p.name: {"vendor": "SCHOTT", "catalog_name": "SCHOTT"} for p in paths}
        index = self.index(paths, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            index.lookup("TEST", vendor="SCHOTT")
        for row in index.list():
            self.assertIn(index.refractive_index("TEST", .55, catalog_id=row["catalog_id"]), (1.5, 1.6))

    def test_duplicate_material_records_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Duplicate material"):
            self.index([self.source(text=agf()+agf())])
        self.assertFalse((self.root/"cache/index.json").exists())

    def test_snapshot_source_changes_do_not_update_indices(self):
        path = self.source()
        index = self.index([path])
        path.write_text(agf(coefficients="2.56 0 0 0 0 0"), encoding="utf-8")
        self.assertEqual(index.refractive_index("TEST", .55), 1.5)
        with self.assertRaises(FileExistsError):
            import_catalogs([path], index.path)
        snapshot = index.path.parent/index.catalogs[0]["snapshot_path"]
        snapshot.write_text(agf(coefficients="2.56 0 0 0 0 0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            GlassCatalogIndex(index.path).refractive_index("TEST", .55)

    def test_snapshot_path_cannot_escape_cache(self):
        index = self.index()
        data = deepcopy(index.document)
        data["catalogs"][0]["snapshot_path"] = "../SCHOTT.AGF"
        evil = index.path.parent/"tampered-index.json"
        evil.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            GlassCatalogIndex(evil).refractive_index("TEST", .55)

    def test_official_latest_metadata_requires_hash_binding(self):
        path = self.source()
        common = {"vendor": "SCHOTT", "official_source_url": "https://www.schott.com/fixture.agf",
                  "version_label": "synthetic", "latest_verified": True, "latest_verified_as_of": "2026-11-09",
                  "verified_at_utc": "2026-11-09T00:00:00Z"}
        index = self.index([path], metadata={path.name: common})
        release = index.lookup("TEST")["release"]
        self.assertEqual(release["status"], "official_metadata_only_local_release_unverified")
        self.assertFalse(release["latest_verified"])
        common["verified_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        verified = import_catalogs([path], self.root/"verified/index.json", metadata={path.name: common})
        self.assertTrue(verified.lookup("TEST")["release"]["latest_verified"])
        self.assertEqual(verified.lookup("TEST")["release"]["latest_verified_as_of"], "2026-11-09")
        common["verified_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            import_catalogs([path], self.root/"invalid/index.json", metadata={path.name: common})

    def test_snapshot_directory_symlink_or_junction_cannot_escape_index_parent(self):
        path = self.source()
        outside = self.root/"outside-cache"
        outside.mkdir()
        cache = self.root/"cache"
        cache.mkdir()
        link = cache/"catalog-snapshots"
        try:
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(outside), str(link))
            else:
                os.symlink(outside, link, target_is_directory=True)
        except (OSError, AttributeError) as error:
            self.skipTest(f"Directory links unavailable in this test environment: {error}")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.index([path])
        self.assertFalse((cache/"index.json").exists())
        self.assertEqual(list(outside.iterdir()), [])

    def test_latest_label_needs_explicit_as_of_even_with_matching_hash(self):
        path = self.source()
        detail = {"vendor": "SCHOTT", "official_source_url": "https://www.schott.com/fixture.agf",
                  "verified_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "verified_at_utc": "2026-11-09T00:00:00Z", "latest_verified": True}
        index = self.index([path], metadata={path.name: detail})
        self.assertFalse(index.lookup("TEST")["release"]["latest_verified"])
        self.assertIsNone(index.lookup("TEST")["release"]["latest_verified_as_of"])

    def model(self):
        return {"vertices_mm": [10., 12., 100.], "surface_rows": [{}, {}, {}],
                "material_names": [None, "TEST", None, None],
                "material_catalogs": [None, "SCHOTT", None, None],
                "wavelengths_um": [.55, .6, .55], "wavelength_weights": [1., .5, .0625],
                "primary_wave": 0, "indices_by_wave": [[1., 1.4, 1., 1.]]*3}

    def test_prepare_preserves_existing_and_only_changes_explicit_channels(self):
        index = self.index()
        model = self.model()
        model["nominal_verification"] = {"passed": True}
        before = deepcopy(model)
        unchanged, provenance = index.prepare_indices(model)
        self.assertEqual(unchanged["indices_by_wave"], model["indices_by_wave"])
        self.assertEqual(provenance["preserved_existing_medium_indices"], [1])
        prepared, manifest = prepare_indices(index, model, {1: {"vendor": "SCHOTT", "name": "TEST"}})
        self.assertNotIn("nominal_verification", prepared)
        self.assertEqual(prepared["catalog_preparation"]["status"], "unverified_indices_prepared")
        self.assertEqual([row[1] for row in prepared["indices_by_wave"]], [1.5]*3)
        self.assertEqual(len(manifest["resolved_media"][0]["indices_by_wave"]), 3)
        for key in ("vertices_mm", "surface_rows", "wavelengths_um", "wavelength_weights", "primary_wave"):
            self.assertEqual(prepared[key], before[key])
        self.assertEqual(model, before)
        with self.assertRaisesRegex(ValueError, "solid"):
            index.prepare_indices(model, {2: {"name": "TEST", "vendor": "SCHOTT"}})

    def test_prepare_new_neutral_model_and_explicit_recompute(self):
        index = self.index()
        model = self.model()
        no_table = deepcopy(model)
        del no_table["indices_by_wave"]
        prepared, _ = index.prepare_indices(no_table)
        self.assertEqual([row[1] for row in prepared["indices_by_wave"]], [1.5]*3)
        recomputed, _ = index.prepare_indices(model, recompute_existing=True)
        self.assertEqual(recomputed["indices_by_wave"], prepared["indices_by_wave"])

    def test_prepare_retains_independent_duplicate_spectral_slots_with_aliased_input_rows(self):
        def n(w):
            return math.sqrt(1.+.5*w*w/(w*w-.01))
        nd = n(.5875618)
        index = self.index([self.source(text=agf(formula=2, coefficients="0.5 0.01 0 0 0 0",
            nd=nd, vd=(nd-1.)/(n(.4861327)-n(.6562725))))])
        model = self.model()  # Existing rows intentionally share one Python list.
        prepared, _ = index.prepare_indices(model, {1: {"vendor": "SCHOTT", "name": "TEST"}})
        actual = [row[1] for row in prepared["indices_by_wave"]]
        self.assertEqual(actual, [n(w) for w in model["wavelengths_um"]])
        self.assertNotEqual(actual[0], actual[1])
        self.assertEqual(actual[0], actual[2])

    def test_cli_import_index_and_search(self):
        path = self.source("catalog with spaces.agf")
        index_path = self.root/"cache/index.json"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["import", str(path), "--index", str(index_path), "--vendor", "CDGM"]), 0)
        self.assertEqual(json.loads(output.getvalue())["material_count"], 1)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["index", "TEST", "--index", str(index_path), "--vendor", "CDGM", "--wavelength", ".55"]), 0)
        self.assertEqual(json.loads(output.getvalue())["refractive_index"], 1.5)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["search", "test", "--index", str(index_path), "--nd-min", "1.4", "--ready-only"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())), 1)

    def test_current_registry_does_not_implicitly_select_all_products(self):
        paths = [self.source("current.agf"), self.source("all.agf", agf()+agf("HISTORICAL", status=2))]
        metadata = {p.name: {"vendor": "OHARA", "catalog_name": p.stem,
                             "default_current": p.stem == "current"} for p in paths}
        index = self.index(paths, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            index.lookup("TEST", vendor="OHARA")
        self.assertEqual(index.lookup("TEST", vendor="OHARA", current_only=True)["catalog_name"], "CURRENT")
        self.assertEqual(index.search(name="HISTORICAL", current_only=True), [])
        self.assertEqual(len(index.search(name="HISTORICAL")), 1)

    def test_transmission_and_uninterpreted_physical_records_preserved(self):
        data = agf()+"IT 0.355 0.9 10\nED 7.1 8.2 2.5\nOD 1.2 -1 -1\n"
        index = self.index([self.source(text=data)])
        record = index.lookup("TEST")
        transmission = record["internal_transmission"]["samples"][0]
        self.assertEqual(transmission["wavelength_um"], .355)
        self.assertEqual(transmission["reference_thickness_mm"], 10.)
        self.assertEqual(transmission["internal_transmission"], .9)
        self.assertEqual(record["additional_agf_records"]["ED"]["raw_fields"], ["7.1", "8.2", "2.5"])
        self.assertAlmostEqual(record["additional_agf_records"]["ED"]["decoded"]["cte_minus30_to70_per_kelvin"], 7.1e-6)
        self.assertEqual(record["additional_agf_records"]["ED"]["decoded"]["density_g_cm3"], 2.5)
        self.assertIsNone(record["additional_agf_records"]["OD"]["decoded"]["CR"])
        self.assertFalse(record["additional_agf_records"]["TD"]["applied_by_current_reference_temperature_tracer"])
        self.assertEqual(record["additional_agf_records"]["TD"]["raw_fields"][-1], "20")

    def test_gross_nm_coefficient_inconsistency_requires_diagnostic_override(self):
        index = self.index([self.source(text=agf(nd=1.7))])
        record = index.lookup("TEST")
        self.assertIn("NM_nd_and_dispersion_coefficients_disagree", record["data_consistency"]["warnings"])
        self.assertEqual(index.search(selection_eligible_only=True), [])
        self.assertEqual(record["nd"], 1.7)  # Preserve supplier metadata; do not repair it.
        with self.assertRaisesRegex(ValueError, "data-consistency"):
            index.refractive_index("TEST", .55)
        self.assertEqual(index.refractive_index("TEST", .55, allow_flagged_data=True), 1.5)
        with self.assertRaisesRegex(ValueError, "data-consistency"):
            index.prepare_indices(self.model(), {1: {"vendor": "SCHOTT", "name": "TEST"}})
        diagnostic, provenance = index.prepare_indices(self.model(),
            {1: {"vendor": "SCHOTT", "name": "TEST"}}, allow_flagged_data=True)
        self.assertEqual(diagnostic["indices_by_wave"][0][1], 1.5)
        self.assertTrue(provenance["allow_flagged_data_explicitly_requested"])

    def test_missing_optional_nm_fields_are_unknown(self):
        data = agf().replace("NM TEST 1 500600 1.5 60.0 0 0 2", "NM TEST 1")
        index = self.index([self.source(text=data)])
        record = index.lookup("TEST")
        self.assertIsNone(record["nd"])
        self.assertIsNone(record["status"]["label"])
        self.assertIsNone(record["status"]["historical_or_obsolete"])


if __name__ == "__main__":
    unittest.main()
