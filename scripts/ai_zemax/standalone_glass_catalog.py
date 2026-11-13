"""Offline, content-addressed local AGF catalog lookup for optical design.

Vendor catalogs are user-supplied data.  Import copies immutable snapshots to
the explicitly selected index directory; this package ships no vendor AGF.
Updating a local source never changes an existing index or prescription.
Only the exact formulas implemented by standalone_model._Glass are evaluated.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from collections.abc import Mapping

from .standalone_model import _decode, _finite, _read_catalog


SCHEMA = "standalone_optics.local_glass_catalog.v1"
SUPPORTED_FORMULAS = {1: "Schott", 2: "Sellmeier 1", 3: "Herzberger"}
AGF_SPEC_URL = "https://optics.ansys.com/hc/en-us/articles/46891219767827-Zemax-AGF-material-catalog-file-format"
STATUS = {0: "standard", 1: "preferred", 2: "obsolete", 3: "special", 4: "melt"}
KNOWN_VENDOR_STEMS = {name: name for name in ("CDGM", "SCHOTT", "OHARA")}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identifier(value, label):
    value = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]*", value):
        raise ValueError(f"{label} must contain only letters, digits, dot, underscore or hyphen")
    return value


def _catalog_label(value):
    value = str(value).strip().upper()
    if not value or not value.isprintable() or "::" in value:
        raise ValueError("Catalog name must be a nonempty printable label without '::'")
    return value


def _optional_number(columns, index):
    if len(columns) <= index or columns[index].strip().upper() in ("-", "--", "?", "N/A", "UNKNOWN"):
        return None
    return _finite(columns[index])


def _optional_int(columns, index):
    value = _optional_number(columns, index)
    if value is not None and value != int(value):
        raise ValueError("AGF integer field contains a noninteger value")
    return int(value) if value is not None else None


def _physical_record(tag, fields):
    raw = list(fields)
    def number(index):
        try:
            return _optional_number(raw, index)
        except ValueError:
            return None
    if tag == "ED":
        first, second = number(0), number(1)
        density, ignore = number(2), number(4)
        decoded = {"cte_minus30_to70_per_kelvin": first*1.e-6 if first is not None else None,
                   "cte_100_to300_per_kelvin": second*1.e-6 if second is not None else None,
                   "catalog_cte_units": "1e-6/K",
                   "density_g_cm3": density if density is not None and density > 0. else None,
                   "dPgF": number(3), "ignore_thermal_expansion_code": ignore}
    elif tag == "TD":
        decoded = {key: number(i) for i, key in enumerate(
            ("D0", "D1", "D2", "E0", "E1", "Ltk_um", "reference_temperature_c"))}
    else:
        decoded = {key: (value if value != -1 else None) for key, value in zip(
            ("relative_cost_BK7_equals_one", "CR", "FR", "SR", "AR", "PR"),
            (number(i) for i in range(6)))}
    return {"raw_fields": raw, "decoded": decoded, "field_definition_source": AGF_SPEC_URL,
            "interpretation": "AGF specification; raw vendor values retained",
            "applied_by_current_reference_temperature_tracer": False}


class _CatalogBytes:
    """Feed the existing parser the exact bytes that were hashed, without a reread race."""
    def __init__(self, raw, name):
        self.raw, self.name = raw, name

    def read_bytes(self):
        return self.raw


def _record_metadata(raw):
    """Read NM metadata without interpreting filename dates as releases."""
    records, headers = {}, []
    current = None
    for line in _decode(raw).splitlines():
        values = line.split()
        if not values:
            continue
        if values[0] == "NM":
            if len(values) < 3:
                raise ValueError("AGF NM record lacks material name or formula")
            name = values[1].upper()
            if name in records:
                raise ValueError(f"Duplicate material {name} within one AGF snapshot")
            # NM tag, name, formula, glass code, nd, vd, exclude, status,
            # melt frequency.  Optional fields are never filled with guesses.
            status = _optional_int(values, 7)
            records[name] = {
                "name": name, "formula": int(values[2]),
                "nm_raw_fields": values[1:],
                "glass_code": values[3] if len(values) > 3 else None,
                "nd": _optional_number(values, 4), "vd": _optional_number(values, 5),
                "exclude_substitution_code": _optional_int(values, 6),
                "status": {"code": status, "label": STATUS.get(status),
                           "interpretation": "AGF specification; vendor semantics require confirmation",
                           "interpretation_source": AGF_SPEC_URL,
                           "historical_or_obsolete": True if status == 2 else None,
                           "procurement_availability": "unknown"},
                "melt_frequency_code": _optional_int(values, 8),
                "comments": [], "reference_temperature_declared": False,
                "dispersion_coefficient_raw_fields": [],
                "internal_transmission": {"samples": [], "interpretation_source": AGF_SPEC_URL,
                    "selection_warning": "LD is dispersion validity, not transmission; no spectral transmission is inferred."},
                "additional_agf_records": {"ED": None, "TD": None, "OD": None},
            }
            current = name
        elif current is None:
            # Header text is evidence only, never an inferred release number.
            if values[0] in ("CC", "GC") and len(headers) < 20:
                headers.append(" ".join(values[1:])[:2000])
        elif values[0] == "GC":
            records[current]["comments"].append(" ".join(values[1:])[:2000])
        elif values[0] == "CD":
            records[current]["dispersion_coefficient_raw_fields"] = values[1:]
        elif values[0] in ("ED", "TD", "OD"):
            records[current]["additional_agf_records"][values[0]] = _physical_record(values[0], values[1:])
            if values[0] == "TD":
                records[current]["reference_temperature_declared"] = len(values) >= 8
        elif values[0] == "IT":
            sample = {"raw_fields": values[1:], "decoded": False}
            if len(values) == 4:
                try:
                    wavelength, transmission, thickness = map(_finite, values[1:])
                    if wavelength > 0. and 0. <= transmission <= 1. and thickness > 0.:
                        sample.update(decoded=True, wavelength_um=wavelength,
                                      internal_transmission=transmission, reference_thickness_mm=thickness)
                except ValueError:
                    sample["parse_warning"] = "unrecognized_transmission_fields_preserved_without_interpretation"
            records[current]["internal_transmission"]["samples"].append(sample)
    if not records:
        raise ValueError("AGF contains no NM material records")
    return records, headers


def _release_metadata(metadata, digest):
    """A latest-release page cannot establish the identity of local bytes."""
    expected = metadata.get("verified_sha256")
    if expected is not None and str(expected).lower() != digest:
        raise ValueError("Local snapshot SHA-256 does not match the declared official release")
    official_url = metadata.get("official_source_url")
    verified_at = metadata.get("verified_at_utc")
    latest_as_of = metadata.get("latest_verified_as_of")
    if latest_as_of is not None:
        try:
            datetime.fromisoformat(str(latest_as_of).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("latest_verified_as_of must be an explicit ISO date or timestamp") from error
    latest = metadata.get("latest_verified") is True and latest_as_of is not None
    if expected and official_url and verified_at:
        status = ("snapshot_matches_declared_latest_official_release" if latest
                  else "snapshot_matches_declared_official_release")
    elif official_url:
        status = "official_metadata_only_local_release_unverified"
    else:
        status = "local_unverified_release"
    return {
        "status": status, "version_label": metadata.get("version_label"),
        "release_date": metadata.get("release_date"),
        "release_date_basis": metadata.get("release_date_basis"),
        "official_source_url": official_url, "verified_at_utc": verified_at,
        "downloaded_at_utc": metadata.get("downloaded_at_utc"),
        "verified_sha256": str(expected).lower() if expected else None,
        "latest_verified": bool(expected and official_url and verified_at and latest),
        "latest_verified_as_of": latest_as_of,
        "verification_scope": "importer-supplied metadata; no network verification performed",
    }


def _contained(parent, path):
    resolved = path.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as error:
        raise ValueError("Catalog snapshot path escapes the selected index directory") from error
    return resolved


def _consistency(glass, record):
    """Conservative coefficient/NM cross-check, not independent optical validation."""
    result = {"checked": False, "warnings": [], "computed_nd": None, "computed_vd": None,
              "nd_absolute_warning_threshold": 1.e-4, "vd_absolute_warning_threshold": .1,
              "basis": "same AGF formula at d/F/C; diagnostic thresholds, no coefficient repair"}
    wavelengths = (.5875618, .4861327, .6562725)
    if (glass.formula not in SUPPORTED_FORMULAS or len(glass.coefficients) < 6
            or not glass.bounds_um or any(not glass.bounds_um[0] <= w <= glass.bounds_um[1] for w in wavelengths)):
        return result
    try:
        nd, nf, nc = map(glass.index, wavelengths)
    except ValueError:
        result["warnings"].append("standard_line_dispersion_evaluation_failed")
        return result
    result.update(checked=True, computed_nd=nd)
    if record["nd"] is not None and abs(nd-record["nd"]) > result["nd_absolute_warning_threshold"]:
        result["warnings"].append("NM_nd_and_dispersion_coefficients_disagree")
    if abs(nf-nc) > 1.e-14:
        vd = (nd-1.)/(nf-nc)
        result["computed_vd"] = vd
        if record["vd"] is not None and abs(vd-record["vd"]) > result["vd_absolute_warning_threshold"]:
            result["warnings"].append("NM_vd_and_dispersion_coefficients_disagree")
    return result


def _paths(sources):
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    paths = []
    for source in sources:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            paths.extend(p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() == ".agf")
        elif path.is_file() and path.suffix.lower() == ".agf":
            paths.append(path)
        else:
            raise ValueError(f"Expected an AGF file or local catalog directory: {path}")
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No local AGF files found")
    return paths


def import_catalogs(sources, index_path, *, vendor=None, metadata=None):
    """Create a NEW immutable index and content-addressed local snapshots.

    metadata maps absolute input paths or filenames to {vendor, catalog_name,
    version_label, release_date, official_source_url, verified_sha256, ...}.
    A renamed or custom catalog requires explicit vendor identity.  Exact
    canonical vendor stems may identify a vendor, but never identify a release.
    Existing index files are refused: select a new index filename for updates.
    """
    index_path = Path(index_path).expanduser().resolve()
    if index_path.exists():
        raise FileExistsError(f"Index already exists; import updates into a new index file: {index_path}")
    metadata = metadata or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Import metadata must map input filenames to metadata objects")
    entries, snapshots, current_catalogs = [], {}, {}
    imported = _now()
    for path in _paths(sources):
        detail = dict(metadata.get(str(path), metadata.get(path.name, {})))
        owner = detail.get("vendor") or vendor or KNOWN_VENDOR_STEMS.get(path.stem.upper())
        if not owner:
            raise ValueError(f"Explicit vendor required for {path.name}; filenames do not prove manufacturer or release")
        owner = _identifier(owner, "vendor")
        catalog_name = _catalog_label(detail.get("catalog_name", path.stem))
        raw = path.read_bytes()
        digest = _sha(raw)
        records, headers = _record_metadata(raw)
        parsed = _read_catalog(_CatalogBytes(raw, path.name), set(records))
        if set(parsed) != set(records):
            raise ValueError("Metadata and exact dispersion parser disagree on material identities")
        catalog_id = f"{owner}::{catalog_name}::{digest}"
        glasses = []
        for name, record in records.items():
            glass = parsed[name]
            reasons = []
            if glass.formula not in SUPPORTED_FORMULAS:
                reasons.append("unsupported_dispersion_formula")
            if len(glass.coefficients) < 6:
                reasons.append("insufficient_dispersion_coefficients")
            if len(glass.coefficients) > 6 and any(glass.coefficients[6:]):
                reasons.append("nonzero_extra_coefficients_not_evaluated_by_current_tracer")
            if not glass.bounds_um:
                reasons.append("missing_wavelength_validity")
            consistency = _consistency(glass, record)
            glasses.append({**record, "vendor": owner, "catalog_name": catalog_name,
                "catalog_id": catalog_id, "snapshot_sha256": digest,
                "coefficient_count": len(glass.coefficients),
                "wavelength_bounds_um": list(glass.bounds_um) or None,
                "reference_temperature_c": glass.reference_temperature_c,
                "reference_temperature_basis": ("AGF_TD" if record["reference_temperature_declared"]
                                                else "existing_parser_default_not_catalog_declared"),
                "data_consistency": consistency,
                "selection": {"eligible_for_nominal_index_preparation": not reasons and not consistency["warnings"],
                    "suitability_verified": False,
                    "warning": "Eligibility covers dispersion data only, not transmission, fluorescence, availability or manufacturing."},
                "dispersion": {"formula": glass.formula,
                    "formula_name": SUPPORTED_FORMULAS.get(glass.formula),
                    "ready_for_current_tracer": not reasons, "blocked_reasons": reasons,
                    "thermal_model": "reference-temperature dispersion only",
                    "index_basis": "AGF formula as supplied; no independent air, temperature or pressure correction"}})
        if detail.get("default_current") is True:
            if owner in current_catalogs and current_catalogs[owner] != catalog_id:
                raise ValueError(f"More than one explicit default-current catalog for {owner}")
            current_catalogs[owner] = catalog_id
        entry = {"catalog_id": catalog_id, "vendor": owner, "catalog_name": catalog_name,
            "scope": detail.get("scope", "unknown"),
            "snapshot_sha256": digest, "snapshot_path": f"catalog-snapshots/{digest}.agf",
            "source_path": str(path), "source_filename": path.name, "byte_count": len(raw),
            "imported_at_utc": imported, "header_comments": headers,
            "release": _release_metadata(detail, digest), "materials": glasses}
        # Identical file imports do not cause duplicate lookup candidates.
        if not any(e["catalog_id"] == catalog_id for e in entries):
            entries.append(entry)
        snapshots[digest] = raw
    index = {"schema": SCHEMA, "created_at_utc": imported,
        "mode": "offline_user_supplied_immutable_snapshots", "catalogs": entries,
        "selection_registry": {"current_catalogs": current_catalogs,
            "basis": "explicit default_current import metadata, never filename guessing",
            "query_behavior": "current_only must be explicitly requested; otherwise ambiguity is an error"},
        "notes": ["No catalogs are downloaded or updated automatically.",
                  "nd/Vd are descriptive metadata; exact evaluation uses snapshotted AGF coefficients.",
                  "LD is dispersion validity, not optical transmission.",
                  "Catalog status/nd/Vd do not establish procurement, 355 nm transparency, fluorescence or manufacturing suitability."]}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    directory = index_path.parent / "catalog-snapshots"
    _contained(index_path.parent, directory)
    directory.mkdir(exist_ok=True)
    for digest, raw in snapshots.items():
        target = _contained(index_path.parent, directory / f"{digest}.agf")
        try:
            with target.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if _sha(target.read_bytes()) != digest:
                raise ValueError(f"Existing content-addressed snapshot was modified: {target}")
    # Exclusive creation prevents concurrent imports from silently replacing an index.
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return GlassCatalogIndex(index_path)


class GlassCatalogIndex:
    """Explicit vendor/name/release lookup; no cross-vendor first-match fallback."""

    def __init__(self, index_path):
        self.path = Path(index_path).expanduser().resolve()
        self.document = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.document.get("schema") != SCHEMA:
            raise ValueError("Unsupported local glass catalog index schema")
        self.catalogs = self.document["catalogs"]
        self._parsed = {}

    @property
    def current_catalogs(self):
        return dict(self.document.get("selection_registry", {}).get("current_catalogs", {}))

    def list(self, *, vendor=None, catalog_id=None, current_only=False):
        return self.search(vendor=vendor, catalog_id=catalog_id, current_only=current_only)

    def search(self, text=None, *, vendor=None, name=None, catalog_id=None, catalog_name=None,
               nd_min=None, nd_max=None, vd_min=None, vd_max=None,
               wavelength_um=None, ready_only=False, status=None, current_only=False,
               selection_eligible_only=False):
        owner = _identifier(vendor, "vendor") if vendor else None
        wavelength = _wavelength(wavelength_um) if wavelength_um is not None else None
        for lower, upper in ((nd_min, nd_max), (vd_min, vd_max)):
            if any(value is not None and not math.isfinite(float(value)) for value in (lower, upper)):
                raise ValueError("nd/Vd search limits must be finite")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("nd/Vd minimum search limit exceeds maximum")
        result = []
        for catalog in self.catalogs:
            if current_only and self.current_catalogs.get(catalog["vendor"]) != catalog["catalog_id"]:
                continue
            if owner and catalog["vendor"] != owner:
                continue
            if catalog_id and catalog["catalog_id"] != catalog_id:
                continue
            if catalog_name and catalog["catalog_name"] != str(catalog_name).upper():
                continue
            for glass in catalog["materials"]:
                if name and glass["name"] != str(name).upper():
                    continue
                if text and str(text).casefold() not in (glass["name"]+" "+" ".join(glass["comments"])).casefold():
                    continue
                if any(limit is not None and (glass[key] is None or
                       (glass[key] < limit if minimum else glass[key] > limit))
                       for key, limit, minimum in (("nd", nd_min, True), ("nd", nd_max, False),
                                                   ("vd", vd_min, True), ("vd", vd_max, False))):
                    continue
                bounds = glass["wavelength_bounds_um"]
                if wavelength is not None and (not bounds or not bounds[0] <= wavelength <= bounds[1]):
                    continue
                if ready_only and not glass["dispersion"]["ready_for_current_tracer"]:
                    continue
                if selection_eligible_only and not glass["selection"]["eligible_for_nominal_index_preparation"]:
                    continue
                if status is not None and str(status).lower() not in (
                        str(glass["status"]["code"]), str(glass["status"]["label"])):
                    continue
                row = copy.deepcopy(glass)
                row["release"] = copy.deepcopy(catalog["release"])
                result.append(row)
        return sorted(result, key=lambda row: (row["vendor"], row["name"], row["catalog_id"]))

    def lookup(self, name, *, vendor=None, catalog_id=None, catalog_name=None, current_only=False):
        matches = self.search(name=name, vendor=vendor, catalog_id=catalog_id,
                              catalog_name=catalog_name, current_only=current_only)
        if not matches:
            raise KeyError(f"Material {vendor or '*'}::{name} not found in this index")
        if len(matches) != 1:
            identities = [row["catalog_id"] for row in matches]
            raise ValueError(f"Ambiguous material {name}; specify vendor and/or exact catalog_id: {identities}")
        return matches[0]

    def _glass(self, record):
        catalog_id = record["catalog_id"]
        if catalog_id not in self._parsed:
            catalog = next(c for c in self.catalogs if c["catalog_id"] == catalog_id)
            path = (self.path.parent / catalog["snapshot_path"]).resolve()
            try:
                path.relative_to(self.path.parent)
            except ValueError as error:
                raise ValueError("Snapshot path escapes the local index directory") from error
            raw = path.read_bytes()
            if _sha(raw) != catalog["snapshot_sha256"]:
                raise ValueError("Immutable catalog snapshot SHA-256 mismatch")
            self._parsed[catalog_id] = _read_catalog(_CatalogBytes(raw, path.name),
                                                   {r["name"] for r in catalog["materials"]})
        return self._parsed[catalog_id][record["name"]]

    def refractive_index(self, name, wavelength_um, *, vendor=None, catalog_id=None,
                         catalog_name=None, current_only=False, allow_flagged_data=False):
        wavelength = _wavelength(wavelength_um)
        record = self.lookup(name, vendor=vendor, catalog_id=catalog_id,
                             catalog_name=catalog_name, current_only=current_only)
        if not record["dispersion"]["ready_for_current_tracer"]:
            raise ValueError("Material is not ready for exact current-tracer dispersion: " +
                             ", ".join(record["dispersion"]["blocked_reasons"]))
        if record["data_consistency"]["warnings"] and not allow_flagged_data:
            raise ValueError("Catalog data-consistency warning blocks refractive-index use; "
                             "explicit allow_flagged_data=True is for source investigation: "+
                             ", ".join(record["data_consistency"]["warnings"]))
        # Range checking is repeated here and inside the exact parser; no extrapolation API.
        glass = self._glass(record)
        if not glass.bounds_um or not glass.bounds_um[0] <= wavelength <= glass.bounds_um[1]:
            raise ValueError(f"Wavelength {wavelength:g} um is outside known catalog validity")
        return glass.index(wavelength)

    def prepare_indices(self, model, selections=None, *, recompute_existing=False, allow_flagged_data=False):
        """Return (neutral model, provenance), preserving existing indices by default.

        selections maps existing solid medium indices to explicit identity dicts:
        {1: {"vendor": "SCHOTT", "name": "N-BK7", "catalog_id": "..."}}.
        Index i is the medium AFTER surface i, including object channel zero.
        Only selected channels change unless recompute_existing=True is explicit,
        or a new model has no index table. Duplicate spectral slots are retained.
        No nominal suitability, transmission or manufacturing acceptance is implied.
        """
        result = copy.deepcopy(model)
        names = list(result["material_names"])
        count = len(names)
        wavelengths = [_wavelength(w) for w in result["wavelengths_um"]]
        if count != len(result["vertices_mm"])+1 or not wavelengths:
            raise ValueError("Neutral geometry/material/spectral dimensions disagree")
        catalogs = list(result.get("material_catalogs", [None]*count))
        if len(catalogs) != count:
            raise ValueError("Neutral material catalog dimensions disagree")
        table = result.get("indices_by_wave")
        has_table = table is not None
        if has_table and (len(table) != len(wavelengths) or any(len(row) != count for row in table)
                          or any(not math.isfinite(float(n)) or float(n) <= 0 for row in table for n in row)):
            raise ValueError("Invalid existing neutral refractive-index table")
        if not has_table:
            table = [[1.]*count for _ in wavelengths]
        else:
            # Python callers can supply aliased rows even though JSON cannot.
            table = [list(row) for row in table]
        selections = selections or {}
        normalized = {}
        for index, identity in selections.items():
            medium = int(index)
            if medium != float(index) or not 0 <= medium < count or names[medium] is None:
                raise ValueError("Material selection must name an existing solid medium channel")
            if not isinstance(identity, Mapping) or not identity.get("name"):
                raise ValueError("Material selections require explicit identity objects with name")
            normalized[medium] = dict(identity)
        resolved, preserved = [], []
        for medium, name in enumerate(names):
            if name is None:
                continue
            if has_table and not recompute_existing and medium not in normalized:
                preserved.append(medium)
                continue
            identity = normalized.get(medium, {"name": name, "catalog_name": catalogs[medium]})
            allowed = {key: identity[key] for key in ("vendor", "catalog_id", "catalog_name", "current_only")
                       if identity.get(key)}
            record = self.lookup(identity["name"], **allowed)
            if record["data_consistency"]["warnings"] and not allow_flagged_data:
                raise ValueError("Catalog data-consistency warning blocks nominal material preparation: "+
                                 ", ".join(record["data_consistency"]["warnings"]))
            indices = [self.refractive_index(record["name"], w, catalog_id=record["catalog_id"],
                                            allow_flagged_data=allow_flagged_data)
                       for w in wavelengths]
            for wave, index in enumerate(indices):
                table[wave][medium] = index
            names[medium], catalogs[medium] = record["name"], record["catalog_name"]
            resolved.append({"medium_index": medium, "vendor": record["vendor"],
                "name": record["name"], "catalog_id": record["catalog_id"],
                "snapshot_sha256": record["snapshot_sha256"], "indices_by_wave": indices,
                "release": record["release"], "data_consistency": record["data_consistency"]})
        result.update(material_names=names, material_catalogs=catalogs, indices_by_wave=table)
        if resolved:
            result.pop("nominal_verification", None)
            result["catalog_preparation"] = {"status": "unverified_indices_prepared",
                "resolved_medium_indices": [row["medium_index"] for row in resolved],
                "new_optical_and_manufacturing_acceptance_required": True}
        provenance = {"schema": "standalone_optics.glass_indices.v1", "index_path": str(self.path),
            "index_sha256": _sha(self.path.read_bytes()), "resolved_media": resolved,
            "preserved_existing_medium_indices": preserved,
            "allow_flagged_data_explicitly_requested": bool(allow_flagged_data),
            "existing_prescriptions_updated_automatically": False,
            "suitability_or_optical_acceptance_verified": False}
        result["glass_catalog_provenance"] = provenance
        return result, provenance


def _wavelength(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.:
        raise ValueError("Wavelength must be finite and positive in micrometres")
    return value


def load_index(index_path):
    return GlassCatalogIndex(index_path)


def refractive_index(index, name, wavelength_um, **identity):
    return (index if isinstance(index, GlassCatalogIndex) else load_index(index)).refractive_index(
        name, wavelength_um, **identity)


def prepare_indices(index, model, selections=None, **options):
    return (index if isinstance(index, GlassCatalogIndex) else load_index(index)).prepare_indices(
        model, selections, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("import", help="explicit offline import into a NEW index file")
    create.add_argument("sources", nargs="+", type=Path)
    create.add_argument("--index", required=True, type=Path)
    create.add_argument("--vendor")
    create.add_argument("--metadata", type=Path, help="per-input-filename metadata JSON; no inferred release dates")
    for command in ("list", "search", "lookup", "index"):
        action = commands.add_parser(command)
        action.add_argument("--index", required=True, type=Path)
        action.add_argument("--vendor")
        action.add_argument("--catalog-id")
        action.add_argument("--catalog-name")
        action.add_argument("--current-only", action="store_true",
                            help="use only explicitly registered current catalog IDs, excluding historical/all catalogs")
        if command == "search":
            action.add_argument("text", nargs="?")
        if command in ("lookup", "index"):
            action.add_argument("name")
        else:
            action.add_argument("--name")
        if command == "index":
            action.add_argument("--wavelength", required=True, type=float, help="catalog formula wavelength in um")
            action.add_argument("--allow-flagged-data", action="store_true",
                                help="explicit source-data investigation override, never repairs catalog coefficients")
        else:
            action.add_argument("--wavelength", type=float)
            action.add_argument("--ready-only", action="store_true")
            action.add_argument("--eligible-only", action="store_true",
                                help="exclude unsupported dispersion and flagged coefficient/NM inconsistencies")
            action.add_argument("--status")
            for bound in ("nd-min", "nd-max", "vd-min", "vd-max"):
                action.add_argument("--"+bound, type=float)
    prepare = commands.add_parser("prepare", help="explicitly prepare selected neutral-model material channels")
    prepare.add_argument("--index", type=Path, required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--selections", type=Path, help="JSON medium-index to explicit identity mapping")
    prepare.add_argument("--recompute-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig")) if args.metadata else None
            catalog = import_catalogs(args.sources, args.index, vendor=args.vendor, metadata=metadata)
            output = {"index": str(catalog.path), "catalog_count": len(catalog.catalogs),
                      "material_count": len(catalog.list()), "mode": "offline_no_automatic_updates"}
        else:
            catalog = load_index(args.index)
            if args.command == "prepare":
                model = json.loads(args.model.read_text(encoding="utf-8-sig"))
                selections = json.loads(args.selections.read_text(encoding="utf-8-sig")) if args.selections else None
                prepared, provenance = catalog.prepare_indices(model, selections,
                    recompute_existing=args.recompute_existing)
                if args.out.resolve() == args.model.resolve():
                    raise ValueError("Use a separate output path to preserve the input prescription")
                with args.out.open("x", encoding="utf-8") as stream:
                    json.dump(prepared, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                output = {"output": str(args.out), "provenance": provenance}
            elif args.command == "index":
                identity = dict(vendor=args.vendor, catalog_id=args.catalog_id,
                                catalog_name=args.catalog_name, current_only=args.current_only)
                output = {"material": catalog.lookup(args.name, **identity),
                          "wavelength_um": args.wavelength,
                          "refractive_index": catalog.refractive_index(args.name, args.wavelength,
                              allow_flagged_data=args.allow_flagged_data, **identity),
                          "allow_flagged_data_explicitly_requested": args.allow_flagged_data}
            else:
                output = catalog.search(getattr(args, "text", None), vendor=args.vendor, name=args.name,
                    catalog_id=args.catalog_id, catalog_name=args.catalog_name,
                    wavelength_um=args.wavelength, ready_only=args.ready_only, status=args.status,
                    nd_min=args.nd_min, nd_max=args.nd_max, vd_min=args.vd_min, vd_max=args.vd_max,
                    current_only=args.current_only, selection_eligible_only=args.eligible_only)
                if args.command == "lookup":
                    # Preserve strict ambiguity handling even with additional metadata filters.
                    exact = catalog.lookup(args.name, vendor=args.vendor,
                        catalog_id=args.catalog_id, catalog_name=args.catalog_name, current_only=args.current_only)
                    output = exact if output else None
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"glass_catalog: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
