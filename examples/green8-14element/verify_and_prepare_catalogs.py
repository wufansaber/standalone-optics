"""Verify the frozen case; optionally copy the owner's exact OHARA catalog.

This checks bytes and catalog identity. It does not perform optical validation.
Requires Python 3.10+, standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(case_dir: Path) -> dict:
    case_dir = case_dir.resolve()
    manifest = json.loads((case_dir / "CASE_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        target = (case_dir / relative).resolve()
        if not target.is_relative_to(case_dir):
            raise ValueError(f"Manifest path escapes case directory: {relative}")
        if not target.is_file():
            raise ValueError(f"Missing case file: {relative}")
        if target.stat().st_size != expected["bytes"] or digest(target) != expected["sha256"]:
            raise ValueError(f"Case file hash/size mismatch: {relative}")
    return manifest


def prepare_catalog(case_dir: Path, supplied: Path, expected_sha: str) -> Path:
    supplied = supplied.resolve()
    if not supplied.is_file() or digest(supplied) != expected_sha:
        raise ValueError("OHARA catalog SHA256 differs from the frozen case; no file was copied")
    target = case_dir.resolve() / "Zemax" / "GLASSCAT" / "OH9AD9D2.AGF"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected_sha:
            raise ValueError("Existing destination catalog differs; refusing to overwrite it")
    elif target != supplied:
        shutil.copyfile(supplied, target)
    if digest(target) != expected_sha:
        raise ValueError("Copied catalog failed SHA256 verification")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Path(__file__).resolve().parent)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--ohara-path", type=Path)
    args = parser.parse_args()
    manifest = verify(args.case_dir)
    result = {"case_files_verified": True, "file_count": len(manifest["files"]),
              "source_model_sha256": manifest["source_model_sha256"],
              "optical_validation_performed": False}
    if args.ohara_path is not None:
        catalog = prepare_catalog(args.case_dir, args.ohara_path,
                                  manifest["required_OHARA_catalog_sha256"])
        result["catalog_prepared"] = str(catalog)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
