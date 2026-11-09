# Manufacturing acceptance audit probes

These are synthetic coaxial lenses, not project prescriptions. They record the acceptance gaps found at commit `c9a46f168f6750b7187e4167405c2fd08386b30b`.

From the repository root, with the normal NumPy dependency installed, run:

```powershell
python examples/manufacturing-audit-20261109/reproduce.py
```

The script imports the repository's `scripts/ai_zemax` package and writes `observed.json` beside itself. The stored result is the observation at the audited baseline. These diagnostic probes are separate from CI regression tests: after a fix, a previously reproduced finding should change. The script does not load Zemax, use credentials, or read a private lens or literature database.

See [the review](../../references/manufacturing-audit-20261109.md) for limitations, literature and implementation priorities. Compare `module_sha256` when rerunning against a later checkout; `audited_commit` identifies the historical review baseline.
