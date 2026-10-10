# Source and tools

`runtime_manifest.json` lists production modules and worker sources included in
Windows builds. Current checks and their fixtures live in `tests/`; run them with
`python tests/run.py core` or name the checks affected by a change. Desktop dependencies are pinned in
`requirements-desktop.txt`; model workers use separately configured runtimes.

Start with [development](../docs/DEVELOPMENT.md). Build Windows packages with
`package_windows.ps1`. Copy `providers/local-models.example.json` to a private
configuration, fill installed paths, and set `COSPLAY_MODEL_REGISTRY` before using
local analysis models. Machine configuration is excluded from releases.

The independent portrait beauty experiment uses `portrait_beauty.py prepare`
and `replay`, with separately installed SegFace / skin models. It is source-only
and is not part of the current EXE or reference workflow. See
[actual scope and usage](../docs/PORTRAIT_BEAUTY_POC.md).
