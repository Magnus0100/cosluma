# Current behavior checks

Production code and workers live in `scripts/`. This directory contains offline
behavior checks and synthetic fixtures; Windows packages exclude them, while the
source archive includes them for development.

From the repository root, using the pinned desktop Python environment:

```powershell
.\.venv\Scripts\python.exe .\tests\run.py core
.\.venv\Scripts\python.exe .\tests\run.py reference_appearance_stack boundary_light_coordination_worker
```

`core` runs the current routine CI checks. `all` runs the full component suite, which
is appropriate when changing shared imports or test organization. Model-specific
checks require the installed analysis runtime, selected with `--model-python`:

```powershell
.\.venv\Scripts\python.exe .\tests\run.py all --model-python "<analysis Python executable>"
```

Model checks execute CPU fixtures; they do not require new
image generation or imply artwork acceptance.

The runner supplies the production import path and disables bytecode writes.
UI checks use temporary workspaces so they do not leave fixture projects in the
checkout. Use `scripts/check_repository.py` for the source/resource/license gate;
do not repeat its syntax validation with `compileall`.

`public_source` runs in `core` and checks ignored/tracked private files, credential
formats, private home paths, snapshot tampering and public version consistency.
Optional private regional fixtures use `COSPLAY_REGIONAL_REGRESSION_DIRS` (an
OS path-separated directory list); unconfigured fixtures are skipped, and an
explicitly configured missing fixture fails.

Release preparation additionally runs `desktop_dependency_gate` and
`release_artifacts` for altered dependency/license bytes and damaged or unsafe
archive members. The actual assembled release is verified by
`scripts/verify_windows_release.py`; component checks do not replace that check.
