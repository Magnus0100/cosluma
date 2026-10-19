# Agent instructions

This file applies to the whole repository. Read it before changing files, then
use the English [Agent development guide](docs/AGENT_GUIDE.md) for setup, code
navigation, task examples and troubleshooting. Follow the developer's current
request and preserve unrelated work already present in the checkout.

## Project in one minute

CosLuma is a Windows x64, Python 3.12 photo-finishing application with a
PySide6 desktop, a local HTTP backend and a CLI. Production Python modules live
in `scripts/`; this is not a conventional `src/` package. Install the pinned
dependencies from `scripts/requirements-desktop.txt`, not with `pip install -e .`.
Model workers use separate registered Python environments.
The public name is CosLuma; existing `COSPLAY_*` variables, schemas, executable
filenames and user-data locations remain compatibility identifiers.

All production routes share this flow:

```text
desktop / CLI / local HTTP
  -> creative_task: check inputs, intent, capabilities and generation budget
  -> production_coordinator: validate the checked execution contract
  -> single_source_workflow: prepare assets, render and save
  -> optional post-completion work, review and export
```

The source owns subject geometry and alpha. Reference images guide appearance
and scene design; BeautyGRPO supplies explicitly selected generated face
appearance. Generated beauty texture must not be described as camera detail.
The canvas represents the current production pipeline; it is not a generic
ComfyUI plugin executor or freely wired DAG.

## Start each task

1. Inspect `git status --short` and the diff before editing. Keep existing user
   changes intact. Confirm the actual workspace and interpreter; never copy a
   maintainer's machine paths into code or documentation.
2. Read only the relevant modules and nearby `tests/*_smoke.py` checks. Use the
   guide's ownership table rather than loading the entire UI or workflow file.
3. Reproduce the behavior with an isolated temporary workspace or synthetic
   fixture. Use an explicit `--root` for CLI/backend/UI commands so experiments
   do not write a project into the checkout or a user's real workspace.
4. Implement the smallest complete change through the existing production
   path. Update affected documentation, resource lists and behavior checks.

## Contracts to preserve

- Product execution goes through `task-plan` / `task-start` or
  `/api/tasks/plan` / `/api/tasks/start`. Internal `one-click` requires the current
  checked task; do not introduce a second pipeline that bypasses it.
- `local_portrait`, `existing_reference` and `create_reference` select inputs
  and enabled nodes in one pipeline. Planning must not perform image generation.
  Existing-reference tasks have zero new-image budget by default; an explicitly
  checked regional task has its own bounded budget. Retries do not grant more.
- Keep `replay_frozen` for saved-result replay distinct from `prepare_current`
  for explicit new preparation. Reuse requires the current input, producer,
  parameter and runtime identities. A graph signature alone does not prove a
  pixel cache hit. Do not make a stale receipt current by changing metadata.
- Preserve source alpha, coordinate systems, color domains, premultiplied
  increments and layer ownership. Do not apply beauty, lighting, display
  transforms or reference effects twice.
- Save complete child revisions atomically. Failure, cancellation and late
  callbacks must preserve the parent and current edits; incomplete work must
  not become the latest saved result. Do not inherit human approval.
- Distinguish configuration, connectivity, model loading, execution, saved
  pixels, automated review and human acceptance. Report missing evidence as
  missing; successful CPU fixtures do not prove artistic quality.

## Validation and delivery

Run commands from the repository root with the configured desktop interpreter:

```powershell
git diff --check
.\.venv\Scripts\python.exe -B .\scripts\check_repository.py
.\.venv\Scripts\python.exe -B .\tests\run.py core
```

Use `tests/run.py` for selected behavior checks; it sets production imports,
offscreen Qt and bytecode policy. `core` is the routine offline CI suite. Run it
for code changes before handing off; for documentation-only work, verify links,
commands and the source gate. Run relevant model/frozen-package checks when the
change requires them. Do not run all model checks or generate images merely to
validate wording. See the guide for the test selection table.

When adding a shipped production module or worker, update the sorted
`scripts/runtime_manifest.json`. When changing public root files, update
`scripts/prepare_source_release.py`'s `TOP_LEVEL` and the required-file gate as
appropriate. Preserve byte-sensitive files covered by `.gitattributes`,
especially license texts and identity-bound producers.

Keep photos, project outputs, weights, runtimes, credentials and local model
registries out of commits and source exports. Preserve the existing license
inventories. For a public repository use a fresh `prepare_source_release.py
--for-repository` export; omit that flag for a frozen release-source snapshot. Local
development history may contain private records. See
[open-source preparation](docs/OPEN_SOURCE_READINESS.md).

In the handoff, state what changed, checks actually run, their results, and any
remaining limitation. If an environment or model is unavailable, name it and
report the unrun check. Update this file or the guide when entry points,
contracts or developer commands change.
