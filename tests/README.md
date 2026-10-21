# Current behavior checks

Production code and workers live in `scripts/`. This directory contains offline
behavior checks and synthetic fixtures; Windows packages exclude them, while the
source archive includes them for development.

From the repository root, using the pinned desktop Python environment:

```powershell
.\.venv\Scripts\python.exe .\tests\run.py core
.\.venv\Scripts\python.exe .\tests\run.py reference_appearance_stack boundary_light_coordination_worker
```

Choose the smallest set that proves the change:

- `core` is the short routine suite for task boundaries, pixel assets, persistent
  request guards, cancellation, publication and export. It does not include the
  repeated native/Qt end-to-end fixtures.
- Name affected checks while implementing. Use `--list` to inspect a selection
  without importing tests, creating a workspace or starting a model.
- `integration` contains the remaining offline regression groups.
- `full` runs `core` and `integration` once each. Use it before release or for a
  justified change across production boundaries; it is not a daily default.
- Model-dependent checks are named explicitly and require `--model-python`.
  The old `all` auto-discovery command is removed; adding a test file no longer
  silently expands an expensive or model-dependent suite.

```powershell
.\.venv\Scripts\python.exe -B .\tests\run.py full --list
.\.venv\Scripts\python.exe -B .\tests\run.py photo_shaping shaping_ui
.\.venv\Scripts\python.exe -B .\tests\run.py full
.\.venv\Scripts\python.exe -B .\tests\run.py sapiens_current --model-python "<analysis Python executable>"
```

Repeated explicit names run once. The runner reports each group's elapsed time
and stops on the first failure. Offline fixtures and actual model checks remain
separate from artistic quality and human acceptance.

Small Qwen prompt-only and grid-diagnostic test files were consolidated into
the real provider and quality-gate checks. Unknown periodicity still cannot
become a rejection without neighbor evidence. Profile tests share fixture
setup rather than inheriting another test class, so its cases do not run twice.

`photo_session_review` runs in `integration`; `vision_transport` remains in `core`. Regional
review checks use validated native edit coverage to compare the whole image
and the same local crop in one board. They check the actual encoded JPEG and
the `detailed_board` transport evidence; missing or standard-profile evidence
is rejected. Transport loopback HTTP fixtures do not contact external models.
These checks do not certify native detail or artistic quality.

`photo_shaping`, `photo_shaping_geometry`, `face_shaping_preflight`,
`shaping_portable` and `shaping_ui` cover the separately checked
opt-in modifier: actual U-to-A/U-to-B conditioning, exact zero-model restore,
owned portable pixels, explicit adoption, startup selection and cancellation.
Model responses remain synthetic; these are not naturalness or installation
acceptance. Active shaped results currently reject ordinary feedback editing.
Face apply additionally checks canonical geometry against a synthetic detector
boundary, including exact protected central pixels. Preflight reads only current
small files; restore consumes no localization model. Body RGBA edits retain their
premultiplied-alpha composition checks; face/body mixtures are not executed.
`face_contour_deform` checks fixed canonical mapping, complete protected pixels,
positive native Jacobians, exact zero, no-op rejection and RGB/G sampling against
an independent coordinate reference. `geometric_appearance` checks the owned U,
mapping/support closure, exact float/preview/G and coherent metadata tampering.
`appearance_proof_cache` checks actual-content and current-runtime audit on each
read, process-local mathematical proof reuse, concurrent validation, atomic
owned copies and rejection of coherent tampering or stale producers.
`appearance_read_scope` checks operation-limited Windows read ownership of
existing saved input files, fresh hashing after release and isolation of copied
or staging outputs. The facade check also verifies fresh directory inventories,
current reader bytes and atomic outer evidence publication while a scope lives.
Unsupported filesystems retain full byte reads; no serialized proof is admitted.
`shaping_inspection_ui` checks cold verification away from the GUI thread,
responsive controls, fresh adoption audits and preservation on stale or
cancelled callbacks. `saved_result_selection` also checks nonwaiting guarded
selection commits, without synchronously rebuilding all saved history.
Public shaping checks distinguish zero-image face deformation from body image
editing; face/body mixtures are blocked during planning. They do not establish
naturalness on real photographs. `reference_condition_inputs` uses loopback HTTP to verify the optional Qwen
source/mask conditioning pair, which is not itself an enforced edit mask.

`qwen_alpha_quality` checks that invisible RGB cannot change grid rejection,
opaque raster scores remain identical, visible periodicity is still rejected,
and evidence from black/white backgrounds cannot be combined into fake peaks.
The provider and request-budget fixtures also verify complete quality-rejected
responses remain charged known outcomes without refunds, automatic retries,
candidate registration or historical receipt rewrites.

The runner supplies the production import path and disables bytecode writes.
UI checks use temporary workspaces so they do not leave fixture projects in the
checkout. Use `scripts/check_repository.py` for the source/resource/license gate;
do not repeat its syntax validation with `compileall`.

`manual_target_budget` keeps the small concurrent/unknown-outcome guards in `core`; the native cross-revision quota scenarios stay in `photo_request_budget` under `integration`.

`public_source` runs in `core` and checks ignored/tracked private files, credential
formats, private home paths, snapshot tampering and public version consistency.
Optional private regional fixtures use `COSPLAY_REGIONAL_REGRESSION_DIRS` (an
OS path-separated directory list); unconfigured fixtures are skipped, and an
explicitly configured missing fixture fails.

Release preparation additionally runs `desktop_dependency_gate` and
`release_artifacts` for altered dependency/license bytes and damaged or unsafe
archive members. The actual assembled release is verified by
`scripts/verify_windows_release.py`; component checks do not replace that check.

`external_python_isolation` proves frozen Windows DLL isolation with a real
child process and a renamed copy of the desktop Python's VC runtime DLL in a
temporary bundle. It checks the leaking control, isolated child, restored parent,
failed launches, concurrent launches, `PATH` boundaries and waiting outside the
DLL lock. It runs offline in `integration`; no installed service or model is used.
