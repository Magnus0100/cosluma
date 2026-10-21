# Agent development guide

This English guide lets a coding agent take a developer from a fresh checkout
to a focused, verifiable change. Start with the root [AGENTS.md](../AGENTS.md).
Use [README.en.md](../README.en.md) for the product overview. Most detailed
design documents currently use Chinese; the module and command names below
are the actual English identifiers used by the implementation.

CosLuma is the public project name. Existing `COSPLAY_*` environment variables,
`cosplay_*` schemas, `cosplay-workbench` executable/launcher filenames and
`CosplayWorkbench` user-data locations are compatibility identifiers. Use the
commands and paths below literally; a branding update does not migrate projects.

For a first coding session, give the agent this request and replace the final
line with the intended change:

```text
Read AGENTS.md and docs/AGENT_GUIDE.md. Inspect existing changes and reuse or
establish the pinned desktop environment. Use an isolated workspace. Trace
the relevant production entry point and nearby tests, implement the change,
then run the required checks. Preserve unrelated edits and report results
and remaining limitations. Do not use real photos or external services just
to establish the development baseline.
Change: <describe the desired behavior and a concrete example>
```

## 1. Establish the checkout and environment

For ongoing product iteration, start with [the product goal](PRODUCT_GOAL.md),
[the engineering state](ENGINEERING_STATE.json) and [the AI harness loop](AI_HARNESS.md).
Run `<desktop-python> -B scripts/engineering_context.py` for the active task,
targeted source locations and current evidence. Pass `--observations` only for
a private engineering directory outside the checkout. The existing coding
agent makes decisions and calls tools; this helper neither runs production
nor converts technical checks into artistic acceptance.

Use Windows x64 and Python 3.12. The pinned desktop environment includes Qt,
CPU image processing and build dependencies. GPU inference environments and
model weights are separate installations; they are unnecessary for routine
offline `core` checks.

From the repository root in PowerShell:

```powershell
git status --short
git diff --stat
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-desktop.txt
.\.venv\Scripts\python.exe -B .\scripts\check_repository.py
.\.venv\Scripts\python.exe -B .\scripts\project_cli.py --help
```

Reuse an existing project environment when available; do not recreate it on
every task. An alternate desktop interpreter must satisfy the same pins. The
build script also checks `.cache/desktop-runtime/Scripts/python.exe`; do not
assume that optional path exists. `pyproject.toml` supplies metadata and tool
configuration, not an editable-install workflow. Keep model dependencies out
of this environment.

Before building a package, verify the actual interpreter and license bytes:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\verify_desktop_dependencies.py --root .
```

## 2. Get a working baseline without models

Create an isolated workspace instead of initializing the source directory:

```powershell
$workspace = Join-Path $env:TEMP ('cosplay-dev-' + [guid]::NewGuid().ToString('N'))
$env:COSPLAY_CONFIG_ROOT = Join-Path $workspace 'config'
$env:COSPLAY_CACHE_ROOT = Join-Path $workspace 'outputs\cache'
.\.venv\Scripts\python.exe -B .\scripts\project_cli.py init "$workspace"
.\.venv\Scripts\python.exe -B .\scripts\project_cli.py --root "$workspace" status
.\.venv\Scripts\python.exe -B .\scripts\project_cli.py --root "$workspace" recovery
.\.venv\Scripts\python.exe -B .\tests\run.py public_source creative_task production_coordinator
```

These checks use synthetic inputs and do not submit external generation jobs.
The workspace has no photos or configured analysis models, so production
preflight may correctly report missing capabilities. Treat that as setup
information, not permission to bypass model checks or set a template to ready.

For an interactive development session:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\qt_workbench.py --root "$workspace" --embedded-backend --auto-start-backend
```

This starts the local backend and initializes the selected managed Qwen
service after saved settings are restored. It does not load unused Beauty or
native analysis models. Service readiness is not model-weight warming; the
first image task may still load weights. Explicit portrait operations retain
their existing on-demand Beauty loading and persistent reuse. Unused model
setup must not override a ready scoped generation plan.
It is an interactive command, not the automated test command. For backend-only
work, run `backend_server.py --root "$workspace" --host 127.0.0.1 --port 8787
--max-workers 1`. A separate UI can connect using `--backend-url
http://127.0.0.1:8787`. Pass the same workspace explicitly in both shells.
The backend has no public-network authentication layer; its normal scope is
localhost.

## 3. Find the code that owns the change

Production code is under `scripts/`. Tests often import sibling production
modules directly; use the test runner to supply their import path. Tests named
after a behavior do not necessarily have a production file of the same name.

| Change area | Starting files | Useful behavior checks |
|---|---|---|
| Desktop controls, async callbacks, simple/professional views | [qt_workbench.py](../scripts/qt_workbench.py) | `creative_task_ui`, `same_photo_finish_ui`, `cos_workspace_integration` |
| CLI and local HTTP task dispatch | [project_cli.py](../scripts/project_cli.py), [backend_server.py](../scripts/backend_server.py) | `task_start_dispatch`, `embedded_backend`, `backend_product_state` |
| Task intent, capability preflight, generation budget | [creative_task.py](../scripts/creative_task.py), [creative_direction.py](../scripts/creative_direction.py) | `creative_task`, `task_model_preflight`, `creative_direction` |
| Scoped design, generated appearance and conditional completion | [photo_editing.py](../scripts/photo_editing.py), [photo_art_direction.py](../scripts/photo_art_direction.py), [photo_edit_finish.py](../scripts/photo_edit_finish.py), [photo_request_budget.py](../scripts/photo_request_budget.py) | `photo_editing`, `photo_edit_autofinish`, `photo_request_budget`, `autofinish_ui_graph` |
| Independently authorized shaping, same-parent recomputation and exact restore | [photo_shaping.py](../scripts/photo_shaping.py), [face_contour_deform.py](../scripts/face_contour_deform.py), [geometric_appearance.py](../scripts/geometric_appearance.py), [appearance_assets.py](../scripts/appearance_assets.py), [appearance_read_scope.py](../scripts/appearance_read_scope.py), [shaping_inspection.py](../scripts/shaping_inspection.py), [shaping_quality.py](../scripts/shaping_quality.py), [shaping contract](REVERSIBLE_SHAPING.md) | `photo_shaping`, `photo_shaping_geometry`, `face_contour_deform`, `geometric_appearance`, `appearance_read_scope`, `appearance_proof_cache`, `face_shaping_preflight`, `shaping_portable`, `shaping_ui`, `shaping_inspection_ui`, `shaping_quality`, `saved_result_selection` |
| Checked execution and result publication | [production_coordinator.py](../scripts/production_coordinator.py), [single_source_workflow.py](../scripts/single_source_workflow.py) | `production_coordinator`, `result_publication`, `saved_task_replanning` |
| Canvas nodes, dependencies, invalidation | [cos_node_registry.py](../scripts/cos_node_registry.py), [cos_workflow_graph.py](../scripts/cos_workflow_graph.py), [cos_workflow_cache.py](../scripts/cos_workflow_cache.py) | `cos_node_registry`, `cos_workflow`, `workflow_producer_identity` |
| Scoped graph, saved product state and measured progress | [scoped_workflow.py](../scripts/scoped_workflow.py), [scoped_product_state.py](../scripts/scoped_product_state.py), [job_source_progress.py](../scripts/job_source_progress.py) | `scoped_workflow`, `scoped_product_state`, `editing_iteration_graph` |
| Reference preparation and native rendering | [reference_pipeline.py](../scripts/reference_pipeline.py), [design_reconstruction_pipeline.py](../scripts/design_reconstruction_pipeline.py), [design_reconstruction.py](../scripts/design_reconstruction.py) | `reference_pipeline`, `reference_appearance_stack`, `frozen_native_paths` |
| Portrait beauty and frozen layer replay | [beautygrpo_prepare.py](../scripts/beautygrpo_prepare.py), [beautygrpo_worker.py](../scripts/beautygrpo_worker.py), [portrait_beauty_layer.py](../scripts/portrait_beauty_layer.py) | `portrait_beauty_workbench`, `beauty_existing_ownership`, `beauty_reference_overlap` |
| Generated references and local Qwen lifecycle | [reference_provider.py](../scripts/reference_provider.py), [start_qwen_image.py](../scripts/start_qwen_image.py), [provider_settings.py](../scripts/provider_settings.py), `providers/comfyui/` | `qwen_provider`, `qwen_service_ownership`, `qwen_gpu_transition` |
| Model discovery, identity and installation configuration | [current_model_registry.py](../scripts/current_model_registry.py), [model_setup.py](../scripts/model_setup.py), [asset_identity.py](../scripts/asset_identity.py) | `model_discovery`, `model_setup`, `asset_identity`, `runtime_preflight` |
| Jobs, persistence, cancellation and restart | [job_store.py](../scripts/job_store.py), [backend_server.py](../scripts/backend_server.py) | `job_store_concurrency`, `job_owner`, `active_job_reentry`, `manifest_persistence` |
| Saved results, portable projects and pixel export | [portable_project.py](../scripts/portable_project.py), [export_artifact.py](../scripts/export_artifact.py), [pipeline_stage_report.py](../scripts/pipeline_stage_report.py) | `saved_result_selection`, `source_copy`, `export_artifact`, `artifact_receipts` |
| Public source, runtime resources and Windows distribution | [prepare_source_release.py](../scripts/prepare_source_release.py), [check_repository.py](../scripts/check_repository.py), [runtime_manifest.json](../scripts/runtime_manifest.json), [package_windows.ps1](../scripts/package_windows.ps1) | `public_source`, `desktop_dependency_gate`, `release_artifacts` |

Find a relevant function before opening a large file:

```powershell
rg -n 'def (plan_task|execution_request)' scripts/creative_task.py
rg -n 'def (task_plan|start_task)' scripts/backend_server.py
rg -n 'replay_frozen|prepare_current' scripts tests
rg --files tests | rg 'publication|cancel|reference|beauty'
```

## 4. Understand a task before modifying it

The common task router owns planning, checked execution and atomic publication.
Current scoped creation observes the source, freezes a source-native float
baseline, evaluates it, and lets the model select a local or joint editing tool.
It independently evaluates actual edits and conditionally improves them before
saving the best complete version. Untouched local coverage keeps native floats;
opaque whole-image generation owns the entire appearance. A transparent result
retains its actual generated coverage and parent contribution. Generated
appearance owns the actual subject/environment lighting; do not blanket-restore
source subject RGB over it. Source-only and ordinary imported-reference tools
retain their applicable native alpha, Beauty, T/B/S and boundary contracts.
Rendering, reopening and export consume the correctly bound frozen assets.

`creative_task.plan_task` turns inputs and intent into a checked plan.
`production_coordinator.checked_request` revalidates execution inputs, and
`execute_workflow` enters `single_source_workflow`. HTTP/desktop starts queue a
job by default; CLI `task-start` explicitly executes synchronously. Keep slow
preparation out of UI request handlers.

The three modes are node selections in one pipeline:

- `local_portrait`: source-only nodes, no reference/scene/boundary preparation;
  it may consume an existing explicitly prepared beauty layer.
- `existing_reference`: import a reference and run the checked post-processing
  pipeline; no new image submissions by default.
- `create_reference`: generate the checked scoped appearance and complete it
  within the same production path. One initial candidate is the default.

Scoped half/full composition separates work scope from style, keeps the source
setting by default, and requires actual visual design at execution. Independent
review defaults on, with `editing_autofinish={"maximum_improvements":2,"maximum_zero_image_finishes":1}`:
one initial model-selected operation, at most two conditional tool rounds and
one final round offering only admitted zero-image tools or stop. The bounds are
three image submissions and ten vision calls for design, a shared source-baseline
review and each actual decision/recheck. The final allowance never authorizes
generation; unknown or stopped decisions consume their origin round. A qualified
CPU photographic integration tool uses no new image submission; its working
input is not automatically the best. An initial edited double
pass uses one image and four vision calls; a passed source baseline is never an edited result. Review off still performs design, the initial decision and tool, without
QA or automatic improvements; the user may also disable automatic improvement separately.
An explicit maximum_improvements without maximum_zero_image_finishes grants
zero final rounds; preserve explicit one-round preferences and historical receipts.
Old origin limits cannot be increased on resume. Use origin
ledger reservations and confirmed totals for external usage, rather than local
session attempt counts; a cached review is not another paid vision call.
Use the existing session/tools/evaluator through `photo_edit_finish`, not a
second runner. Unknown outcomes require reconciliation before resubmission.

The manual `editing_iteration` route is a separately checked new task bound to
a committed parent and actual independent review; it is not a required second
click for default completion. An explicitly opted-in native regional task also
retains its bounded contract. Preserve origin budgets and evidence across
retry, resume, candidate selection and reopening; none grants fresh quota.

To inspect a real developer-provided request, use:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\project_cli.py --root "$workspace" task-plan --request-file "<request.json>"
```

The placeholder must be replaced with a private request file. Inspect the
returned `ok`, capability failures, `request` and `plan.plan_id`. For an
intended production execution, construct the start body from that returned
request plus its `plan_id`, then use `task-start --request-file
"<checked-execution.json>"`. It can run configured models and services. Do not
invent a plan ID, call internal `one-click` as a shortcut, or turn execution
into a model-free smoke check by weakening validation.

Saved versions bind source/reference hashes, parameters, producer code, model
and runtime identities, and frozen assets. `replay_frozen` reopens the saved
result; `prepare_current` performs explicitly requested current preparation.
The latter may reuse a genuinely matching cache, saves a separate revision
and does not overwrite parent pixels or inherit approval. Use existing schema
validation and publication helpers rather than repairing receipts in place.

Scoped generated results reopen without inference for viewing, comparison,
review and export. Old source-space pixel editors are unavailable for them;
their read-only graph must represent conditional execution honestly. A large
output canvas is not proof of native camera detail. CPU fixtures, real model
execution, cross-image artistic quality and standalone installed acceptance
are separate checks; consult the current iteration record before claiming any.

## 5. Keep installation resources separate from user data

| Content | How to handle it |
|---|---|
| Shipped configuration templates, workflows, presets, notices | Resolve through [installed_resources.py](../scripts/installed_resources.py); they belong to the installation, not the project workspace |
| Photos, `project.json`, runs, revisions and exports | Keep in the explicitly selected workspace; preserve parent versions and current selections |
| Model weights and model Python environments | Configure through Model Management or a private registry; public `*.example.json` files remain unconfigured |
| API credentials | Use configured environment variables or credential storage; public settings store variable names, not secret values |
| Rebuildable cache | Use `COSPLAY_CACHE_ROOT`; direct worker identity/RAW caches also support `COSPLAY_ASSET_IDENTITY_CACHE` and `COSPLAY_SOURCE_METADATA_CACHE` |

For actual analysis, start from
[local-models.example.json](../providers/local-models.example.json) and configure
a private registry through `COSPLAY_MODEL_REGISTRY`. Local Qwen also needs its
separate deployment and service setup. See [model layout](../model/README.md),
[installation](INSTALLATION.md) and [Qwen setup](QWEN_IMAGE_LOCAL.md).

The PowerShell launcher reads `.env` and `.env.local` as literal `KEY=VALUE`
data. Direct Python commands do not automatically load them; set required
variables in the launching shell. A configured catalog does not prove service
connectivity or model loading. Preserve service ownership when cancelling or
releasing GPU memory; do not stop an externally owned ComfyUI process.

Launch registered Python runtimes through `external_python_popen` or
`external_python_run` in [runtime_resources.py](../scripts/runtime_resources.py).
On frozen Windows desktops these remove bundle directories from the child
`PATH` and reset the inherited DLL directory only while creating the process,
then restore the parent's previous directory. Waiting for a worker never holds
the DLL launch lock. Keep external-worker working directories outside the
bundle's `_internal` directory, which Windows can also search for DLLs. Source
launches retain the existing subprocess behavior. This boundary is for external
model runtimes; bundled desktop helpers retain their own launch contract.
The managed Qwen launch policy binds this isolation to a new service launch;
older live services follow the existing verified-owner, idle-queue restart
path. A busy service keeps its original receipt and is never re-signed.

## 6. Choose verification that proves the change

```powershell
git diff --check
.\.venv\Scripts\python.exe -B .\scripts\check_repository.py
.\.venv\Scripts\python.exe -B .\tests\run.py core
```

Use the short `core` suite for routine delivery after targeted checks.
`integration` contains the wider offline regressions; `full` combines both for
release or a justified cross-cutting change. Pass suite names alone, or name
checks without `_smoke.py`, such as `tests/run.py creative_task result_publication`.
Use `tests/run.py full --list` to inspect membership without execution. The old
implicit `all` command is removed. Model checks are named explicitly and require
`--model-python`. Repeated explicit names run once.
The runner sets `PYTHONPATH`, offscreen Qt, thread limits and bytecode policy,
and reports each group's elapsed time. The source gate parses Python syntax;
no duplicate `compileall` is needed. The lists in [tests/run.py](../tests/run.py)
are authoritative; avoid hard-coded test counts in new documentation.

Common setup failures have specific remedies:

| Symptom | Next step |
|---|---|
| A test cannot import a module from `scripts/` | Run it through `tests/run.py` from the repository root |
| Missing PySide6 or a desktop dependency | Check which interpreter is running and install the pinned desktop requirements into that environment |
| Model preflight is blocked | Check the capabilities requested by this task and the selected private registry; model-free onboarding does not require enabling them |
| A source snapshot hash check fails after editing | Keep the original snapshot intact, develop in a separate worktree/copy without the generated snapshot manifest, then export a new candidate |
| UI/backend shows a different project | Confirm both commands use the same explicit `--root` and the intended backend endpoint |

| Change | Additional verification |
|---|---|
| Documentation only | Check links, commands and referenced paths; run the source gate. Keep real generation out of wording checks |
| UI, backend, jobs or task planning | Affected ownership-table checks, then short `core`; cover cancellation, stale callbacks and saved state. Use `full` only for release or justified cross-cutting changes |
| Pixel producers, coordinates or compositing | Relevant component checks plus current frozen native replay/save/export evidence; inspect real native pixels when evaluating image quality |
| Model worker imports or shared dependency changes | Inspect the full suite's prerequisites; select only affected model checks with `tests/run.py <check-name> --model-python "<analysis-python.exe>"` |
| Runtime closure, dependencies or distribution | `desktop_dependency_gate`, `release_artifacts`, dependency/license verification and actual Debug/Release build smoke |

Tests under `tests/` contain synthetic fixtures and are excluded from the
Windows runtime. Optional private regional regression fixtures are selected
with `COSPLAY_REGIONAL_REGRESSION_DIRS`; on Windows separate directories with
semicolons. Unconfigured fixtures skip explicitly; missing explicitly selected
fixtures fail. Never commit the underlying private images.

If a check fails, reproduce the first failure in isolation and establish
whether the patch, environment or a pre-existing issue caused it. Keep
assertions meaningful. Report unavailable interpreters/models and unrun checks
instead of claiming a pass. Execution, numerically correct replay, automated
diagnostics and human artwork approval are separate results.

## 7. Keep contributions and public exports reproducible

New production modules and external worker sources need an entry in the sorted
[runtime manifest](../scripts/runtime_manifest.json). Build-time resources also
need the relevant PyInstaller spec/resource entries. Public top-level files
need the source export allowlist; normal public `docs/*.md` files are discovered
by the exporter. Update linked onboarding instructions when commands change.

The software version comes from `SOFTWARE_VERSION` in
[single_source_workflow.py](../scripts/single_source_workflow.py). A version
change must keep `pyproject.toml`, both READMEs, `CITATION.cff` and the first
changelog entry consistent. Do not bump a version solely for an ordinary
documentation edit. Preserve license bytes and the identity-sensitive files
specified in [.gitattributes](../.gitattributes).

To build and smoke-check the Windows Debug and Release bundles with the pinned
desktop environment:

```powershell
.\scripts\package_windows.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
```

Distribute the complete one-folder bundle including `_internal`. Building an
EXE does not install analysis models or prove image quality; follow the release
guide for corresponding source archives and final artifact verification.

To seed an editable public repository, export to a new directory outside the
checkout. This mode excludes Git history and generated snapshot metadata:

```powershell
$repositoryExport = Join-Path $env:TEMP ('cosplay-public-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B .\scripts\prepare_source_release.py --for-repository --output "$repositoryExport"
```

For a release-source asset, use the default snapshot mode instead:

```powershell
$sourceExport = Join-Path $env:TEMP ('cosplay-source-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B .\scripts\prepare_source_release.py --output "$sourceExport"
```

Inspect the exported files and, for a release snapshot, `SOURCE_MANIFEST.json`. The export carries
current working-tree source, including eligible uncommitted changes, without
old Git history. Do not overwrite a candidate snapshot in place or manually
change its hashes to make a check pass; create a new export after edits. A
snapshot manifest describes immutable release input, not a live development
worktree. Use `--for-repository` for continued development in an editable export;
keep original release snapshots intact and export anew after edits.

Public delivery uses clean history because the original development history
may contain private records. Do not mirror old branches/tags into a public
repository. Follow [open-source preparation](OPEN_SOURCE_READINESS.md),
[release preparation](RELEASING.md), [third-party notices](THIRD_PARTY_NOTICES.md)
and [security guidance](../SECURITY.md) for actual distribution.

Finish the task with a short handoff: changed behavior and files, validation
commands and results, and any material remaining limitation. A future agent
should be able to reproduce the stated result without private machine paths
or unexplained local artifacts.
