# CosLuma

[![CI](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

CosLuma is a Windows desktop workbench for source-faithful cosplay photo finishing. It keeps the source subject's structure and texture while preparing editable appearance, lighting, background and effect layers. The project includes a desktop UI, CLI and local HTTP backend.

The current source version is **0.5.46**. Windows EXE packages are built locally; publication to GitHub is a separate release step.

One production pipeline handles source → imported/generated reference → Beauty → scene/native T/B/S → boundary → save. GPT Image, local ComfyUI/Qwen Image and compatible providers remain selectable upstream. Reference creation defaults to one image; multiple candidates are explicit. Source-only work disables the reference and scene nodes in the same pipeline. External text/vision planning and review are optional and default off.

## Install and run

- Use the packaged Windows application when a release package is available.
- For source development, use Windows x64 and Python 3.12. Follow the [installation guide](docs/INSTALLATION.md).
- Local model weights, model runtimes, photos, projects and credentials are installed or configured by each user. They are never committed to this repository.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-desktop.txt
.\.venv\Scripts\python.exe .\scripts\project_cli.py init "$env:USERPROFILE\CosplayWorkbench\workspace"
.\.venv\Scripts\python.exe .\scripts\qt_workbench.py --root "$env:USERPROFILE\CosplayWorkbench\workspace" --embedded-backend --auto-start-backend
```

Open **Model Management** from the main window. Put installed models in the [model folder](model/README.md) beside the EXE, or select an existing installation directory. Standard folders and relative configuration paths are supported; saved selections survive restart. Individual paths and JSON import are available under advanced settings. Read the [ComfyUI architecture notes](docs/COMFYUI_ARCHITECTURE.md) and [repository layout](docs/REPOSITORY_LAYOUT.md) for implementation scope.

The professional canvas and task plans share typed node contracts and checked layer dependencies. Original subject alpha and generated beauty appearance remain separate. Status refreshes preserve selection, layout and zoom. Connections use the current production preset; a generic DAG executor and editable wiring are future work.

The desktop launcher attempts Beauty prewarming and reports success only after a completed loading receipt. Qwen configuration, service connectivity and model loading are separate states. Tasks show measured stage and elapsed time; an ETA requires at least three compatible successful runs, otherwise it remains pending. AI review, execution success and file export do not constitute the user's final human approval.

The compiled layer plan now also carries ComfyUI-style upstream input signatures. A beauty strength change invalidates BeautyGRPO and downstream appearance, lighting, composite and export work while allowing independent source analysis, reference and background results to be reused. Pixel bytes remain validated by the existing frozen-asset workers.

## Documentation and development

- [Agent instructions](AGENTS.md) and [Agent development guide](docs/AGENT_GUIDE.md) (English): setup, architecture, module ownership and verification.
- [Documentation index](docs/README.md)
- [Development and Windows build](docs/DEVELOPMENT.md)
- [Contributing](CONTRIBUTING.md)
- [Support](SUPPORT.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Open-source preparation](docs/OPEN_SOURCE_READINESS.md) (Chinese): export a clean source tree before creating public Git history.

Run the repository checks from a Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_repository.py
.\.venv\Scripts\python.exe .\tests\run.py core
```

## License

The project code is licensed under [GPL-3.0-or-later](LICENSE). Third-party dependencies, provider integrations and model code or weights retain their own licenses; see [third-party notices](docs/THIRD_PARTY_NOTICES.md) and [binary licenses](docs/BINARY_LICENSES.md).
