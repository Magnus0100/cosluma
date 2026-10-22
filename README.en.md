# CosLuma

[![CI](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosluma/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

CosLuma is a Windows desktop workbench for cosplay photo finishing. Retouching, partial compositing and full compositing share checked tasks, a desktop UI, CLI and local HTTP backend.

**Download 0.7.2:** [Windows x64 application ZIP](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/cosplay-workbench-0.7.2-windows-x64.zip) · [matching source ZIP](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/cosplay-workbench-0.7.2-source.zip) · [SHA-256 checksums](https://github.com/Magnus0100/cosluma/releases/download/v0.7.2/SHA256SUMS.txt).

Extract the complete Windows folder and open `cosplay-workbench.exe`. Model weights and inference runtimes are configured separately. Developers can clone this repository or use the source archive. This release distributes the standalone desktop application. Optional shaping remains experimental; naturalness across photos, high-resolution enhancement and series consistency have not completed product acceptance. Review each AI result before adopting it.

The current source version is **0.7.6**, with source-native initialization, photographic development and model-selected tools connected. Verify the installed application identity, actual model output and cross-image quality separately. Windows packages are built locally; publication is a separate release step.

AI creation observes the source, freezes a native float baseline, selects a photographic, local or joint editing tool, independently reviews actual pixels and conditionally improves them before saving. Design and review default on. Partial compositing keeps the original setting by default. Qwen and GPT receive model-specific instructions for the same goal. Generated content uses its actual visible and permitted coverage; untouched pixels retain native floats. Joint editing permits changes to both subject and setting. Generated detail is not camera detail. Cross-image artistic acceptance remains incomplete.

Source-only and ordinary imported-reference tools remain available through the same checked production entry. Removed asset formats and retired workflows have no compatibility or migration path.

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

Open **Model Management** from the main window. Put installed models in the [model folder](model/README.md) beside the EXE, or select an existing installation directory. Standard folders and relative configuration paths are supported; saved selections survive restart. Individual paths and JSON import are available under advanced settings. Read the [production architecture](docs/PRODUCT_ARCHITECTURE.md) and [repository layout](docs/REPOSITORY_LAYOUT.md) for implementation scope.

The professional canvas and task plans share typed node contracts and checked layer dependencies. Original subject alpha and generated beauty appearance remain separate. Status refreshes preserve selection, layout and zoom. Connections show the current production dependency graph; arbitrary editable wiring is unsupported.

After restoring saved settings, the desktop initializes the selected managed service; unused Beauty models are not loaded unconditionally. Qwen configuration, service connectivity and model loading are separate states. Tasks show measured stage and elapsed time; an ETA requires at least three compatible successful runs, otherwise it remains pending. AI review, execution success and file export do not constitute the user's final human approval.

The layer plan records upstream input signatures and invalidates affected downstream work. Actual reuse additionally verifies frozen asset content, producer and runtime identities; graph status alone does not prove a pixel cache hit.

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
