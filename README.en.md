# Cosplay Studio

[![CI](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Magnus0100/cosplay/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/) [![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

Cosplay Studio is a Windows desktop workbench for source-faithful cosplay photo finishing. It keeps the source subject's structure and texture while preparing editable appearance, lighting, background and effect layers. The project includes a desktop UI, CLI and local HTTP backend.

The current source version is **0.4.8**. It is developed in a private GitHub repository and does not ship a public release yet.

## Install and run

- Use the packaged Windows application when a release package is available.
- For source development, use Windows x64 and Python 3.12. Follow the [installation guide](docs/INSTALLATION.md).
- Local model weights, model runtimes, photos, projects and credentials are installed or configured by each user. They are never committed to this repository.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-desktop.txt
.\.venv\Scripts\python.exe .\scripts\project_cli.py init "$env:LOCALAPPDATA\CosplayWorkbench\workspace"
.\.venv\Scripts\python.exe .\scripts\qt_workbench.py --root "$env:LOCALAPPDATA\CosplayWorkbench\workspace" --embedded-backend --auto-start-backend
```

Configure local models from `providers/local-models.example.json` and optional Qwen services from `providers/qwen-image-2.1-models.example.json`. Read the [repository layout](docs/REPOSITORY_LAYOUT.md) before adding files.

## Documentation and development

- [Documentation index](docs/README.md)
- [Development and Windows build](docs/DEVELOPMENT.md)
- [Contributing](CONTRIBUTING.md)
- [Support](SUPPORT.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

Run the repository checks from a Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_repository.py
.\.venv\Scripts\python.exe .\tests\run.py core
```

## License

The project code is licensed under [GPL-3.0-or-later](LICENSE). Third-party dependencies, provider integrations and model code or weights retain their own licenses; see [third-party notices](docs/THIRD_PARTY_NOTICES.md) and [binary licenses](docs/BINARY_LICENSES.md).
