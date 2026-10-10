# Binary licenses and corresponding source

The application is distributed under GPL-3.0-or-later; see the root `LICENSE`. Its dependencies retain their own licenses. The Windows distribution is a one-folder package: keep the executable, `_internal`, `LICENSES`, notices and source information together.

The application source archive must accompany the Windows release and contain the Python sources, provider workflow/node/patch sources, public configuration templates, requirements, Windows specs and build scripts. It contains no user images, API credentials, local model paths, model weights or installed runtimes.

## Components in the desktop package

| Component | Version | License and preserved text |
|---|---|---|
| Python | 3.12.14 | PSF and included third-party terms; `LICENSES/Python-3.12.14/LICENSE.txt` |
| PySide6 / shiboken6 | 6.11.2 | LGPL-3.0 or the applicable GPL alternative; tagged upstream texts in `LICENSES/PySide6-6.11.2/` and `LICENSES/pyside-setup-6.11.2/` |
| Qt Core / Gui / Widgets | 6.11.2 | LGPL-3.0 route; Qt Base notices in `LICENSES/Qt-6.11.2/qtbase/` |
| Qt SVG and image-format plugins | 6.11.2 | Applicable LGPL/GPL terms and third-party codec notices in `LICENSES/Qt-6.11.2/qtsvg/` and `qtimageformats/` |
| rawpy | 0.27.1 | MIT wrapper; exact wheel and source texts in `LICENSES/rawpy-0.27.1/` and `rawpy-0.27.1-source/` |
| LibRaw inside rawpy | 0.22.1 | LGPL-2.1 route; copyright, LGPL/CDDL originals and source contributor headers in `LICENSES/LibRaw-0.22.1/` |
| NumPy | 2.5.3 | BSD and bundled native notices in `LICENSES/numpy-2.5.3/` |
| SciPy | 1.18.1 | BSD and bundled native notices, including OpenBLAS/GCC runtime terms, in `LICENSES/scipy-1.18.1/` |
| Pillow | 12.3.0 | MIT-CMU and bundled codec notices in `LICENSES/Pillow-12.3.0/` |
| tifffile | 2026.9.20 | BSD-3-Clause in `LICENSES/tifffile-2026.9.20/` |
| ONNX Runtime | 1.30.0 | MIT and complete `ThirdPartyNotices.txt` in `LICENSES/onnxruntime-1.30.0/` |
| OpenSSL | 3.5.8 | Apache-2.0; `LICENSES/OpenSSL-3.5.8/LICENSE.txt` |
| Tcl/Tk, if present | Runtime bundled versions | Interpreter and frozen data notices in `LICENSES/Python-3.12.14/` and `LICENSES/Tcl-Tk/` |
| PyInstaller bootloader | 6.22.3 | GPL-2.0-or-later with its bootloader distribution exception; `LICENSES/PyInstaller-6.22.3/` |

`LICENSES/manifest.json` records the exact SHA-256 and origin of each preserved text. The generic GPL/LGPL texts supplement the component notices; they do not replace copyright notices or extend a commercial Qt license. No commercial Qt entitlement is claimed.

The rawpy wheel reports both GPL2 and GPL3 demosaic packs disabled. LibRaw includes additional BSD/MIT contributor code; its original copyright file and licensing headers are retained. A change to the rawpy/LibRaw build requires checking its actual enabled features and new notices.

The Qt PDF image plugin, virtual-keyboard plugin and their QML/Quick dependencies are optional and are excluded from the intended desktop release. If a future build includes `Qt6VirtualKeyboard.dll`, it introduces a GPL-3.0/commercial module. If it includes `Qt6Pdf.dll`, preserve the Qt PDF and PDFium/Chromium notices and matching source information. Validate the final DLL inventory rather than inferring licenses from the top-level PySide package name.

Microsoft Visual C++ runtime DLLs and Windows system libraries retain Microsoft's terms and are not relicensed by this project. See Microsoft's [Visual Studio redistribution guidance](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution). Redistribution must use eligible runtime files and comply with those terms.

## Source locations

The versions below match the collected notices. Publish equivalent access to these corresponding sources alongside the binary download and preserve their build scripts and upstream licenses. Maintain access for the period required by the applicable license. A downstream distributor who changes a library must provide the corresponding modified source and build information.

| Component | Matching upstream source |
|---|---|
| Qt Base 6.11.2 | [qtbase source archive](https://download.qt.io/official_releases/qt/6.11/6.11.2/submodules/qtbase-everywhere-src-6.11.2.tar.xz) |
| Qt SVG 6.11.2 | [qtsvg source archive](https://download.qt.io/official_releases/qt/6.11/6.11.2/submodules/qtsvg-everywhere-src-6.11.2.tar.xz) |
| Qt Image Formats 6.11.2 | [qtimageformats source archive](https://download.qt.io/official_releases/qt/6.11/6.11.2/submodules/qtimageformats-everywhere-src-6.11.2.tar.xz) |
| PySide / shiboken 6.11.2 | [pyside-setup source archive](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/pyside-setup-everywhere-src-6.11.2.tar.xz) |
| rawpy 0.27.1 | [release source](https://pypi.org/project/rawpy/0.27.1/#files), [build documentation](https://github.com/letmaik/rawpy#installation) |
| LibRaw 0.22.1 | [LibRaw source archive](https://www.libraw.org/data/LibRaw-0.22.1.tar.gz) |
| Python 3.12.14 | [CPython source](https://www.python.org/ftp/python/3.12.14/Python-3.12.14.tar.xz) |
| OpenSSL 3.5.8 | [OpenSSL source tag](https://github.com/openssl/openssl/tree/openssl-3.5.8) |
| NumPy / SciPy / Pillow / tifffile / ONNX Runtime | Their exact version source distributions and repositories identified in `LICENSES/dependency-sources.json` |
| PyInstaller 6.22.3 | [PyInstaller source](https://github.com/pyinstaller/pyinstaller/tree/v6.22.3) |

Qt's official [licensing information](https://doc.qt.io/qt-6/licensing.html) and [third-party attribution inventory](https://doc.qt.io/qt-6/licenses-used-in-qt.html) explain its module and codec terms. The preserved notices come from the matching 6.11.2 source archives, rather than a newer documentation version.

## Rebuild and replace libraries

Use Windows x64 and Python 3.12. Create an isolated environment and install the exact desktop requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts\requirements-desktop.txt
New-Item -ItemType Directory -Force ..\clean-smoke | Out-Null
.\scripts\package_windows.ps1 -PythonExecutable .\.venv\Scripts\python.exe -SmokeRoot ..\clean-smoke -DistPath ..\release-dist -WorkPath ..\release-build
```

The package script builds both GUI and debug executables and runs their offline native/Qt smoke. Additional synthetic contract checks and archive/privacy checks belong in CI before publishing. Model workers use separately installed, registered runtimes and are not part of this desktop dependency set.

To rebuild Qt, use the matching Qt Base sources, a supported Windows MSVC toolchain and the upstream CMake/configure instructions, selecting a shared open-source build. Build Qt SVG and Qt Image Formats against that same Qt installation. Build PySide/shiboken from the matching `pyside-setup` sources with the rebuilt Qt path and the upstream documented LLVM/toolchain requirements. Replace the matching DLLs, plugins and binding modules together, or rebuild the application package against that environment. See [Qt Windows build instructions](https://doc.qt.io/qt-6/windows-building.html) and [Qt for Python build instructions](https://doc.qt.io/qtforpython-6/building_from_source/index.html).

To rebuild LibRaw/rawpy, use the matching sources and rawpy's build scripts. rawpy can statically include LibRaw, so replacing a separate DLL alone may not replace the library: rebuild the rawpy extension or wheel, install it in the build environment, and package again. For example, `python -m pip wheel --no-binary=rawpy rawpy==0.27.1` uses the release source distribution and its build instructions. Retain the modified LibRaw/rawpy source and enabled-feature information.

The one-folder package permits replacement of shared libraries. Recipients may modify and reverse engineer the LGPL-covered portions to debug those modifications. This project imposes no additional restriction on that right. Keep ABI-compatible versions together; a rebuilt library can change application behavior.

## Provider and model boundary

The project includes its provider adapter and empty-canvas node source. ComfyUI itself is an independently installed GPL-3.0 program; its license is preserved for the node integration. The compatibility patch contains Apache-2.0 ComfyUI-GGUF context credited to City96; its original license and patch notice are in `LICENSES/`.

The background SR worker adapts Real-ESRGAN source under BSD-3-Clause, with its original copyright and disclaimer retained. The application's license does not relicense model weights. Qwen/community weights, DINOv3, SAM, Sapiens and the other optional models retain their respective model terms and must be obtained and accepted separately. They are excluded from the source and Windows archives. See `docs/THIRD_PARTY_NOTICES.md` for the model/code distinction.
