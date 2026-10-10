# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import ast
import sysconfig
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.win32.versioninfo import (VSVersionInfo, FixedFileInfo,
    StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct)

# One version source for manifests, UI and Windows executable metadata.
_version_node = next(node for node in ast.parse(Path('scripts/single_source_workflow.py').read_text('utf8')).body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 'SOFTWARE_VERSION'
                         for target in node.targets))
_software_version = ast.literal_eval(_version_node.value)
_version_numbers = tuple(map(int, _software_version.split('-', 1)[0].split('.'))) + (0,)
_version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=_version_numbers, prodvers=_version_numbers,
        mask=0x3f, flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable('040904B0', [
        StringStruct('FileDescription', 'Cosplay Workbench Debug'),
        StringStruct('FileVersion', '.'.join(map(str, _version_numbers))),
        StringStruct('InternalName', 'cosplay-workbench-debug'),
        StringStruct('OriginalFilename', 'cosplay-workbench-debug.exe'),
        StringStruct('ProductName', 'Cosplay Workbench'),
        StringStruct('ProductVersion', _software_version),
    ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])

# Keep the application's NumPy ahead of optional ONNX dependency targets.
# Otherwise modulegraph picks target NumPy while hook-numpy collects the
# installed wheel's differently named OpenBLAS DLL, producing a mixed bundle.
_site_packages = Path(sysconfig.get_path('purelib'))
_native_binaries = collect_dynamic_libs('rawpy')
_ort_msvcp = _site_packages / 'PySide6' / 'MSVCP140_1.dll'
if _ort_msvcp.is_file():
    _native_binaries.append((str(_ort_msvcp), '.'))

hiddenimports = ['backend_server', 'project_cli', 'reference_provider', 'provider_settings',
                 'workbench_store', 'tifffile', 'numpy', 'rawpy', 'export_artifact',
                 'raw_support', 'image_helpers', 
                 'register_reference',
                 'source_alpha', 'onnxruntime', 'colour_space',
                 'automatic_foreground_effects', 'automatic_foreground_effects_review',
                 'foreground_effects_runtime', 'effect_conditioning', 'effect_layout',
                 'effect_code_identity', 'effect_sampling', 'installed_resources', 'system_post_completion',
]


# Keep physical runtime sources for cache identities and external workers.
# Developer checks and maintenance commands remain source-only. The manifest
# includes production imports, identity dependencies and worker entry points.
import json
_runtime_names = frozenset(json.loads(Path('scripts/runtime_manifest.json').read_text('utf8'))['files'])

def _source_only_script_names():
    return {p.name for p in Path('scripts').glob('*.py')} - _runtime_names

def _runtime_script_datas():
    missing = sorted(name for name in _runtime_names if not (Path('scripts')/name).is_file())
    if missing:
        raise RuntimeError('Missing current runtime source: '+', '.join(missing))
    return [(str(Path('scripts')/name), '.') for name in sorted(_runtime_names)]

a = Analysis(
    ['scripts/qt_workbench.py'],
    pathex=['scripts', str(_site_packages)],
    binaries=_native_binaries,
    datas=[('providers/local-models.example.json', 'providers'), ('providers/registry.json', 'providers'), ('providers/qwen-image-2.1-models.example.json', 'providers'),
           ('providers/workflows', 'providers/workflows'), ('providers/comfyui', 'providers/comfyui'), ('presets/presets.json', 'presets')]
          + _runtime_script_datas()
          + [('LICENSE', '.'), ('LICENSES', 'LICENSES'),
             ('docs/BINARY_LICENSES.md', 'docs'), ('docs/THIRD_PARTY_NOTICES.md', 'docs')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/frozen_native_paths.py'],
    # Exclude developer modules from PYZ as well as their physical sources.
    # A source-only guarded import must not reintroduce them via modulegraph.
    excludes=sorted(Path(name).stem for name in _source_only_script_names()),
    noarchive=False,
    optimize=0,
)
# Poppler's optional ICU DLLs collide with Qt's generic ICU names in a
# one-folder bundle.  They are not used by this GUI; excluding them lets
# QtCore resolve its own dependencies instead of loading Poppler's ABI.
_qt_conflicts = {"icuuc.dll", "icudt78.dll"}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in _qt_conflicts]
# Only Widgets and image codecs are used. Remove plugins that pull in optional
# VirtualKeyboard/Qml/Pdf/network components and their unused shared libraries.
_unused_qt = {
    'qtvirtualkeyboardplugin.dll', 'qt6virtualkeyboard.dll', 'qt6quick.dll',
    'qt6qml.dll', 'qt6qmlmeta.dll', 'qt6qmlmodels.dll', 'qt6qmlworkerscript.dll',
    'qt6opengl.dll', 'qt6network.dll', 'qtnetwork.pyd', 'qpdf.dll', 'qt6pdf.dll',
    'qtuiotouchplugin.dll', 'qnetworklistmanager.dll', 'qcertonlybackend.dll',
    'qopensslbackend.dll', 'qschannelbackend.dll',
}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in _unused_qt]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cosplay-workbench-debug',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=_version_info,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cosplay-workbench-debug',
)
