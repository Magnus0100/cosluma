# Build with the configured Python: python -m PyInstaller CosLuma.spec
from pathlib import Path
import shutil
from PyInstaller.utils.win32.versioninfo import (FixedFileInfo,StringFileInfo,
    StringStruct,StringTable,VarFileInfo,VarStruct,VSVersionInfo)

root = Path(SPECPATH)
sources = root / 'scripts'
version = {}
exec((sources/'app_version.py').read_text(encoding='utf-8'),version)
analysis = Analysis(
    [str(sources/'qt_workbench.py')],
    pathex=[str(sources)],
    datas=[],
    hiddenimports=['backend_server','project_cli','rawpy','color_matcher'],
    excludes=['torch','torchvision','torchaudio','transformers','diffusers','cupy','cupyx','cupy_backends','cuda',
              'onnxruntime','triton','nvidia','matplotlib','pandas','IPython',
              'pytest','tkinter','PySide6.QtWebEngineCore','PySide6.QtWebEngineWidgets',
              'PySide6.QtQml','PySide6.QtQuick'],
    noarchive=False,
)
# This Qt build links to Windows' system ICU API. An unrelated ICU from
# another application on PATH has incompatible exports and must not shadow it.
analysis.binaries = [entry for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in {'icuuc.dll','icudt78.dll'}]
archive = PYZ(analysis.pure)
executable = EXE(archive,analysis.scripts,[],exclude_binaries=True,name='CosLuma',
    debug=False,strip=False,upx=False,console=False,
    version=VSVersionInfo(
        ffi=FixedFileInfo(filevers=tuple(map(int,version['VERSION'].split('.')))+(0,),
            prodvers=tuple(map(int,version['VERSION'].split('.')))+(0,),
            mask=0x3f,flags=0,OS=0x40004,fileType=1,subtype=0,date=(0,0)),
        kids=[StringFileInfo([StringTable('040904B0',[
            StringStruct('CompanyName','CosLuma contributors'),
            StringStruct('FileDescription','CosLuma photo finishing'),
            StringStruct('FileVersion',version['VERSION']),
            StringStruct('ProductName','CosLuma'),
            StringStruct('ProductVersion',version['VERSION']),
            StringStruct('OriginalFilename','CosLuma.exe'),
            StringStruct('LegalCopyright','GPL-3.0-or-later'),
        ])]),VarFileInfo([VarStruct('Translation',[1033,1200])])]),
)
package = COLLECT(executable,analysis.binaries,analysis.datas,strip=False,upx=False,name='CosLuma')
destination = Path(DISTPATH)/'CosLuma'
for name in ('README.md','AGENTS.md','LICENSE'):
    shutil.copyfile(root/name,destination/name)
for name in ('LICENSES','providers'):
    shutil.copytree(root/name,destination/name,dirs_exist_ok=True)
