"""Installer transactions use CPU file fixtures; EXE resource parsing is isolated."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import windows_installation as installer


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf8')


def fixture(root):
    source, dist = root/'source', root/'dist'
    names = ['scripts/start_studio.ps1', 'scripts/windows_installation.py', 'Start-Cosplay-Studio.cmd',
        'cosplay-workbench.spec', 'cosplay-workbench-debug.spec',
        'providers/registry.json', 'providers/local-models.example.json',
        'providers/beautygrpo.example.json', 'providers/qwen-image-2.1-models.example.json']
    for name in names:
        write(source/name, 'CPU installer fixture '+name)
    write(source/'scripts/single_source_workflow.py', 'SOFTWARE_VERSION = "9.8.7"\n')
    write(source/'scripts/worker.py', 'print("CPU worker")\n')
    write(source/'scripts/runtime_manifest.json', json.dumps({'files': ['single_source_workflow.py', 'worker.py']}))
    write(source/'SOURCE_MANIFEST.json', json.dumps({'source_commit': 'a'*40}))
    for name in ('single_source_workflow.py', 'worker.py'):
        write(dist/'_internal'/name, (source/'scripts'/name).read_text(encoding='utf8'))
    for name in names:
        if name.startswith('providers/'):
            write(dist/'_internal'/name, (source/name).read_text(encoding='utf8'))
    write(dist/'_internal/build-identity.json', json.dumps(installer.build_identity(source)))
    write(dist/'cosplay-workbench.exe', 'CPU EXE resource fixture')
    return source, dist


def reject(call, message):
    try:
        call()
    except (OSError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Invalid installation was accepted')


def real_executable_reader(root, metadata):
    if os.name != 'nt':
        return False
    import PyInstaller
    from PyInstaller.utils.win32.versioninfo import (VSVersionInfo, FixedFileInfo,
        StringFileInfo, StringTable, StringStruct, write_version_info_to_executable)
    target = root/'resource-reader.exe'
    bootloader = Path(PyInstaller.__file__).parent/'bootloader/Windows-64bit-intel/runw.exe'
    shutil.copy2(bootloader, target)
    info = VSVersionInfo(ffi=FixedFileInfo(), kids=[StringFileInfo([StringTable('040904B0', [
        StringStruct(key, value) for key, value in metadata.items()])])])
    write_version_info_to_executable(str(target), info)
    assert installer._executable_metadata(target) == metadata
    return True


def main():
    with tempfile.TemporaryDirectory(prefix='windows-install-') as temporary:
        root = Path(temporary)
        source, dist = fixture(root)
        identity = installer.build_identity(source)
        metadata = {'SourceCommit': identity['source_commit'],
                    'BuildContentIdentity': installer.identity_digest(identity)}
        reader_verified = real_executable_reader(root, metadata)
        with patch.object(installer, '_executable_metadata', return_value=metadata):
            # The source-only installer is hashed in the build identity but
            # must not be required as an executable runtime module.
            assert not (dist/'_internal/windows_installation.py').exists()
            assert installer.verify_build(source, dist) == identity
            retired = dist/'_internal/retired.py'
            retired.write_text('retired program', encoding='utf8')
            reject(lambda: installer.verify_build(source, dist), 'inventory differs')
            retired.unlink()
            original = (dist/'_internal/worker.py').read_bytes()
            (dist/'_internal/worker.py').write_bytes(b'stale-runtime')
            reject(lambda: installer.verify_build(source, dist), 'Packaged runtime differs')
            (dist/'_internal/worker.py').write_bytes(original)
            resource = dist/'_internal/providers/beautygrpo.example.json'
            original = resource.read_bytes()
            resource.write_bytes(b'changed-resource')
            reject(lambda: installer.verify_build(source, dist), 'Packaged resource differs')
            resource.write_bytes(original)
            destination = root/'installed'
            manifest = installer.install(source, dist, destination)
            assert manifest['software_version'] == '9.8.7'
            assert manifest['build_identity_sha256'] == metadata['BuildContentIdentity']
            assert manifest['build_stamp_sha256'] == installer.digest(destination/'_internal/build-identity.json')
            write(destination/'model/user-config.json', 'preserved-model-config')
            write(destination/'runtimes/user-runtime', 'preserved-runtime')
            write(destination/'scripts/retired.py', 'obsolete-software')
            before = {name: (destination/name).read_bytes() for name in (
                '_internal/worker.py', 'cosplay-workbench.exe', 'scripts/start_studio.ps1',
                'scripts/retired.py', 'install-manifest.json', 'model/user-config.json')}
            actual_replace = installer.os.replace
            def fail_publish(source_path, target):
                if (Path(target) == destination/'scripts'
                        and Path(source_path).parent.name == 'new'):
                    raise PermissionError('fixture blocked publish')
                return actual_replace(source_path, target)
            with patch.object(installer.os, 'replace', side_effect=fail_publish):
                reject(lambda: installer.install(source, dist, destination), 'fixture blocked publish')
            assert all((destination/name).read_bytes() == payload for name, payload in before.items())
            assert not list(destination.glob('.install-transaction-*'))
            manifest = installer.install(source, dist, destination)
            assert not (destination/'scripts/retired.py').exists()
            assert (destination/'model/user-config.json').read_text() == 'preserved-model-config'
            assert (destination/'runtimes/user-runtime').read_text() == 'preserved-runtime'
            assert 'runtimes' in manifest['required_directories']
            assert not list(destination.glob('.install-transaction-*'))
            assert not list(destination.rglob('*.tmp'))
            reject(lambda: installer.install(source, dist, source.parent), 'separate from the source')
            unknown = root/'unknown'
            write(unknown/'keep.txt', 'unrelated-user-data')
            reject(lambda: installer.install(source, dist, unknown), 'unrecognized installation')
            assert (unknown/'keep.txt').read_text() == 'unrelated-user-data'
        with patch.object(installer, '_executable_metadata', return_value={**metadata, 'SourceCommit': 'b'*40}):
            reject(lambda: installer.verify_build(source, dist), 'Executable differs')
    print(json.dumps({'ok': True, 'embedded_identity_required': True,
        'real_PE_version_resource_reader_verified': reader_verified,
        'current_runtime_and_resource_hashes_checked': True, 'failed_publish_rolls_back': True,
        'model_runtime_preserved': True, 'retired_software_deleted': True,
        'scope': 'CPU file transactions and real PE resource reader; transaction EXE identities are fixtures'}))


if __name__ == '__main__':
    main()
