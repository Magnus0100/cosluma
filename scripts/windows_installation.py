"""Build identity and complete replacement of the desktop application payload."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':')).encode('utf8')).hexdigest()


def _executable_metadata(path):
    """Read the EXE's embedded build identity independently of external files."""
    import pefile
    with pefile.PE(str(path), fast_load=True) as executable:
        executable.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']])
        for group in getattr(executable, 'FileInfo', ()):
            for entry in group:
                if getattr(entry, 'Key', None) == b'StringFileInfo':
                    for table in entry.StringTable:
                        values = {key.decode('utf8'): value.decode('utf8')
                                  for key, value in table.entries.items()}
                        if 'BuildContentIdentity' in values and 'SourceCommit' in values:
                            return values
    raise ValueError('Executable has no embedded build identity; rebuild the complete application')


def build_identity(source):
    source = Path(source).resolve()
    if (source/'.git').exists():
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        commit = json.loads((source/'SOURCE_MANIFEST.json').read_text('utf8'))['source_commit']
    runtime = json.loads((source/'scripts/runtime_manifest.json').read_text('utf8'))
    names = ['scripts/'+name for name in runtime['files']]
    names += ['scripts/start_studio.ps1', 'Start-Cosplay-Studio.cmd',
              'scripts/runtime_manifest.json', 'scripts/windows_installation.py',
              'cosplay-workbench.spec', 'cosplay-workbench-debug.spec']
    names += [p.relative_to(source).as_posix() for folder in ('presets', 'providers/workflows', 'providers/comfyui')
              for p in (source/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    names += ['providers/registry.json', 'providers/qwen-image-2.1-models.example.json',
              'providers/local-models.example.json', 'providers/beautygrpo.example.json']
    return {'schema': 'cosplay_build_identity.v1', 'source_commit': commit,
            'files': {name: digest(source/name) for name in sorted(names)}}


def verify_build(source, package):
    source, package = Path(source).resolve(), Path(package).resolve()
    identity = json.loads((package/'_internal/build-identity.json').read_text('utf8'))
    expected = build_identity(source)
    if identity != expected:
        raise ValueError('Build does not match the current source commit/content; rebuild the complete application')
    embedded = _executable_metadata(package/'cosplay-workbench.exe')
    if (embedded['BuildContentIdentity'] != identity_digest(identity)
            or embedded['SourceCommit'] != identity['source_commit']):
        raise ValueError('Executable differs from its build identity; rebuild the complete application')
    runtime = json.loads((source/'scripts/runtime_manifest.json').read_text('utf8'))['files']
    shipped_sources = {path.name for path in (package/'_internal').glob('*.py') if path.is_file()}
    if shipped_sources != set(runtime):
        raise ValueError('Packaged runtime source inventory differs from the production allowlist')
    for name in runtime:
        if digest(package/'_internal'/name) != identity['files']['scripts/'+name]:
            raise ValueError('Packaged runtime differs from its build identity: '+name)
    for name, expected_sha in identity['files'].items():
        if name.startswith(('providers/', 'presets/')):
            if digest(package/'_internal'/name) != expected_sha:
                raise ValueError('Packaged resource differs from its build identity: '+name)
    return identity


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _software_files(package):
    for folder in ('_internal', 'scripts'):
        directory = package/folder
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(package):
                raise ValueError('Software payload must not contain external links: '+str(path))
            if path.is_file() and '__pycache__' not in path.parts:
                yield path
    for name in ('cosplay-workbench.exe', 'Start-Cosplay-Studio.cmd'):
        path = package/name
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError('Software payload must remain inside the installation: '+name)
        yield path


def write_manifest(source, package):
    package = Path(package).resolve()
    identity = verify_build(source, package)
    inventory = {path.relative_to(package).as_posix(): digest(path) for path in _software_files(package)}
    from verify_windows_release import software_version
    manifest = {'schema': 'cosplay_desktop_installation.v1',
        'source_commit': identity['source_commit'], 'software_version': software_version(Path(source)),
        'build_identity_sha256': identity_digest(identity),
        'build_stamp_sha256': digest(package/'_internal/build-identity.json'),
        'architecture': 'windows-x64', 'executable': 'cosplay-workbench.exe',
        'executable_sha256': inventory['cosplay-workbench.exe'],
        'required_directories': [name for name in ('_internal', 'model', 'models', 'runtimes')
                                 if (package/name).is_dir()],
        'source_checkout_required': False, 'prewarm': True,
        'external_model_configuration_required': True,
        'software_files_sha256': inventory}
    _atomic_json(package/'install-manifest.json', manifest)
    return manifest


def install(source, dist, destination):
    source, dist, destination = (Path(p).resolve() for p in (source, dist, destination))
    verify_build(source, dist)
    if (destination == source or destination.is_relative_to(source) or source.is_relative_to(destination)
            or dist == destination or dist.is_relative_to(destination) or destination.is_relative_to(dist)
            or (destination/'.git').exists()):
        raise ValueError('Installation must be separate from the source checkout')
    if destination.exists():
        try:
            installed = json.loads((destination/'install-manifest.json').read_text('utf8'))
            if installed.get('schema') != 'cosplay_desktop_installation.v1':
                raise ValueError('Unknown installation manifest')
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError('Refusing to replace an unrecognized installation directory') from error
    destination.mkdir(parents=True, exist_ok=True)
    from cache_store import exclusive_file_lock
    with exclusive_file_lock(destination/'.install.lock'):
        return _install_locked(source, dist, destination)


def _install_locked(source, dist, destination):
    transaction = destination/('.install-transaction-'+uuid.uuid4().hex)
    staged, previous = transaction/'new', transaction/'old'
    staged.mkdir(parents=True)
    previous.mkdir()
    actions = []
    created_model = False
    try:
        # Stage and verify every software byte before changing the runnable
        # installation. Model/configuration/runtime directories stay owned by
        # the existing installation and never enter this transaction.
        list(_software_files(dist))
        shutil.copytree(dist/'_internal', staged/'_internal')
        shutil.copy2(dist/'cosplay-workbench.exe', staged/'cosplay-workbench.exe')
        (staged/'scripts').mkdir()
        shutil.copy2(source/'scripts/start_studio.ps1', staged/'scripts/start_studio.ps1')
        shutil.copy2(source/'Start-Cosplay-Studio.cmd', staged/'Start-Cosplay-Studio.cmd')
        (staged/'model').mkdir()
        manifest = write_manifest(source, staged)
        manifest['required_directories'] = ['_internal', 'model'] + [
            name for name in ('models', 'runtimes') if (destination/name).is_dir()]
        _atomic_json(staged/'install-manifest.json', manifest)
        if not (destination/'model').exists():
            (destination/'model').mkdir()
            created_model = True
        for name in ('_internal', 'cosplay-workbench.exe', 'scripts',
                     'Start-Cosplay-Studio.cmd', 'install-manifest.json'):
            target, backup = destination/name, previous/name
            had_original = target.exists()
            if had_original:
                os.replace(target, backup)
            action = {'name': name, 'had_original': had_original, 'published': False}
            actions.append(action)
            os.replace(staged/name, target)
            action['published'] = True
    except Exception as error:
        rollback_errors = []
        for action in reversed(actions):
            target, backup = destination/action['name'], previous/action['name']
            try:
                if action['published']:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if action['had_original']:
                    os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        if created_model:
            try:
                (destination/'model').rmdir()
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError('Installation publish failed and rollback needs recovery at '+
                               str(transaction)+': '+'; '.join(rollback_errors)) from error
        if not transaction.resolve().is_relative_to(destination):
            raise RuntimeError('Installation transaction escaped its destination') from error
        shutil.rmtree(transaction)
        raise
    else:
        if not transaction.resolve().is_relative_to(destination):
            raise RuntimeError('Installation transaction escaped its destination')
        shutil.rmtree(transaction)
        return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = install(args.source, args.dist, args.destination)
    print(json.dumps({'source_commit': result['source_commit'],
                      'verified_software_files': len(result['software_files_sha256']),
                      'destination': str(args.destination)}))
