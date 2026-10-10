"""Verify release archives, public source and packaged file identities offline."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from check_repository import check, validate_public_path
from verify_desktop_dependencies import VerificationError, checked_relative


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def software_version(source):
    tree = ast.parse((source/'scripts/single_source_workflow.py').read_text('utf8'))
    version = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == 'SOFTWARE_VERSION' for target in node.targets))
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise VerificationError('Release version must use major.minor.patch')
    return version


def local_links(root):
    count = 0
    for document in root.rglob('*.md'):
        if any(part in {'_internal', 'LICENSES', '.git'} for part in document.relative_to(root).parts):
            continue
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', document.read_text('utf8')):
            target = target.strip().strip('<>').split('#', 1)[0]
            if not target or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            resolved = (document.parent/target).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise VerificationError('Missing local document link: ' + document.relative_to(root).as_posix())
            count += 1
    return count


def validate_package_path(name, preserved_licenses):
    if name in preserved_licenses:
        return
    path = Path(name)
    if (any(part in {'.git', '.cache', '__pycache__', 'tests', 'outputs', 'workspace'} for part in path.parts)
            or path.name in {'project.json', '.env', '.env.local', 'local-models.current.json',
                'qwen-image-2.1-models.json'} or path.suffix.lower() in
                {'.pyc', '.pyo', '.pt', '.pth', '.onnx', '.gguf', '.safetensors', '.cosplayproj'}
            or path.name.endswith(('_smoke.py', '_fixture.py'))):
        raise VerificationError('Private data, model or development check in Windows package: ' + name)


def verify_archive(path, folder, expected):
    """Compare every archive member to its separately verified directory bytes."""
    prefix = folder.name + '/'
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise VerificationError('Duplicate ZIP entry: ' + path.name)
        members = {}
        for entry in archive.infolist():
            relative = PurePosixPath(entry.filename)
            if (entry.is_dir() or entry.orig_filename != entry.filename or '\\' in entry.filename
                    or relative.is_absolute() or '..' in relative.parts):
                raise VerificationError('Invalid ZIP entry: ' + path.name)
            if not entry.filename.startswith(prefix):
                raise VerificationError('Unexpected ZIP root: ' + path.name)
            name = entry.filename[len(prefix):]
            if name not in expected:
                raise VerificationError('Unexpected ZIP file: ' + name)
            with archive.open(entry) as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if entry.file_size != expected[name]['size'] or checksum != expected[name]['sha256']:
                raise VerificationError('ZIP member differs from verified source: ' + name)
            members[name] = checksum
        if set(members) != set(expected):
            raise VerificationError('ZIP is missing verified files: ' + path.name)
    return len(members)


def verify(source, release):
    source, release = source.resolve(), release.resolve()
    gate = check(source)
    version = software_version(source)
    source_manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text('utf8'))
    source_records = {row['path']: row for row in source_manifest['files']}
    for name in source_records:
        validate_public_path(name)
    source_records['SOURCE_MANIFEST.json'] = {'size': (source/'SOURCE_MANIFEST.json').stat().st_size,
                                            'sha256': digest(source/'SOURCE_MANIFEST.json')}
    actual_source = {path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file()
                     and '.git' not in path.relative_to(source).parts}
    if actual_source != set(source_records):
        raise VerificationError('Clean source contains unlisted files')
    manifest = json.loads((release/'release-manifest.json').read_text('utf8'))
    if (manifest.get('schema') != 'cosplay_release.v1' or manifest.get('version') != version
            or manifest.get('license') != 'GPL-3.0-or-later'
            or manifest.get('private_git_history_included') is not False
            or manifest.get('model_weights_included') is not False):
        raise VerificationError('Release metadata does not match current public source')
    package = release/f'cosplay-workbench-{version}-windows-x64'
    if not (package/'cosplay-workbench.exe').is_file() or not (package/'_internal').is_dir():
        raise VerificationError('Windows release must contain the complete one-folder application')
    package_manifest = json.loads((package/'PACKAGE_MANIFEST.json').read_text('utf8'))
    if (package_manifest.get('schema') != 'cosplay_windows_release.v1'
            or package_manifest.get('version') != version or package_manifest.get('architecture') != 'x64'
            or package_manifest.get('models_included') is not False):
        raise VerificationError('Windows package metadata is invalid')
    license_records = json.loads((source/'LICENSES/manifest.json').read_text('utf8'))['texts']
    preserved_licenses = {prefix + row['target'] for row in license_records for prefix in ('', '_internal/')}
    package_records = {}
    for row in package_manifest['files']:
        name = checked_relative(row['path']).as_posix()
        path = package/name
        if name in package_records or path.is_symlink() or not path.resolve().is_relative_to(package):
            raise VerificationError('Duplicate or external package file')
        validate_package_path(name, preserved_licenses)
        if not path.is_file() or path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            raise VerificationError('Changed or missing package file: ' + name)
        package_records[name] = row
    package_records['PACKAGE_MANIFEST.json'] = {'size': (package/'PACKAGE_MANIFEST.json').stat().st_size,
                                               'sha256': digest(package/'PACKAGE_MANIFEST.json')}
    if {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()} != set(package_records):
        raise VerificationError('Windows package contains unlisted files')
    runtime = json.loads((source/'scripts/runtime_manifest.json').read_text('utf8'))['files']
    if manifest.get('runtime_modules') != len(runtime):
        raise VerificationError('Release runtime count differs from source')
    for name in runtime:
        if digest(package/'_internal'/name) != digest(source/'scripts'/name):
            raise VerificationError('Packaged runtime differs from source: ' + name)
    for name in ('LICENSE', 'LICENSES/manifest.json', 'LICENSES/dependency-sources.json'):
        if digest(package/name) != digest(source/name) or digest(package/'_internal'/name) != digest(source/name):
            raise VerificationError('Packaged license inventory differs from source: ' + name)
    for row in license_records:
        for prefix in ('', '_internal/'):
            if digest(package/(prefix + row['target'])) != row['sha256']:
                raise VerificationError('Packaged license text differs from preserved source: ' + row['target'])
    expected_archives = {f'cosplay-workbench-{version}-windows-x64.zip': (package, package_records),
                         f'cosplay-workbench-{version}-source.zip': (source, source_records)}
    rows = manifest['archives']
    if len(rows) != 2 or {row['file'] for row in rows} != set(expected_archives):
        raise VerificationError('Release requires exactly one Windows archive and matching source archive')
    archive_counts = {}
    for row in rows:
        path = release/row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise VerificationError('Archive checksum differs: ' + row['file'])
        archive_counts[row['file']] = verify_archive(path, *expected_archives[row['file']])
    checksum_lines = ''.join(f"{row['sha256']}  {row['file']}\n" for row in rows)
    if (release/'SHA256SUMS.txt').read_text('ascii') != checksum_lines:
        raise VerificationError('Published checksums differ from release manifest')
    if not (release/'RELEASE_NOTES.md').is_file():
        raise VerificationError('Release notes are missing')
    document_links = local_links(source) + local_links(package)
    return {'schema': 'cosplay_public_release_validation.v1', 'ok': True, 'version': version,
            'source_gate': gate, 'runtime_modules': len(runtime), 'archive_files': archive_counts,
            'local_document_links_checked': document_links,
            'archives_match_verified_directories': True, 'private_history_or_models_included': False,
            'artwork_acceptance_granted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.source, args.release)
    except (OSError, ValueError, VerificationError, zipfile.BadZipFile) as error:
        print(json.dumps({'ok': False, 'error': str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
