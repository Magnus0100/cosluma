"""Assemble complete Windows and clean-source ZIPs with SHA-256 manifests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from check_repository import check
from verify_desktop_dependencies import VerificationError
from verify_windows_release import software_version, verify


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def archive(folder, target, files=None):
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in sorted(files if files is not None else folder.rglob('*')):
            if not path.is_file() or any(part in {'.git', '__pycache__'} for part in path.relative_to(folder).parts):
                continue
            output.write(path, (Path(folder.name)/path.relative_to(folder)).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dist', type=Path, required=True, help='The complete cosplay-workbench folder')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, dist, output = (p.resolve() for p in (args.source, args.dist, args.output))
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(dist):
        parser.error('Output must be a new directory outside source and dist')
    check(source)
    version = software_version(source)
    if not (source/'SOURCE_MANIFEST.json').is_file():
        parser.error('Use a verified clean source export with SOURCE_MANIFEST.json')
    if not (dist/'cosplay-workbench.exe').is_file() or not (dist/'_internal').is_dir():
        parser.error('Dist must contain the complete one-folder application')
    runtime = json.loads((source/'scripts/runtime_manifest.json').read_text('utf8'))
    for name in runtime['files']:
        if digest(dist/'_internal'/name) != digest(source/'scripts'/name):
            raise VerificationError('Stale runtime source: ' + name)
    if digest(dist/'_internal/LICENSES/manifest.json') != digest(source/'LICENSES/manifest.json'):
        raise VerificationError('Packaged license inventory differs from source')
    manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text('utf8'))
    output.mkdir(parents=True)
    package = output/f'cosplay-workbench-{version}-windows-x64'
    shutil.copytree(dist, package)
    for name in ('LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md',
                 '.env.example', 'Start-Cosplay-Studio.cmd'):
        shutil.copy2(source/name, package/name)
    windows_readme = (source/'docs/WINDOWS_README.md').read_text('utf8')
    for name in ('INSTALLATION.md', 'BOUNDARY_LIGHT.md', 'PORTABLE_EDITABLE_PROJECT.md', 'THIRD_PARTY_NOTICES.md'):
        windows_readme = windows_readme.replace('](' + name + ')', '](docs/' + name + ')')
    windows_readme = windows_readme.replace('](../LICENSE)', '](LICENSE)')
    (package/'README.md').write_text(windows_readme, encoding='utf8')
    shutil.copytree(source/'LICENSES', package/'LICENSES')
    shutil.copytree(source/'docs', package/'docs')
    (package/'scripts').mkdir()
    shutil.copy2(source/'scripts/start_studio.ps1', package/'scripts/start_studio.ps1')
    (package/'providers').mkdir()
    for name in ('local-models.example.json', 'qwen-image-2.1-models.example.json'):
        shutil.copy2(source/'providers'/name, package/'providers'/name)
    shutil.copytree(source/'providers/patches', package/'providers/patches')
    records = [{'path': p.relative_to(package).as_posix(), 'size': p.stat().st_size, 'sha256': digest(p)}
               for p in sorted(package.rglob('*')) if p.is_file()]
    (package/'PACKAGE_MANIFEST.json').write_text(json.dumps({'schema': 'cosplay_windows_release.v1',
        'version': version, 'architecture': 'x64', 'models_included': False,
        'files': records}, indent=2)+'\n', encoding='utf8')
    windows_zip = output/f'cosplay-workbench-{version}-windows-x64.zip'
    source_zip = output/f'cosplay-workbench-{version}-source.zip'
    archive(package, windows_zip)
    archive(source, source_zip, [source/e['path'] for e in manifest['files']] + [source/'SOURCE_MANIFEST.json'])
    archives = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in (windows_zip, source_zip)]
    (output/'SHA256SUMS.txt').write_text(''.join(f"{e['sha256']}  {e['file']}\n" for e in archives), encoding='ascii')
    (output/'release-manifest.json').write_text(json.dumps({'schema': 'cosplay_release.v1',
        'version': version, 'license': 'GPL-3.0-or-later', 'archives': archives,
        'runtime_modules': len(runtime['files']), 'private_git_history_included': False,
        'model_weights_included': False}, indent=2)+'\n', encoding='utf8')
    changelog = (source/'CHANGELOG.md').read_text('utf8')
    marker = '## ' + version + ' — '
    if marker not in changelog:
        raise VerificationError('Current version is missing from changelog')
    changes = changelog.split(marker, 1)[1].split('\n## ', 1)[0]
    (output/'RELEASE_NOTES.md').write_text('# Cosplay Studio ' + version + '\n\n' +
        changes.split('\n', 1)[1].strip() + '\n\n' +
        '下载 Windows x64 ZIP 并完整解压，保留 `_internal`；使用说明见包内 README.md。\n'
        '同版本 source.zip 提供对应源码，SHA256SUMS.txt 提供两份下载的 SHA-256。\n'
        '本版本不含模型权重、模型运行时、私人照片或凭据。\n'
        '清理及发布验证不代表新的成片效果验收。\n', encoding='utf8')
    validation = verify(source, output)
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'version': version, 'directory': str(output), 'archives': archives}))


if __name__ == '__main__':
    main()
