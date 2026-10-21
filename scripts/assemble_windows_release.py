"""Assemble complete Windows and clean-source ZIPs with SHA-256 manifests."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from check_repository import check
from verify_desktop_dependencies import VerificationError
from verify_windows_release import software_version, verify
from windows_installation import verify_build, write_manifest


WINDOWS_USER_DOCS = ('INSTALLATION.md', 'WINDOWS_README.md', 'PRESETS.md',
    'QWEN_IMAGE_LOCAL.md', 'QWEN_PROMPTING.md', 'PORTABLE_EDITABLE_PROJECT.md',
    'PORTRAIT_BEAUTY.md', 'BOUNDARY_LIGHT.md', 'SOURCE_ALPHA.md', 'SOURCE_SCENE_ENHANCEMENT.md',
    'COSPLAY_POST_STANDARD.md', 'THIRD_PARTY_NOTICES.md', 'BINARY_LICENSES.md')


def copy_windows_documentation(source, package, runtime, version):
    """Keep contributor material in source.zip and preserve usable user links."""
    (package/'docs').mkdir()
    for name in WINDOWS_USER_DOCS:
        shutil.copy2(source/'docs'/name, package/'docs'/name)
    (package/'SOURCE_CODE.md').write_text(
        '# Matching source archive\n\n'
        f'Download `cosplay-workbench-{version}-source.zip` from the same release. '
        'It contains the complete source, tests, build tools and contributor documentation.\n\n'
        'Start with `AGENTS.md` and `docs/AGENT_GUIDE.md` in that archive for agent-assisted '
        'development, or `docs/DEVELOPMENT.md` for build instructions. Developer-only links '
        'in this Windows package point here because those files belong to the source archive.\n\n'
        'Production Python modules shipped with the executable are in `_internal/`. '
        'See [binary licenses](docs/BINARY_LICENSES.md) and [LICENSE](LICENSE).\n', encoding='utf8')
    for document in package.rglob('*.md'):
        if any(part in {'_internal', 'LICENSES'} for part in document.relative_to(package).parts):
            continue
        content = document.read_text('utf8')
        if document == package/'README.md':
            for public_doc in (source/'docs').glob('*.md'):
                content = content.replace('](' + public_doc.name + ')', '](docs/' + public_doc.name + ')')
        for name in runtime:
            content = content.replace('](../scripts/' + name + ')', '](../_internal/' + name + ')')

        def link(match):
            target = match[2].strip().strip('<>').split('#', 1)[0]
            if not target or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                return match[0]
            resolved = (document.parent/target).resolve()
            if resolved.is_relative_to(package.resolve()) and resolved.is_file():
                return match[0]
            if (not resolved.is_relative_to(package.resolve())
                    or not (source/resolved.relative_to(package.resolve())).is_file()):
                raise VerificationError('Unknown local documentation link: ' + document.relative_to(package).as_posix())
            fallback = os.path.relpath(package/'SOURCE_CODE.md', document.parent).replace('\\', '/')
            return match[1] + fallback + match[3]

        document.write_text(re.sub(r'(\[[^\]\n]+\]\()([^)\n]+)(\))', link, content), encoding='utf8')


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
    verify_build(source, dist)
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
    for name in ('LICENSE', 'SECURITY.md', 'CHANGELOG.md',
                 '.env.example', 'Start-Cosplay-Studio.cmd'):
        shutil.copy2(source/name, package/name)
    windows_readme = (source/'docs/WINDOWS_README.md').read_text('utf8')
    windows_readme = windows_readme.replace('](../LICENSE)', '](LICENSE)')
    windows_readme = windows_readme.replace('](../model/README.md)', '](model/README.md)')
    (package/'README.md').write_text(windows_readme, encoding='utf8')
    shutil.copytree(source/'LICENSES', package/'LICENSES')
    (package/'scripts').mkdir()
    shutil.copy2(source/'scripts/start_studio.ps1', package/'scripts/start_studio.ps1')
    (package/'providers').mkdir()
    for name in ('local-models.example.json', 'qwen-image-2.1-models.example.json',
                 'beautygrpo.example.json'):
        shutil.copy2(source/'providers'/name, package/'providers'/name)
    shutil.copytree(source/'providers/patches', package/'providers/patches')
    (package/'model').mkdir()
    shutil.copy2(source/'model/README.md', package/'model/README.md')
    copy_windows_documentation(source, package, runtime['files'], version)
    write_manifest(source, package)
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
    (output/'RELEASE_NOTES.md').write_text('# CosLuma ' + version + '\n\n' +
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
