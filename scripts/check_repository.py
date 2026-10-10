"""Compile source and validate public runtime/resources without importing models."""
from __future__ import annotations
import ast
import argparse
import hashlib
import json
from pathlib import Path
import re

from prepare_source_release import public_files
from verify_desktop_dependencies import (VerificationError, canonical_name, checked_relative, read_pins,
                                         verify_license_inventory)


FORBIDDEN_SUFFIXES = {'.arw', '.nef', '.cr2', '.cr3', '.dng', '.jpg', '.jpeg', '.png', '.psd', '.psb',
                      '.pt', '.pth', '.onnx', '.gguf', '.safetensors', '.task', '.cosplayproj',
                      '.pyc', '.pyo', '.pyd', '.dll', '.exe', '.zip', '.7z'}
PRIVATE_FILENAMES = {'project.json', 'local-models.current.json', 'qwen-image-2.1-models.json',
                     '.env', '.env.local', 'DATASET.json', 'DATASET.md'}


def validate_public_path(value):
    relative = checked_relative(value)
    if any(part in {'.git', '.cache', '__pycache__', 'build', 'dist', 'workspace'} for part in relative.parts):
        raise VerificationError('Private state or runtime cache selected for public source')
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES or relative.name in PRIVATE_FILENAMES:
        raise VerificationError('Private asset, config or binary selected for public source')
    if relative.parts[0] == 'output' or relative.parts[0] == 'docs' and any(
            part in {'meetings', 'reviews'} for part in relative.parts[1:]):
        raise VerificationError('Private execution evidence selected for public source')
    if relative.parts[0] in {'data', 'outputs'} and relative.as_posix() not in {'data/README.md', 'outputs/README.md'}:
        raise VerificationError('Private project data selected for public source')
    return relative


def validate_source_manifest(root):
    path = root/'SOURCE_MANIFEST.json'
    if not path.is_file():
        return 0
    manifest = json.loads(path.read_text('utf8'))
    if manifest.get('schema') != 'cosplay_public_source.v1' or manifest.get('private_history_included') is not False:
        raise VerificationError('Public source snapshot contract is invalid')
    names = set()
    for row in manifest['files']:
        relative = validate_public_path(row['path'])
        name = relative.as_posix()
        if name in names:
            raise VerificationError('Duplicate public source snapshot record')
        names.add(name)
        target = root.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise VerificationError('Public source snapshot file is missing or escapes the export')
        data = target.read_bytes()
        if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise VerificationError('Public source snapshot file changed: ' + name)
    selected = {path.relative_to(root).as_posix() for path in public_files(root)}
    if names != selected:
        raise VerificationError('Public source snapshot does not match the current export allowlist')
    return len(names)


def check(root):
    paths = sorted(path for tree in ('scripts', 'tests', 'providers') for path in (root/tree).rglob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.relative_to(root).as_posix())
    manifest = json.loads((root/'scripts/runtime_manifest.json').read_text('utf8'))
    names = manifest['files']
    if len(names) != len(set(names)) or names != sorted(names):
        raise VerificationError('Runtime manifest must contain unique sorted source filenames')
    if not all(Path(name).name == name and (root/'scripts'/name).is_file() for name in names):
        raise VerificationError('Runtime manifest contains a missing or invalid source filename')
    for path in list((root/'providers').rglob('*.json')) + list((root/'presets').glob('*.json')):
        json.loads(path.read_text('utf8'))
    for filename in ('LICENSE', 'README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md',
                     'docs/INSTALLATION.md', 'docs/DEVELOPMENT.md', 'docs/THIRD_PARTY_NOTICES.md',
                     'docs/BINARY_LICENSES.md', 'LICENSES/manifest.json', 'LICENSES/dependency-sources.json',
                     'README.en.md', 'SUPPORT.md', 'CODE_OF_CONDUCT.md', 'CITATION.cff'):
        if not (root/filename).is_file():
            raise VerificationError('Required public file is missing: ' + filename)
    if json.loads((root/'providers/local-models.example.json').read_text('utf8'))['ready']:
        raise VerificationError('Public model template must remain unconfigured')
    if json.loads((root/'providers/qwen-image-2.1-models.example.json').read_text('utf8'))['deployment_root']:
        raise VerificationError('Public Qwen template must not contain an installation path')
    pins = read_pins(root/'scripts/requirements-desktop.txt')
    license_texts = verify_license_inventory(root)
    for row in license_texts:
        if row.get('origin', '').startswith('installed wheel RECORD: '):
            pin = pins.get(canonical_name(row['component']))
            if pin is None or row.get('version') != pin['version']:
                raise VerificationError('Preserved wheel license version differs from desktop requirements')
    for path in public_files(root):
        relative = validate_public_path(path.relative_to(root).as_posix())
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise VerificationError('Public source selection contains a missing or external file')
        if path.suffix.lower() in {'.py', '.md', '.json', '.txt', '.cmd', '.ps1', '.toml', '.yml', '.yaml', '.spec', '.example', '.cff'}:
            content = path.read_text(encoding='utf8', errors='replace')
            if re.search(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b', content):
                raise VerificationError('Credential-like value in public source: ' + relative.as_posix())
    snapshot_files = validate_source_manifest(root)
    return {'ok': True, 'python_sources': len(paths), 'runtime_modules': len(names),
            'license_texts': len(license_texts), 'desktop_pins': len(pins), 'snapshot_files': snapshot_files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        result = check(args.root.resolve())
    except (VerificationError, AssertionError) as error:
        print(json.dumps({'ok': False, 'error': str(error)}))
        return 2
    except Exception:
        print(json.dumps({'ok': False, 'error': 'Repository validation could not complete'}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
