"""Compile source and validate public runtime/resources without importing models."""
from __future__ import annotations
import ast
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

from prepare_source_release import public_files
from verify_desktop_dependencies import (VerificationError, canonical_name, checked_relative, read_pins,
                                         verify_license_inventory)


FORBIDDEN_SUFFIXES = {'.arw', '.nef', '.cr2', '.cr3', '.dng', '.jpg', '.jpeg', '.png', '.psd', '.psb',
                      '.pt', '.pth', '.onnx', '.gguf', '.safetensors', '.task', '.cosplayproj',
                      '.pyc', '.pyo', '.pyd', '.dll', '.exe', '.zip', '.7z'}
FORBIDDEN_SUFFIXES |= {'.orf', '.raf', '.rw2', '.tif', '.tiff', '.webp', '.heic', '.bmp',
                       '.ckpt', '.bin', '.model', '.weights', '.engine', '.trt', '.tflite',
                       '.mlmodel', '.mlpackage', '.pb', '.npy', '.npz', '.pem', '.key', '.p12', '.pfx',
                       '.sqlite', '.sqlite3', '.db', '.log', '.tmp', '.bak'}
FORBIDDEN_SUFFIXES |= {'.tar', '.gz', '.tgz', '.rar', '.orig', '.rej', '.mp4', '.mov', '.wav', '.mp3'}
PRIVATE_FILENAMES = {'project.json', 'local-models.current.json', 'qwen-image-2.1-models.json',
                     '.env', '.env.local', 'DATASET.json', 'DATASET.md'}


def validate_public_path(value):
    relative = checked_relative(value)
    parts = tuple(part.lower() for part in relative.parts)
    name = relative.name.lower()
    if any(part in {'.git', '.cache', '__pycache__', 'build', 'dist', 'workspace', '.venv', 'venv',
                    'runtimes', 'runtime', 'models', 'weights', 'checkpoints', '.pytest_cache', '.ruff_cache'}
           for part in parts):
        raise VerificationError('Private state or runtime cache selected for public source')
    if parts[0] != 'licenses' and any(part in {'evidence', 'reports', 'artifacts', 'scratch', 'logs',
                                               '.mypy_cache', 'htmlcov'} for part in parts):
        raise VerificationError('Local execution records selected for public source')
    if (relative.suffix.lower() in FORBIDDEN_SUFFIXES or name in {value.lower() for value in PRIVATE_FILENAMES}
            or (name.startswith('.env.') and name != '.env.example') or name in {'id_rsa', 'id_ed25519'}):
        raise VerificationError('Private asset, config or binary selected for public source')
    if parts[0] == 'output' or parts[0] == 'docs' and any(
            part in {'meetings', 'reviews'} for part in parts[1:]):
        raise VerificationError('Private execution evidence selected for public source')
    if relative.parts[0] in {'data', 'outputs'} and relative.as_posix() not in {'data/README.md', 'outputs/README.md'}:
        raise VerificationError('Private project data selected for public source')
    if parts[0] == 'model' and relative.as_posix() != 'model/README.md':
        raise VerificationError('Local model selected for public source')
    return relative


def validate_public_content(content, name):
    if re.search(r'\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b', content):
        raise VerificationError('Credential-like value in public source: ' + name)
    if re.search(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----', content):
        raise VerificationError('Private key in public source: ' + name)
    if re.search(r'(?i)(?:[a-z]:[/\\]+Users[/\\]+|/(?:Users|home)/)(?!Public(?:[/\\]|\b))[^\s/\\`"<>]+', content):
        raise VerificationError('Personal home path in public source: ' + name)


def validate_versions(root):
    tree = ast.parse((root/'scripts/single_source_workflow.py').read_text('utf8'))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                   and target.id == 'SOFTWARE_VERSION' for target in node.targets))
    if tomllib.loads((root/'pyproject.toml').read_text('utf8'))['project']['version'] != version:
        raise VerificationError('Project metadata version differs from runtime')
    for name, pattern in (('README.md', r'当前源码版本为 `([^`]+)`'),
                          ('README.en.md', r'The current source version is \*\*([^*]+)\*\*'),
                          ('CITATION.cff', r'(?m)^version: "([^"]+)"')):
        match = re.search(pattern, (root/name).read_text('utf8'))
        if not match or match[1] != version:
            raise VerificationError('Public version differs from runtime: ' + name)
    if not (root/'CHANGELOG.md').read_text('utf8').startswith('# 更新记录\n\n## ' + version + ' '):
        raise VerificationError('Changelog must begin with the current source version')


def validate_tracked_files(root, selected):
    if not (root/'.git').exists():
        return 0
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                            check=True, capture_output=True)
    tracked = set(result.stdout.decode('utf8').split('\0')) - {''}
    public = {path.relative_to(root).as_posix() for path in selected}
    unexpected = tracked - public
    if unexpected:
        raise VerificationError('Tracked file outside public source allowlist: ' + sorted(unexpected)[0])
    return len(tracked)


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
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()
              and not any(part in {'.git', '__pycache__'} for part in path.relative_to(root).parts)}
    if actual != names | {'SOURCE_MANIFEST.json'}:
        raise VerificationError('Public source snapshot contains unlisted files')
    return len(names)


def check(root):
    root = Path(root).resolve()
    validate_versions(root)
    selected = public_files(root)
    tracked_files = validate_tracked_files(root, selected)
    paths = [path for path in selected if path.suffix == '.py']
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.relative_to(root).as_posix())
    manifest = json.loads((root/'scripts/runtime_manifest.json').read_text('utf8'))
    names = manifest['files']
    if len(names) != len(set(names)) or names != sorted(names):
        raise VerificationError('Runtime manifest must contain unique sorted source filenames')
    if not all(Path(name).name == name and (root/'scripts'/name).is_file() for name in names):
        raise VerificationError('Runtime manifest contains a missing or invalid source filename')
    for path in selected:
        if path.relative_to(root).parts[0] not in {'providers', 'presets'} or path.suffix != '.json':
            continue
        json.loads(path.read_text('utf8'))
    for filename in ('AGENTS.md', 'docs/AGENT_GUIDE.md', 'LICENSE', 'README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md',
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
    for path in selected:
        relative = validate_public_path(path.relative_to(root).as_posix())
        if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                or any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)
                       if parent != root and parent.is_relative_to(root))):
            raise VerificationError('Public source selection contains a missing or external file')
        content = path.read_text(encoding='utf8', errors='replace')
        validate_public_content(content, relative.as_posix())
    snapshot_files = validate_source_manifest(root)
    return {'ok': True, 'python_sources': len(paths), 'runtime_modules': len(names),
            'license_texts': len(license_texts), 'desktop_pins': len(pins), 'snapshot_files': snapshot_files,
            'tracked_public_files': tracked_files}


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
