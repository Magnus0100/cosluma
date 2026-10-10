"""Export a clean public source tree without Git history or private local data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

TOP_LEVEL = ('README.md', 'LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'CHANGELOG.md',
             '.gitignore', '.gitattributes', '.editorconfig', '.env.example', 'pyproject.toml',
             'Start-Cosplay-Studio.cmd', 'Start-Qwen-Image.cmd',
             'cosplay-workbench.spec', 'cosplay-workbench-debug.spec')
TREES = ('.github', 'docs', 'scripts', 'tests', 'providers', 'presets', 'LICENSES')
PRIVATE_CONFIGS = {'local-models.current.json', 'qwen-image-2.1-models.json'}


def public_files(root):
    selected = [root/name for name in TOP_LEVEL]
    selected += [root/'data/README.md', root/'outputs/README.md']
    for tree in TREES:
        for path in (root/tree).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}:
                continue
            relative = path.relative_to(root)
            if tree == 'docs' and len(relative.parts) > 2:
                continue
            if tree == 'providers' and path.name in PRIVATE_CONFIGS:
                continue
            selected.append(path)
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    destination = args.output.expanduser().resolve()
    if destination.exists() or destination.is_relative_to(root):
        parser.error('Output must be a new directory outside the checkout')
    files = public_files(root)
    from check_repository import check
    check(root)
    destination.mkdir(parents=True)
    records = []
    for path in files:
        relative = path.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({'path': relative.as_posix(), 'size': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination/'SOURCE_MANIFEST.json').write_text(json.dumps({'schema': 'cosplay_public_source.v1',
        'private_history_included': False, 'files': records}, indent=2)+'\n', encoding='utf8')
    check(destination)
    print(json.dumps({'source': str(destination), 'files': len(records),
                      'size': sum(e['size'] for e in records)}))


if __name__ == '__main__':
    main()
