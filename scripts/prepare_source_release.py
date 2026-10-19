"""Export a clean public source tree without Git history or private local data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TOP_LEVEL = ('AGENTS.md', 'README.md', 'README.en.md', 'LICENSE', 'CONTRIBUTING.md', 'SECURITY.md',
             'SUPPORT.md', 'CODE_OF_CONDUCT.md', 'CITATION.cff', 'CHANGELOG.md',
             '.gitignore', '.gitattributes', '.editorconfig', '.env.example', 'pyproject.toml',
             'Start-Cosplay-Studio.cmd', 'Start-Qwen-Image.cmd',
             'cosplay-workbench.spec', 'cosplay-workbench-debug.spec')
TREES = ('.github', 'docs', 'scripts', 'tests', 'providers', 'presets', 'LICENSES')
PRIVATE_CONFIGS = {'local-models.current.json', 'qwen-image-2.1-models.json'}


def public_files(root):
    root = Path(root).resolve()
    # Respect local ignore rules, including nested .gitignore files. Tracked
    # files remain selected so the gate rejects accidentally committed assets.
    candidates = None
    if (root/'.git').exists():
        result = subprocess.run(['git', '-C', str(root), 'ls-files', '--cached', '--others',
                                 '--exclude-standard', '-z'], check=True, capture_output=True)
        candidates = set(result.stdout.decode('utf8').split('\0'))
    selected = [root/name for name in TOP_LEVEL]
    selected += [root/'data/README.md', root/'model/README.md']
    for tree in TREES:
        for path in (root/tree).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}:
                continue
            relative = path.relative_to(root)
            if candidates is not None and relative.as_posix() not in candidates:
                continue
            if tree == 'docs' and len(relative.parts) > 2:
                continue
            if tree == 'providers' and path.name in PRIVATE_CONFIGS:
                continue
            selected.append(path)
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--for-repository', action='store_true',
                        help='Export editable public repository files without generated snapshot metadata')
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
    if not args.for_repository:
        (destination/'SOURCE_MANIFEST.json').write_text(json.dumps({'schema': 'cosplay_public_source.v1',
            'private_history_included': False,
            'source_commit': (subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
                              if (root/'.git').exists() else None),
            'files': records}, indent=2)+'\n', encoding='utf8')
    check(destination)
    print(json.dumps({'source': str(destination), 'files': len(records),
                      'mode': 'repository' if args.for_repository else 'release-source',
                      'size': sum(e['size'] for e in records)}))


if __name__ == '__main__':
    main()
