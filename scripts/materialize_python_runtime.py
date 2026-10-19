"""One-time provisioning: replace shared site-packages with local dependencies.

Run with the target virtual environment's Python, before starting workers.
Materializes only explicitly requested dependency closures from installed
RECORD inventories. Same-volume immutable files can use hard links instead of
duplicate copies. Never downloads packages or model weights.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time


def _publish_text(path, text):
    temporary = path.with_name(path.name + '.materializing')
    temporary.write_text(text, encoding='utf8')
    temporary.replace(path)


def materialize(roots, receipt_path, *, dry_run=False, file_mode='copy', source_site=None):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if file_mode not in {'copy', 'hardlink'}:
        raise ValueError('Unsupported dependency materialization mode: ' + str(file_mode))

    prefix = Path(sys.prefix).resolve()
    site = prefix / 'Lib/site-packages'
    cfg = prefix / 'pyvenv.cfg'
    if sys.prefix == sys.base_prefix or not site.is_dir() or not cfg.is_file():
        raise ValueError('Run with the target Windows virtual environment Python')
    explicit_site = Path(source_site).resolve() if source_site is not None else None
    if explicit_site is not None and not explicit_site.is_dir():
        raise ValueError('Installed dependency source site is unavailable: ' + str(explicit_site))
    # First occurrence is the distribution actually selected by this runtime.
    installed = {}
    inventory = (metadata.distributions(path=[str(explicit_site), str(site)])
                 if explicit_site is not None else metadata.distributions())
    for dist in inventory:
        installed.setdefault(canonicalize_name(dist.metadata['Name']), dist)
    selected, pending = {}, [Requirement(name) for name in roots]
    pending = [row for row in pending if not row.marker or row.marker.evaluate({'extra': ''})]
    while pending:
        required = pending.pop()
        name = canonicalize_name(required.name)
        dist = installed.get(name)
        if dist is None or not required.specifier.contains(dist.version, prereleases=True):
            raise ValueError('Missing compatible installed dependency: ' + str(required))
        # Evaluate base requirements and any extras explicitly requested by a
        # dependency. Revisit only when an additional extra is discovered.
        previous = selected.get(name)
        extras = set(required.extras) | (previous['extras'] if previous else set())
        if previous and extras == previous['extras']:
            continue
        selected[name] = {'dist': dist, 'extras': extras}
        for text in dist.requires or []:
            child = Requirement(text)
            if not child.marker or any(child.marker.evaluate({'extra': extra}) for extra in {'', *extras}):
                pending.append(child)
    records = []
    for name, item in sorted(selected.items()):
        dist = item['dist']
        source_site = Path(dist.locate_file('')).resolve()
        files = list(dist.files or [])
        if not files:
            raise ValueError('No installed RECORD inventory: ' + name)
        row = {'name': name, 'version': dist.version, 'source_site': str(source_site),
               'files': [], 'bytes': 0, 'copied': source_site != site}
        seen_targets = set()
        for relative in files:
            path = Path(str(relative))
            # Console launchers are unnecessary for worker library imports.
            if '..' in path.parts:
                if (path.parts[:2] == ('..', '..') and len(path.parts) > 2
                        and path.parts[2] in {'Scripts', 'share'}):
                    continue
                raise ValueError('Unexpected external installed file: ' + str(relative))
            if path.suffix == '.pyc' or '__pycache__' in path.parts:
                continue
            source = (source_site / path).resolve()
            target = (site / path).resolve()
            if not source.is_relative_to(source_site) or not target.is_relative_to(site) or not source.is_file():
                raise ValueError('Unavailable installed library file: ' + str(source))
            if target in seen_targets:
                continue
            seen_targets.add(target)
            identical_link = (file_mode == 'hardlink' and target.is_file()
                              and os.path.samefile(source, target))
            if source_site != site and target.exists() and not identical_link:
                raise ValueError('Existing local file conflicts with shared dependency: ' + str(target))
            row['files'].append((str(source), str(target)))
            row['bytes'] += source.stat().st_size
        records.append(row)
    # All requirements and paths are checked before mutating the runtime.
    old_cfg = cfg.read_text(encoding='utf8')
    path_hooks = []
    installed_import_hooks = []
    local_distributions = None
    for pth in site.glob('*.pth'):
        text = pth.read_text(encoding='utf8')
        entries = [line.strip() for line in text.splitlines()
                   if line.strip() and not line.lstrip().startswith('#')]
        external = []
        for line in entries:
            if line.startswith(('import ', 'import\t')):
                recorded_source = next((Path(source) for row in records for source, target in row['files']
                                        if Path(target) == pth.resolve()), None)
                if (explicit_site is not None and recorded_source is not None
                        and recorded_source.is_relative_to(explicit_site)
                        and os.path.samefile(recorded_source, pth)):
                    if not any(row['path'] == str(pth) for row in installed_import_hooks):
                        installed_import_hooks.append({'path': str(pth), 'source': str(recorded_source),
                                                       'verification': 'selected_record_same_inode'})
                    continue
                # Adding one library must not replace or import a second
                # runtime's setuptools just to retain this environment's
                # already installed shim. Trust only an exact local RECORD
                # owner; arbitrary executable hooks still require review.
                if local_distributions is None:
                    local_distributions = list(metadata.distributions(path=[str(site)]))
                local_owner = next((dist for dist in local_distributions
                    if Path(dist.locate_file('')).resolve() == site
                    and any((site / str(item)).resolve() == pth.resolve()
                            for item in dist.files or [])), None)
                if local_owner is not None:
                    if not any(row['path'] == str(pth) for row in installed_import_hooks):
                        installed_import_hooks.append({'path': str(pth), 'source': str(pth),
                            'verification': 'target_local_record',
                            'distribution': local_owner.metadata['Name'], 'version': local_owner.version})
                    continue
                raise ValueError('Review executable .pth before materializing: ' + str(pth))
            resolved = (site / line).resolve()
            if not resolved.is_relative_to(prefix):
                external.append(line)
        if external:
            if len(external) != len(entries):
                raise ValueError('Mixed internal/external .pth requires review: ' + str(pth))
            path_hooks.append({'path': str(pth), 'original_text': text})
    if dry_run:
        result = {'runtime': str(prefix), 'distributions': [{k: v for k, v in row.items() if k != 'files'}
                                                          for row in records],
                  'file_mode': file_mode,
                  'materialized_bytes': sum(row['bytes'] for row in records if row['copied']),
                  'copied_bytes': sum(row['bytes'] for row in records if row['copied']) if file_mode == 'copy' else 0,
                  'linked_bytes': sum(row['bytes'] for row in records if row['copied']) if file_mode == 'hardlink' else 0,
                  'installed_import_hooks': installed_import_hooks,
                  'disabled_path_hooks': path_hooks}
        print(json.dumps(result, indent=2))
        return result
    receipt_path = Path(receipt_path).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'python_runtime_materialization.v1', 'status': 'copying', 'runtime': str(prefix),
              'roots': roots, 'downloads': 0, 'model_files_copied': 0,
              'file_mode': file_mode, 'linked_files': 0, 'reused_linked_files': 0, 'copied_files': 0,
              'previous_pyvenv_cfg': old_cfg, 'disabled_path_hooks': path_hooks,
              'installed_import_hooks': installed_import_hooks,
              'materialized_bytes': sum(row['bytes'] for row in records if row['copied']),
              'copied_bytes': sum(row['bytes'] for row in records if row['copied']) if file_mode == 'copy' else 0,
              'linked_bytes': sum(row['bytes'] for row in records if row['copied']) if file_mode == 'hardlink' else 0,
              'distributions': [{k: v for k, v in row.items() if k != 'files'} | {'file_count': len(row['files'])}
                                for row in records]}
    _publish_text(receipt_path, json.dumps(result, indent=2) + '\n')
    started = time.perf_counter()
    for row in records:
        if row['copied']:
            print('Materializing ' + row['name'] + ' ' + row['version'], flush=True)
            for source, target in row['files']:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                if file_mode == 'hardlink':
                    if Path(target).is_file() and os.path.samefile(source, target):
                        result['reused_linked_files'] += 1
                    else:
                        os.link(source, target)
                        result['linked_files'] += 1
                else:
                    shutil.copy2(source, target)
                    result['copied_files'] += 1
    new_cfg = '\n'.join('include-system-site-packages = false'
                        if line.startswith('include-system-site-packages') else line
                        for line in old_cfg.splitlines()) + '\n'
    # Disable only fully external path hooks; retain their text in the receipt.
    for row in path_hooks:
        _publish_text(Path(row['path']), '# Dependencies materialized in this runtime.\n')
    _publish_text(cfg, new_cfg)
    result.update(status='materialized', seconds=time.perf_counter() - started)
    _publish_text(receipt_path, json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'distributions'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, help='Installed distribution requirement')
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--file-mode', choices=('copy', 'hardlink'), default='copy',
                        help='Hardlink immutable installed files on the same volume without duplicate package data')
    parser.add_argument('--source-site', type=Path,
                        help='Select one installed dependency source ahead of target-local metadata')
    args = parser.parse_args()
    materialize(args.root, args.receipt, dry_run=args.dry_run, file_mode=args.file_mode,
                source_site=args.source_site)
