"""Freeze development code for the existing checked production CLI.

This source-only engineering helper copies an allowlisted execution closure and
prints argv/environment instructions. It neither executes photo work nor reads,
copies or rewrites private model/provider configuration. Plan and start use the
same snapshot's project_cli, coordinator, modules and public resources.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

SCHEMA = 'cosluma_engineering_execution.v1'
MANIFEST = 'execution-manifest.json'
MANIFEST_SHA = 'execution-manifest.sha256'
TOOL = 'scripts/engineering_execution.py'
BASE_RESOURCES = ('LICENSE', 'providers/registry.json', 'providers/local-models.example.json',
    'providers/qwen-image-2.1-models.example.json', 'providers/beautygrpo.example.json',
    'presets/presets.json', 'providers/workflows/qwen-image-2.1-effect-rgba.json')
PRIVATE_FALLBACKS = {'providers/local-models.current.json', 'providers/qwen-image-2.1-models.json'}
ENV_REMOVE = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'COSPLAY_PACKAGE_ROOT')
ENV_SET = {'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _identity(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _relative(value):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Snapshot entries require portable relative paths')
    path = Path(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or ':' in value or path.as_posix() != value:
        raise ValueError('Snapshot entry escapes its execution root')
    return path


def _file(root, relative):
    path = root/_relative(relative)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Execution closure file is absent or escapes the checkout: '+relative)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise ValueError('Execution closure must copy actual files, not links: '+relative)
    return path


def _public_resource(value):
    _relative(value)
    return value in BASE_RESOURCES or (value.startswith('providers/workflows/') and value.endswith('.json')) or (
        value.startswith('providers/comfyui/') and value.endswith('.py'))


def collect_files(source):
    """Use the runtime inventory and literal/registry public resources only.

    No tree glob can pull in a user's private files added under providers,
    scripts, model, config, data or outputs. Private fallback configs remain
    external even if present beside their example templates.
    """
    source = Path(source).expanduser().resolve(strict=True)
    runtime_path = _file(source, 'scripts/runtime_manifest.json')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    names = runtime.get('files')
    if (runtime.get('schema') != 'cosplay_runtime_sources.v1' or not isinstance(names, list)
            or names != sorted(set(names)) or any(not isinstance(name, str)
            or Path(name).name != name or not name.endswith('.py') for name in names)):
        raise ValueError('A sorted, complete production runtime manifest is required')
    required = {'project_cli.py', 'creative_task.py', 'production_coordinator.py',
                'single_source_workflow.py', 'runtime_resources.py', 'installed_resources.py'}
    if not required.issubset(names):
        raise ValueError('Runtime inventory omits a required existing production entry')
    selected = {'scripts/runtime_manifest.json', TOOL, *BASE_RESOURCES}
    selected.update('scripts/'+name for name in names)
    for name in names:
        module = _file(source, 'scripts/'+name)
        tree = ast.parse(module.read_text(encoding='utf-8-sig'), filename=name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'resource_path' and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                resource = node.args[0].value
                if resource in PRIVATE_FALLBACKS:
                    continue
                if not _public_resource(resource):
                    raise ValueError('Review this new public resource before freezing: '+resource)
                selected.add(resource)
    registry = json.loads(_file(source, 'providers/registry.json').read_text(encoding='utf-8'))
    def resource_strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('_resource'):
                    if not isinstance(item, str) or not _public_resource(item):
                        raise ValueError('Provider registry references an unsupported public resource')
                    selected.add(item)
                else:
                    resource_strings(item)
        elif isinstance(value, list):
            for item in value:
                resource_strings(item)
    resource_strings(registry)
    return [_file(source, name) for name in sorted(selected)]


def _records(source, files):
    return [{'path': path.relative_to(source).as_posix(), 'bytes': path.stat().st_size,
             'sha256': _hash(path)} for path in files]


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer); writer.flush(); os.fsync(writer.fileno())


def freeze(source, output):
    """Atomically publish a new code snapshot; never overwrite any directory."""
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Execution snapshot already exists; verify its saved identity instead of replacing it')
    if output.is_relative_to(source):
        raise ValueError('Execution snapshot must be outside the source checkout')
    files = collect_files(source); records = _records(source, files)
    body = {'schema': SCHEMA, 'purpose': 'existing_checked_production_cli',
        'resource_policy': 'runtime_inventory_and_explicit_public_resources_v1', 'files': records,
        'private_configuration_included': False, 'weights_included': False,
        'photos_included': False, 'credentials_included': False}
    manifest = {**body, 'snapshot_id': _identity(body)}
    payload = _json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.'+output.name+'.snapshot-'+uuid.uuid4().hex+'.partial')
    staging.mkdir()
    try:
        for record, path in zip(records, files):
            _copy_file(path, staging/record['path'])
            target = staging/record['path']
            if target.stat().st_size != record['bytes'] or _hash(target) != record['sha256']:
                raise ValueError('Source changed while copying execution code')
        # Detect edits both before and after each copy, not just files changed
        # before their turn. A changing producer cannot become a mixed snapshot.
        if _records(source, collect_files(source)) != records:
            raise ValueError('Source closure changed while execution code was being frozen')
        for name, data in ((MANIFEST, payload), (MANIFEST_SHA, (hashlib.sha256(payload).hexdigest()+'\n').encode('ascii'))):
            with (staging/name).open('xb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
        verify(staging, expected_id=manifest['snapshot_id'])
        # Independent bytes, not hardlinks to the mutable checkout. Read-only
        # files also prevent accidental editing by normal app/file operations.
        for path in staging.rglob('*'):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        os.rename(staging, output)
    finally:
        if staging.exists():
            # Only this exclusively created sibling can be removed on failure.
            if staging.parent != output.parent or not staging.name.startswith('.'+output.name+'.snapshot-'):
                raise RuntimeError('Refusing cleanup outside this snapshot staging directory')
            for path in staging.rglob('*'):
                if path.is_file():
                    path.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(staging)
    result = verify(output, expected_id=manifest['snapshot_id'])
    return {**result, 'source': str(source), 'reused': False}


def verify(snapshot, *, expected_id=None):
    """Check every frozen byte and reject extra files; do not consult checkout."""
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    payload = _file(snapshot, MANIFEST).read_bytes()
    manifest_sha = hashlib.sha256(payload).hexdigest()
    if _file(snapshot, MANIFEST_SHA).read_text(encoding='ascii').strip() != manifest_sha:
        raise ValueError('Execution manifest SHA differs')
    manifest = json.loads(payload)
    body = {key: value for key, value in manifest.items() if key != 'snapshot_id'}
    token = _identity(body)
    if (manifest.get('schema') != SCHEMA or manifest.get('snapshot_id') != token
            or expected_id is not None and expected_id != token):
        raise ValueError('Execution snapshot identity differs')
    records = manifest.get('files')
    if (not isinstance(records, list) or [row.get('path') for row in records]
            != sorted(set(row.get('path') for row in records))):
        raise ValueError('Execution snapshot file inventory changed')
    for row in records:
        if set(row) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Execution snapshot file record changed')
        path = _file(snapshot, row['path'])
        if path.stat().st_size != row['bytes'] or _hash(path) != row['sha256']:
            raise ValueError('Execution snapshot bytes changed: '+row['path'])
    expected_files = {row['path'] for row in records} | {MANIFEST, MANIFEST_SHA}
    actual_files = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*') if path.is_file()}
    if actual_files != expected_files:
        raise ValueError('Unexpected or missing files in frozen execution snapshot')
    return {'snapshot': str(snapshot), 'snapshot_id': token, 'manifest_sha256': manifest_sha,
            'files': len(records), 'bytes': sum(row['bytes'] for row in records),
            'identity_basis': 'expected_snapshot_id_and_actual_bytes' if expected_id else 'self_consistent_actual_bytes',
            'production_jobs_created': 0, 'model_calls': 0}


def command(snapshot, *, expected_id, python, workspace, action, request_file):
    """Return exact argv/environment edits for one explicit existing command.

    Apply environment.remove to inherited env, then environment.set. Keep
    existing credentials and absolute external model/config selectors in that
    inherited environment; never persist its contents in this public result.
    Private configs using ${PACKAGE_ROOT} need an externally resolved binding.
    This function never executes or retries the command.
    """
    checked = verify(snapshot, expected_id=expected_id)
    if action not in {'task-plan', 'task-start'}:
        raise ValueError('Only explicit checked task-plan/task-start commands are allowed')
    python = Path(python).expanduser().resolve(strict=True)
    workspace = Path(workspace).expanduser().resolve()
    request_file = Path(request_file).expanduser().resolve(strict=True)
    snapshot = Path(checked['snapshot'])
    if not python.is_file() or not request_file.is_file() or workspace.is_relative_to(snapshot) or request_file.is_relative_to(snapshot):
        raise ValueError('Interpreter/request must be files and project/private request must stay outside the code snapshot')
    return {**checked, 'action': action, 'argv': [str(python), '-I', '-B', '-X', 'utf8',
        str(snapshot/'scripts/project_cli.py'), '--root', str(workspace), action, '--request-file', str(request_file)],
        'cwd': str(snapshot), 'shell': False, 'automatic_retry': False,
        'environment': {'remove': list(ENV_REMOVE), 'set': dict(ENV_SET)},
        'external_configuration': {'copied': False, 'rewritten': False,
            'model_paths': 'Use existing absolute external registry/config paths; resolve PACKAGE_ROOT macros outside the snapshot.',
            'provider_credentials': 'Inherited existing credential storage/environment; never included in snapshot or command output.'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='operation', required=True)
    create = actions.add_parser('snapshot'); create.add_argument('--source', type=Path, required=True); create.add_argument('--output', type=Path, required=True)
    check = actions.add_parser('verify'); check.add_argument('--snapshot', type=Path, required=True); check.add_argument('--expected-id')
    launch = actions.add_parser('command'); launch.add_argument('--snapshot', type=Path, required=True); launch.add_argument('--expected-id', required=True)
    for flag in ('python', 'workspace', 'request-file'):
        launch.add_argument('--'+flag, type=Path, required=True)
    launch.add_argument('--action', choices=('task-plan', 'task-start'), required=True)
    args = parser.parse_args(argv)
    if args.operation == 'snapshot':
        result = freeze(args.source, args.output)
    elif args.operation == 'verify':
        result = verify(args.snapshot, expected_id=args.expected_id)
    else:
        result = command(args.snapshot, expected_id=args.expected_id, python=args.python,
            workspace=args.workspace, action=args.action, request_file=args.request_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
