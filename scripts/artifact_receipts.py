"""Workspace-local observations of exports and verified portable imports.

Saved run manifests are immutable. Receipts stay outside their directories and
portable archives, and never grant completion to another workspace or revision.
Refreshes compare recorded file stats, not newly computed image checksums.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4


def file_stamp(path):
    stat = Path(path).stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def binding_for_run(run):
    return {'run_id': run.get('run_id'), 'revision': run.get('revision'),
            'source_sha256': (run.get('source') or {}).get('sha256')}


def receipt_directory(root, run_id):
    identifier = hashlib.sha256(str(run_id).encode('utf8')).hexdigest()
    return Path(root).resolve() / 'outputs' / 'artifact-receipts' / identifier


def write_receipt(root, run, receipt):
    value = {**receipt, 'binding': binding_for_run(run),
             'workspace_root': str(Path(root).resolve())}
    directory = receipt_directory(root, run['run_id'])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid4().hex + '.json')
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return target


def observe_receipts(root, run):
    """Read small receipts/sidecars and file stats without hashing image assets."""
    root = Path(root).resolve()
    directory = receipt_directory(root, run.get('run_id'))
    result = {'exports': [], 'imports': []}

    def unchanged(asset, expected_path=None):
        if not isinstance(asset, dict) or not asset.get('path') or not asset.get('sha256'):
            return False
        target = Path(asset['path']).resolve()
        return (expected_path is None or target == expected_path) and target.is_file() and file_stamp(target) == asset.get('stamp')

    def artifact_path(key):
        value = (run.get('artifacts') or {}).get(key)
        if not isinstance(value, str) or not value:
            return None
        target = Path(value)
        return (target if target.is_absolute() else root / target).resolve()

    # Ignore unfinished writes and corrupt metadata. There is no manifest fallback.
    for path in sorted(directory.glob('*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)[:64]:
        try:
            if path.stat().st_size > 128 * 1024:
                continue
            row = json.loads(path.read_text(encoding='utf8'))
            if not isinstance(row, dict) or row.get('workspace_root') != str(root):
                continue
            if row.get('binding') != binding_for_run(run):
                continue
            if row.get('schema_version') == 'source_export_receipt.v2':
                saved = row.get('saved_render') or {}
                expected = artifact_path(row.get('render_artifact'))
                if expected is None or not unchanged(saved, expected) or not unchanged(row.get('export')):
                    continue
                sidecar = row.get('sidecar') or {}
                if not unchanged(sidecar):
                    continue
                data = Path(sidecar['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != sidecar.get('sha256'):
                    continue
                metadata = json.loads(data)
                if not isinstance(metadata, dict):
                    continue
                expected_metadata = {**row['binding'], 'format': row.get('format'),
                    'artifact': row.get('artifact'), 'actual_bit_depth': row.get('actual_bit_depth'),
                    'export_path': row['export']['path'], 'export_sha256': row['export']['sha256'],
                    'render_sha256': saved.get('sha256')}
                if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                    continue
                result['exports'].append({**row, 'files_unchanged': True,
                    'metadata_matches': True, 'pixels_rehashed_on_refresh': False})
            elif row.get('schema_version') == 'portable_import_receipt.v1':
                saved = row.get('saved_render_artifacts') or {}
                keys = ('auto_draft', 'render_master', 'render_scene_master')
                if not all(artifact_path(key) is not None and unchanged(saved.get(key), artifact_path(key)) for key in keys):
                    continue
                result['imports'].append({**row, 'files_unchanged': True,
                    'pixels_rehashed_on_refresh': False})
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return result
