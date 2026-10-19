#!/usr/bin/env python3
"""Small local JSON store for named workbench templates and recent runs."""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NAME = re.compile(r"[^0-9A-Za-z_\-\u4e00-\u9fff .]+")


def _settings_dir(root: Path) -> Path:
    path = root / "outputs" / "settings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default


def _write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.02*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _clean_name(value: Any) -> str:
    name = _NAME.sub("", str(value or "")).strip(" .")
    if not name:
        raise ValueError("template name is required")
    return name[:80]


def list_templates(root: Path) -> list[dict[str, Any]]:
    value = _read(_settings_dir(root) / "workbench_templates.json", {"templates": []})
    rows = value.get("templates", []) if isinstance(value, dict) else []
    return sorted((row for row in rows if isinstance(row, dict) and row.get("name")), key=lambda row: str(row.get("name")))


def save_template(root: Path, name: Any, payload: dict[str, Any]) -> dict[str, Any]:
    identifier = _clean_name(name)
    if not isinstance(payload, dict):
        raise ValueError("template must be a JSON object")
    # Generated face pixels are bound to one source, never a reusable preset.
    payload = copy.deepcopy(payload)
    from cos_workflow_graph import validate_template
    for owner in (payload, payload.get('options')):
        if not isinstance(owner, dict):
            continue
        owner.pop('portrait_beauty', None)
        if owner.get('workflow_graph'):
            owner['workflow_graph'] = validate_template(owner['workflow_graph'])
        revision = owner.get('appearance_revision')
        if isinstance(revision, dict):
            revision.pop('portrait_beauty', None)
            if revision.get('workflow_graph'):
                revision['workflow_graph'] = validate_template(revision['workflow_graph'])
    path = _settings_dir(root) / "workbench_templates.json"
    current = _read(path, {"schema_version": "workbench_templates.v1", "templates": []})
    rows = [row for row in current.get("templates", []) if isinstance(row, dict) and row.get("name") != identifier]
    old = next((row for row in current.get("templates", []) if isinstance(row, dict) and row.get("name") == identifier), {})
    row = {
        "schema_version": "workbench_template.v1",
        "name": identifier,
        "created_at": old.get("created_at") or _now(),
        "updated_at": _now(),
        "parameters": payload,
    }
    rows.append(row)
    _write(path, {"schema_version": "workbench_templates.v1", "templates": rows})
    return row


def delete_template(root: Path, name: Any) -> dict[str, Any]:
    identifier = _clean_name(name)
    path = _settings_dir(root) / "workbench_templates.json"
    current = _read(path, {"schema_version": "workbench_templates.v1", "templates": []})
    rows = [row for row in current.get("templates", []) if isinstance(row, dict) and row.get("name") != identifier]
    _write(path, {"schema_version": "workbench_templates.v1", "templates": rows})
    return {"name": identifier, "deleted": len(rows) != len(current.get("templates", []))}


def _saved_item(root: Path, run_id: str) -> dict[str, Any] | None:
    """Only complete, existing saved pixels can become a startup result."""
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        return None
    manifest = root / 'outputs/runs' / run_id / 'manifest.json'
    run = _read(manifest, {})
    if (not isinstance(run, dict) or run.get('kind') != 'single_input_run' or run.get('run_id') != run_id
            or run.get('status') not in {'needs_visual_review', 'accepted', 'delivered'}
            or not run.get('auto_draft')):
        return None
    publication = run.get('result_publication')
    if (not isinstance(publication, dict) or publication.get('schema') != 'cosplay_saved_result_publication.v1'
            or publication.get('state') != 'committed'):
        return None
    # Verified portable imports can retain their original immutable
    # publication owner; relocation must not fabricate a new commit.
    owner = (run.get('portable_import') or {}).get('original_run_id') or run_id
    if publication.get('run_id') != owner:
        return None
    scope = publication.get('scope')
    if scope == 'checked_production':
        import hashlib
        contract = run.get('production_contract')
        if (not isinstance(contract, dict) or contract.get('schema') != 'cosplay_production_contract.v1'
                or contract.get('plan_id') != publication.get('plan_id')):
            return None
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode('utf8')).hexdigest()
        if digest != publication.get('contract_sha256'):
            return None
    elif scope != 'local_save':
        return None
    continuation = run.get('continuation_edit') or {}
    if not isinstance(continuation, dict) or continuation and continuation.get('render_state') != 'saved':
        return None
    artifacts = run.get('artifacts') or {}
    if not isinstance(artifacts, dict):
        return None
    for key in ('auto_draft', 'render_master'):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            return None
        asset = Path(value)
        if not (asset if asset.is_absolute() else root / asset).is_file():
            return None
    revision = run.get('revision') or 1
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        return None
    return {'sample': str(run.get('sample') or run_id), 'run_id': run_id,
            'source': str((run.get('source') or {}).get('path') or ''),
            'created_at': str(run.get('created_at') or ''),
            'saved_at': str(continuation.get('saved_at') or run.get('saved_at')
                            or run.get('created_at') or ''),
            'revision': revision}


def _recent_value(root: Path) -> dict[str, Any]:
    value = _read(_settings_dir(root) / 'recent_projects.json', {})
    if not isinstance(value, dict):
        return {}
    if not isinstance(value.get('items', []), list):
        value['items'] = []
    return value


def _saved_rows(root: Path, value: dict[str, Any]) -> list[dict[str, Any]]:
    indexed = {str(row.get('run_id')): row for row in value.get('items', [])
               if isinstance(row, dict) and row.get('run_id')}
    identifiers = set(indexed)
    identifiers.update(path.parent.name for path in (root / 'outputs/runs').glob('*/manifest.json'))
    rows = []
    for identifier in identifiers:
        item = _saved_item(root, identifier)
        if item is None:
            continue
        old = indexed.get(identifier, {})
        if old.get('saved_at'):
            item['saved_at'] = str(old['saved_at'])
        item['opened_at'] = str(old.get('opened_at') or item['created_at'])
        rows.append(item)
    # A later review or an asset relocation changes mtime, never save order.
    return sorted(rows, key=lambda row: (row['saved_at'], row['created_at'],
                                         row['revision'], row['run_id']), reverse=True)[:20]


def record_recent(root: Path, *, sample: str, run_id: str = "", source: str = "") -> dict[str, Any]:
    from cache_store import exclusive_file_lock
    root = Path(root)
    path = _settings_dir(root) / 'recent_projects.json'
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=10):
        item = _saved_item(root, str(run_id))
        if item is None:
            return {'recorded': False, 'run_id': str(run_id), 'reason': 'no_complete_saved_result'}
        current = _recent_value(root)
        rows = _saved_rows(root, current)
        previous = next((row for row in rows if row['run_id'] == run_id), None)
        indexed_before = any(isinstance(row, dict) and row.get('run_id') == run_id
                             for row in current.get('items', []))
        item['saved_at'] = previous['saved_at'] if indexed_before and previous else _now()
        item['opened_at'] = (previous or {}).get('opened_at') or item['created_at']
        rows = [row for row in rows if row['run_id'] != run_id] + [item]
        rows.sort(key=lambda row: (row['saved_at'], row['created_at'], row['revision'], row['run_id']), reverse=True)
        rows = rows[:20]
        current.update(schema_version='recent_projects.v2', items=rows,
                       latest_saved_run_id=rows[0]['run_id'])
        _write(path, current)
        return {**item, 'recorded': True}


def list_recent(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    return _saved_rows(root, _recent_value(root))


def saved_result_state(root: Path) -> dict[str, Any]:
    root = Path(root)
    value = _recent_value(root)
    rows = _saved_rows(root, value)
    selected = str(value.get('selected_saved_run_id') or '')
    if selected and _saved_item(root, selected) is None:
        selected = ''
    return {'latest_saved_run_id': rows[0]['run_id'] if rows else '',
            'selected_saved_run_id': selected,
            'selection_explicit': bool(selected and value.get('selection_explicit'))}


def select_saved_result(root: Path, run_id: str, *, explicit: bool = False) -> dict[str, Any]:
    from cache_store import exclusive_file_lock
    root = Path(root)
    path = _settings_dir(root) / 'recent_projects.json'
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=10):
        if _saved_item(root, str(run_id)) is None:
            raise ValueError('Only a complete saved result can be selected for startup')
        current = _recent_value(root)
        # Loading or completing a candidate must not replace the user's
        # explicitly chosen saved result. Viewing remains independent of the
        # startup selection; another explicit choice may replace it.
        chosen = str(current.get('selected_saved_run_id') or '')
        if (not explicit and current.get('selection_explicit') and chosen
                and _saved_item(root, chosen) is not None):
            return saved_result_state(root)
        current.update(selected_saved_run_id=str(run_id), selection_explicit=bool(explicit))
        _write(path, current)
    return saved_result_state(root)
