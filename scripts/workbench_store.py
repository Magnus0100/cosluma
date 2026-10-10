#!/usr/bin/env python3
"""Small local JSON store for named workbench templates and recent runs."""

from __future__ import annotations

import json
import re
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def record_recent(root: Path, *, sample: str, run_id: str = "", source: str = "") -> dict[str, Any]:
    path = _settings_dir(root) / "recent_projects.json"
    current = _read(path, {"schema_version": "recent_projects.v1", "items": []})
    rows = [row for row in current.get("items", []) if isinstance(row, dict)]
    key = (str(sample), str(run_id))
    rows = [row for row in rows if (str(row.get("sample")), str(row.get("run_id"))) != key]
    rows.insert(0, {"sample": str(sample), "run_id": str(run_id), "source": str(source), "opened_at": _now()})
    rows = rows[:20]
    _write(path, {"schema_version": "recent_projects.v1", "items": rows})
    return rows[0]


def list_recent(root: Path) -> list[dict[str, Any]]:
    value = _read(_settings_dir(root) / "recent_projects.json", {"items": []})
    rows = [row for row in value.get("items", []) if isinstance(row, dict)] if isinstance(value, dict) else []
    seen = {(str(row.get("sample")), str(row.get("run_id"))) for row in rows}
    runs = root / "outputs" / "runs"
    if runs.is_dir():
        manifests = sorted(runs.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for manifest in manifests[:20]:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            item = {"sample": data.get("sample", manifest.parent.name.split("-r", 1)[0]), "run_id": data.get("run_id", manifest.parent.name), "source": (data.get("source") or {}).get("path", ""), "opened_at": data.get("created_at", "")}
            key = (str(item["sample"]), str(item["run_id"]))
            if key not in seen:
                rows.append(item); seen.add(key)
    return rows[:20]
