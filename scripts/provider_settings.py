"""Per-user public provider settings and read-only CCSwitch credentials.

Only public connection metadata is persisted. Keys stay in their existing
environment/CCSwitch store and are read into memory only for transport.
"""
from __future__ import annotations

import json
from contextlib import closing
import os
from pathlib import Path
import re
import sqlite3
import tomllib
from typing import Any
from urllib.parse import urlsplit

FIELDS = {"endpoint", "model", "secret_env", "credential_source"}
SOURCES = {"auto", "environment", "ccswitch", "none"}


def settings_path() -> Path:
    override = os.environ.get("COSPLAY_PROVIDER_SETTINGS")
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get('COSPLAY_CONFIG_ROOT'):
        return Path(os.environ['COSPLAY_CONFIG_ROOT']).expanduser().resolve()/'provider-settings.json'
    folder = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".config")
    return folder / "CosplayWorkbench" / "provider-settings.json"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "1.0", "default_provider": None, "providers": {}}
    providers = data.get("providers", {}) if isinstance(data, dict) else {}
    return {"schema_version": "1.0", "default_provider": data.get("default_provider") if isinstance(data, dict) else None,
            "providers": {str(name): {key: value for key, value in config.items() if key in FIELDS}
                          for name, config in providers.items() if isinstance(config, dict)}}


def _public_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials, query or fragment")
    return value.rstrip("/")


def save_settings(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Merge public fields; reject secret or unknown fields before writing."""
    if set(payload) - {"schema_version", "default_provider", "providers"}:
        raise ValueError("provider settings accept only public configuration fields")
    data = load_settings(path)
    if "default_provider" in payload:
        data["default_provider"] = str(payload["default_provider"]).strip() if payload["default_provider"] else None
    for name, changes in payload.get("providers", {}).items():
        if not isinstance(changes, dict) or set(changes) - FIELDS:
            raise ValueError("provider configuration accepts endpoint/model/secret_env/credential_source only")
        config = dict(data["providers"].get(str(name), {}))
        for field, value in changes.items():
            if value is None or value == "":
                config.pop(field, None)
                continue
            value = str(value).strip()
            if field == "credential_source" and value not in SOURCES:
                raise ValueError("credential_source must be auto/environment/ccswitch/none")
            if field == "secret_env" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise ValueError("secret_env must be an environment variable name")
            config[field] = _public_endpoint(value) if field == "endpoint" else value
        data["providers"][str(name)] = config
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return data


def _ccswitch_record(path: Path | None = None) -> dict[str, Any]:
    """Private in-memory record. Never return this from an API or log it."""
    target = path or Path.home() / ".cc-switch" / "cc-switch.db"
    if not target.is_file():
        return {"status": "not_found"}
    try:
        with closing(sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT name, settings_config FROM providers WHERE app_type = 'codex' AND is_current = 1 AND lower(name) LIKE '%sub2api%' LIMIT 1").fetchone()
        if not row:
            return {"status": "no_current_sub2api"}
        settings = json.loads(row[1])
        config = tomllib.loads(settings.get("config") or "")
        provider = config.get("model_providers", {}).get(config.get("model_provider"), {})
        endpoint = provider.get("base_url")
        key = settings.get("auth", {}).get("OPENAI_API_KEY") or ""
        return {"status": "ready" if endpoint and key else "incomplete", "provider_name": row[0],
                "endpoint": _public_endpoint(str(endpoint)) if endpoint else None, "key": str(key)}
    except (sqlite3.Error, ValueError, TypeError):
        return {"status": "unreadable_configuration"}


def discover_ccswitch(path: Path | None = None) -> dict[str, Any]:
    record = _ccswitch_record(path)
    return {"available": record.get("status") == "ready", "status": record.get("status"),
            "provider_name": record.get("provider_name"), "endpoint": record.get("endpoint"),
            "credential_source": "ccswitch", "api_key_configured": bool(record.get("key")),
            "read_only": True}


def public_settings(path: Path | None = None) -> dict[str, Any]:
    return {**load_settings(path), "settings_path": str(path or settings_path()), "discovery": discover_ccswitch()}


def resolve_credential(provider: str, secret_env: str | None, credential_source: str,
                       endpoint: str | None = None) -> str:
    """Resolve a transport-only secret, bound to the discovered CCSwitch host."""
    if credential_source == "none":
        return ""
    if credential_source in {"auto", "environment"}:
        key = os.environ.get(secret_env, "") if secret_env else ""
        if not key and secret_env in {"COSPLAY_REFERENCE_API_KEY", "OPENAI_API_KEY"}:
            key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            return key
    if credential_source == "ccswitch" or (credential_source == "auto" and provider == "ccswitch_sub2api"):
        record = _ccswitch_record()
        base = str(record.get("endpoint") or "").rstrip("/")
        request_base = str(endpoint or "").rstrip("/")
        if request_base.endswith("/images/edits"):
            request_base = request_base[:-len("/images/edits")]
        if base and request_base == base:
            return record.get("key") or ""
    return ""
