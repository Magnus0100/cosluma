#!/usr/bin/env python3
"""Small, file-backed preset registry for source-only runs.

Presets only describe provider prompt and appearance parameters.  They never
contain image bytes or geometry instructions, so applying one cannot bypass
the source-preserving renderer or visual review gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "presets.v1"
DEFAULT_PATH = (Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]) / "presets" / "presets.json"

def _normalise(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    ident = str(value.get("id") or "").strip()
    if not ident or not all(ch.isalnum() or ch in "_-" for ch in ident):
        return None
    appearance = value.get("appearance") if isinstance(value.get("appearance"), dict) else {}
    return {
        "id": ident,
        "name": str(value.get("name") or ident),
        "description": str(value.get("description") or ""),
        "prompt": str(value.get("prompt") or "").strip(),
        "appearance": {str(k): float(v) for k, v in appearance.items() if isinstance(k, str) and isinstance(v, (int, float))},
        "denoise_strength": float(value.get("denoise_strength", 0.0)),
        "appearance_regions": value.get("appearance_regions") if isinstance(value.get("appearance_regions"), dict) else {},
        "manual_layers": value.get("manual_layers") if isinstance(value.get("manual_layers"), dict) else {},
        **({"provider": str(value["provider"])} if value.get("provider") else {}),
        **({"model": str(value["model"])} if value.get("model") else {}),
    }


def list_presets(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the single bundled catalog; optional files override by identifier."""
    payload = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    values = [_normalise(item) for item in payload["presets"]]
    values = [item for item in values if item]
    if path is not None and path.resolve() != DEFAULT_PATH.resolve():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("presets", payload) if isinstance(payload, dict) else payload
        overrides = {item["id"]: item for item in (_normalise(value) for value in raw) if item}
        values = [overrides.pop(item["id"], item) for item in values] + list(overrides.values())
    return values


def get_preset(identifier: str | None, path: Path | None = None) -> dict[str, Any] | None:
    ident = str(identifier or "").strip()
    if not ident:
        return None
    for item in list_presets(path):
        if item["id"] == ident:
            return item
    raise ValueError(f"unknown preset: {ident}")


def public_payload(path: Path | None = None) -> dict[str, Any]:
    return {"ok": True, "schema_version": SCHEMA_VERSION, "presets": list_presets(path)}
