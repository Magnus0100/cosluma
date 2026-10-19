#!/usr/bin/env python3
"""Small, file-backed preset registry for source-only runs.

Presets only describe provider prompt and appearance parameters.  They never
contain image bytes or geometry instructions, so applying one cannot bypass
the source-preserving renderer or visual review gate.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any
from installed_resources import resource_path

SCHEMA_VERSION = "presets.v1"
def _catalog_path():
    if not getattr(sys, 'frozen', False) and Path(__file__).resolve().parent.name == '_internal':
        path = Path(__file__).resolve().parent/'presets/presets.json'
        if not path.is_file():
            raise FileNotFoundError('Installed preset catalog is unavailable: '+str(path))
        return path
    return resource_path('presets/presets.json')


DEFAULT_PATH = _catalog_path()
REFINEMENT_PRESET = 'source_refinement'

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
        **({'refinement_profile': value['refinement_profile']} if ident == REFINEMENT_PRESET
           and isinstance(value.get('refinement_profile'), dict) else {}),
    }


def normalize_refinement_selector(selector):
    if (not isinstance(selector, dict) or set(selector) - {'id', 'version', 'overrides'}
            or selector.get('id') != REFINEMENT_PRESET
            or type(selector.get('version')) is not int or selector['version'] != 1):
        raise ValueError('Unknown bundled refinement profile/version')
    result = {'id': REFINEMENT_PRESET, 'version': 1}
    if 'overrides' in selector:
        overrides = selector['overrides']
        if (not isinstance(overrides, dict) or set(overrides) != {'input_prompt', 'character_notes'}
                or any(not isinstance(overrides[key], str) or len(overrides[key]) > limit
                       for key, limit in (('input_prompt', 64000), ('character_notes', 16000)))):
            raise ValueError('Refinement overrides must bind actual editable request text')
        result['overrides'] = dict(overrides)
    return result


def get_refinement_profile(selector):
    """Resolve only the bundled, fixed-scope recommendation, never overrides."""
    normalize_refinement_selector(selector)
    row = next((item for item in json.loads(DEFAULT_PATH.read_text(encoding='utf8'))['presets']
                if item.get('id') == REFINEMENT_PRESET), None)
    profile = row.get('refinement_profile') if isinstance(row, dict) else None
    allowed_goals = {'scene': 'preserve_and_enhance_source',
        'skin': 'coherent_in_requested_lighting_without_whitening',
        'materials': 'distinct_native_materials',
        'atmosphere': 'restrained_depth_without_unrequested_entities'}
    if (not isinstance(profile, dict) or set(profile) != {'id', 'version', 'goals'}
            or profile['id'] != REFINEMENT_PRESET or type(profile['version']) is not int
            or profile['version'] != 1 or profile['goals'] != allowed_goals
            or not isinstance(row.get('prompt'), str) or not row['prompt'].strip()
            or row.get('appearance') or row.get('appearance_regions') or row.get('manual_layers') or row.get('denoise_strength') != 0
            or row.get('provider') or row.get('model')):
        raise ValueError('Bundled refinement profile exceeds its fixed recommendation scope')
    canonical = {**profile, 'preset_id': REFINEMENT_PRESET, 'prompt': row['prompt']}
    content = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {**canonical, 'profile_sha256': hashlib.sha256(content.encode('utf8')).hexdigest()}


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
