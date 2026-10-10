#!/usr/bin/env python3
"""Smoke the file-backed style presets and source-only integration."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

import preset_store
import single_source_workflow
from generation_prompt import DEFAULT_PROMPT, PROMPT_VERSION, STYLE_MARKER, compose_generation_prompt


def main() -> int:
    presets = preset_store.list_presets()
    expected = {"anime_light_color", "stage_light", "night_scene", "soft_skin", "cinematic_backlight", "fantasy_atmosphere", "shrine_drama"}
    assert expected.issubset({item["id"] for item in presets}), presets
    with tempfile.TemporaryDirectory(prefix="cosplay-preset-") as temp:
        root = Path(temp)
        source = root / "source.png"
        Image.new("RGB", (160, 120), (90, 120, 160)).save(source)
        args = single_source_workflow.parse_args([
            "--root", str(root), "--source", str(source), "--preset", "night_scene",
        ])
        run = single_source_workflow.create_run(args)
        assert run["options"]["preset"]["id"] == "night_scene", run
        assert run["options"]["appearance_revision"]["params"] == {}, run
        style = preset_store.get_preset("night_scene")["prompt"]
        assert run["options"]["prompt"].endswith(STYLE_MARKER + style), run["options"]["prompt"]
        assert run["options"]["preset"]["prompt"] == style
        assert not run["options"].get("manual_layers"), run
        assert run["options"]["prompt"].startswith(DEFAULT_PROMPT)
        assert run["options"]["prompt_template_version"] == PROMPT_VERSION
        assert run["options"]["effective_prompt"] == run["options"]["prompt"]
        assert compose_generation_prompt(run["options"]["prompt"], style) == run["options"]["prompt"]
        custom = "Use soft neutral light and keep the existing environment."
        assert compose_generation_prompt(custom) == custom
    print("preset smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
