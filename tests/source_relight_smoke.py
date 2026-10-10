"""Behavioral checks for manual source exposure and colour fields."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

import source_relight
from source_relight import apply_source_render_plan, source_render_plan_for_run


def main():
    height, width = 529, 557
    yy, xx = np.indices((height, width), dtype=np.float32)
    detail = .012 * ((xx.astype(np.int32) % 2) * 2 - 1)
    source = np.broadcast_to((.1 + yy[..., None] / height * .7), (height, width, 3)).copy()
    source += detail[..., None]
    source = source.astype(np.float32)
    source[0, 0] = 0
    source[0, 1] = .999
    plan = {"enabled": True, "light_fields": [{"id": "fill", "center": [.5, .45],
             "radius": [.45, .5], "rotation_deg": 25, "exposure_ev": .7,
             "temperature": .15, "tint": .25, "opacity": .8, "scope": "all"}]}
    off, summary = apply_source_render_plan(source, source, {**plan, "enabled": False})
    assert off is source and not summary["applied"]
    tiled, summary = apply_source_render_plan(source, source, plan)
    assert tiled.dtype == np.float32 and summary["field_count"] == 1
    assert np.array_equal(tiled[0, 0], source[0, 0])
    assert np.isfinite(tiled[0, 1]).all()
    assert summary['highlight_protection'] == 'deferred_to_output_display_transform'
    assert tiled[height // 2, width // 2, 0] > source[height // 2, width // 2, 0]
    assert tiled[height // 2, 101, 0] > tiled[height // 2, 100, 0]
    old_tile = source_relight._TILE
    source_relight._TILE = max(height, width)
    try:
        full, _ = apply_source_render_plan(source, source, plan)
    finally:
        source_relight._TILE = old_tile
    assert np.array_equal(full, tiled), "Tile boundaries must not change the operation"
    assert not np.shares_memory(source, tiled)

    silhouette = Image.new("L", (width, height), 0)
    outside, _ = apply_source_render_plan(source, source,
                                          {**plan, "light_fields": [{**plan["light_fields"][0], "scope": "subject"}]}, silhouette)
    assert np.array_equal(outside, source)
    run = {"generation_design": {"design_brief": {"source_render_plan": plan}},
           "options": {"appearance_revision": {"source_render_plan": {"enabled": False}}}}
    assert not source_render_plan_for_run(run)["enabled"]
    assert not source_render_plan_for_run({})["enabled"]

    print(json.dumps({"ok": True, "checks": ["disabled_identity", "linear_highlight_headroom", "texture_retained",
                                              "tile_continuity", "scope", "user_override"]}))


if __name__ == "__main__":
    main()
