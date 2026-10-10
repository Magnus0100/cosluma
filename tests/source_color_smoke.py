"""Necessary behavioral checks for source-selected photographic color gains."""
import json

import numpy as np

import source_color
from source_color import (apply_source_color_plan, diagnose_source_colors,
                          encode_binary_mask, selection_mask, source_color_plan_for_run)


def main():
    height, width = 521, 537
    yy, xx = np.indices((height, width))
    detail = (1 + (xx % 2) * .04 + yy / height * .12).astype(np.float32)
    source = np.asarray([.14, .24, .10], dtype=np.float32)[None, None, :] * detail[..., None]
    source[:, width // 2:] = np.asarray([.025, .07, .23], dtype=np.float32)[None, None, :] * detail[:, width // 2:, None]
    source[0, 0] = 0
    region = {"id": "skin", "selection": {"base": "ellipse", "center": [.5, .5], "radius": [1, 1], "feather": 0},
              "correction": {"target_rgb": [.65, .45, .38], "strength": .7, "exposure_ev": 0},
              "color_guard": {"enabled": True, "reference_rgb": [.43, .55, .36], "tolerance": .20}}
    plan = {"enabled": True, "regions": [region]}
    off, summary = apply_source_color_plan(source, source, {**plan, "enabled": False})
    assert off is source and not summary["applied"]
    corrected, summary = apply_source_color_plan(source, source, plan)
    assert summary["region_count"] == 1 and corrected.dtype == np.float32
    assert np.array_equal(corrected[0, 0], source[0, 0])
    assert np.allclose(corrected @ source_color._LUMA, source @ source_color._LUMA, atol=1e-7)
    assert corrected[250, 150, 0] / corrected[250, 150, 1] > source[250, 150, 0] / source[250, 150, 1]
    assert np.array_equal(corrected[:, width // 2:], source[:, width // 2:]), "Source blue cloth must be excluded"
    assert corrected[250, 151, 1] > corrected[250, 150, 1], "Texture is retained, never flattened to target RGB"
    old_tile = source_color._TILE
    source_color._TILE = max(height, width)
    try:
        whole, _ = apply_source_color_plan(source, source, plan)
    finally:
        source_color._TILE = old_tile
    assert np.array_equal(corrected, whole)

    # The rendered right half is now green, but the immutable source was blue:
    # rendering changes must never move the color guard into clothing.
    rendered = source.copy()
    rendered[:, width // 2:] = [.14, .24, .10]
    guarded, _ = apply_source_color_plan(source, rendered, plan)
    assert np.array_equal(guarded[:, width // 2:], rendered[:, width // 2:])
    disabled = {"generation_design": {"design_brief": {"source_color_plan": plan}},
                "options": {"appearance_revision": {"source_color_plan": {"enabled": False}}}}
    assert not source_color_plan_for_run(disabled)["enabled"]
    assert not source_color_plan_for_run({})["enabled"]

    selection = {"base": "empty", "feather": 0, "strokes": [
        {"mode": "add", "radius": .1, "points": [[.3, .5], [.7, .5]]},
        {"mode": "erase", "radius": .04, "points": [[.5, .5]]}]}
    mask = selection_mask((400, 200), selection)
    assert mask.mode == "L" and mask.size == (400, 200)
    assert mask.getpixel((200, 100)) == 0 and mask.getpixel((160, 100)) == 255
    assert mask.getpixel((160, 130)) == 0, "Brush radius uses the short side"
    binary = np.zeros((40, 60), dtype=np.uint8)
    binary[10:30, 15:45] = 1
    encoded = encode_binary_mask(binary)
    automatic = {"base": "mask", "mask": encoded, "feather": 0, "strokes": []}
    restored = selection_mask((60, 40), automatic)
    assert np.array_equal(np.asarray(restored) > 0, binary > 0)
    enlarged = selection_mask((120, 80), {**automatic, "strokes": [
        {"mode": "erase", "radius": .08, "points": [[.5, .5]]}]})
    assert enlarged.getpixel((60, 40)) == 0 and enlarged.getpixel((35, 25)) == 255
    assert encode_binary_mask(np.ones((2, 3), dtype=bool))["counts"] == [0, 6]
    empty, summary = apply_source_color_plan(source, source, {"regions": [{**region, "selection": {"base": "empty"}}]})
    assert empty is source and not summary["applied"]
    diagnosis = diagnose_source_colors(source, regions=[region])
    assert diagnosis["grid"]["columns"] == 8 and diagnosis["grid"]["rows"] == 12
    assert diagnosis["local_regions"][0]["selected_samples"] > 0
    print(json.dumps({"ok": True, "checks": ["disabled_identity", "luminance_and_texture", "green_cast_correction",
          "source_guard_clothing", "source_guard_immutable", "tile_continuity", "brush_add_erase_short_side",
          "explicit_user_disable", "empty_identity", "source_diagnosis", "portable_auto_mask_rle_and_edit"]}))


if __name__ == "__main__":
    main()
