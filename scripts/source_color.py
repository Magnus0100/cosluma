"""Source-selected, luminance-preserving regional color-cast corrections."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PLAN_VERSION = "source_color_plan.v1"
_TILE = 512
_LUMA = np.asarray([.2126, .7152, .0722], dtype=np.float32)


def _number(value, low, high):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Regional color parameters must be finite")
    return float(np.clip(value, low, high))


def _pair(values, low, high):
    if len(values) != 2:
        raise ValueError("A selection coordinate requires two values")
    return [_number(value, low, high) for value in values]


def _rgb(values):
    if len(values) != 3:
        raise ValueError("A color reference requires three sRGB channels")
    return [_number(value, 0, 1) for value in values]


def _selection(selection):
    base = str(selection.get("base", "ellipse"))
    if base not in ("ellipse", "empty", "mask"):
        raise ValueError(f"Unsupported selection base: {base}")
    strokes = []
    for stroke in selection.get("strokes", []):
        mode = str(stroke.get("mode", "add"))
        if mode not in ("add", "erase"):
            raise ValueError(f"Unsupported brush mode: {mode}")
        strokes.append({"mode": mode, "radius": _number(stroke.get("radius", .02), .0001, 1),
                        "points": [_pair(point, 0, 1) for point in stroke.get("points", [])]})
    normalized = {"base": base, "center": _pair(selection.get("center", [.5, .5]), 0, 1),
            "radius": _pair(selection.get("radius", [.1, .1]), .0001, 1),
            "rotation_deg": _number(selection.get("rotation_deg", 0), -360, 360),
            "feather": _number(selection.get("feather", .01), 0, 1), "strokes": strokes}
    if base == "mask":
        encoded = selection["mask"]
        size = [int(value) for value in encoded["size"]]
        counts = [int(value) for value in encoded["counts"]]
        if len(size) != 2 or min(size) < 1 or any(value < 0 for value in counts) or sum(counts) != size[0] * size[1]:
            raise ValueError("Binary mask RLE must match its width/height")
        normalized["mask"] = {"size": size, "counts": counts}
    return normalized




def normalize_source_color_plan(plan: dict | None) -> dict:
    if not plan:
        return {"schema_version": PLAN_VERSION, "enabled": False, "summary": "", "regions": []}
    regions = []
    for index, region in enumerate(plan.get("regions", [])):
        correction = region.get("correction") or {}
        guard = region.get("color_guard") or {}
        target = _rgb(correction.get("target_rgb", [.65, .45, .38]))
        if not any(target):
            raise ValueError("Target RGB must define a nonblack chromaticity")
        regions.append({"id": str(region.get("id") or f"region-{index + 1}"),
                        "label": str(region.get("label") or f"局部颜色 {index + 1}"),
                        "enabled": bool(region.get("enabled", True)),
                        "selection": _selection(region.get("selection") or {}),
                        "correction": {"target_rgb": target,
                            "strength": _number(correction.get("strength", .5), 0, 1),
                            "exposure_ev": _number(correction.get("exposure_ev", 0), -2, 2)},
                        "color_guard": {"enabled": bool(guard.get("enabled", bool(guard))),
                            "reference_rgb": _rgb(guard.get("reference_rgb", [.5, .35, .25])),
                            "tolerance": _number(guard.get("tolerance", .25), .001, 1)}})
    return {"schema_version": PLAN_VERSION, "enabled": bool(plan.get("enabled", True)),
            "summary": str(plan.get("summary", "")), "regions": regions}


def source_color_plan_for_run(run: dict) -> dict:
    options = run.get("options") or {}
    revision = options.get("appearance_revision") or {}
    if "source_color_plan" in revision:
        return normalize_source_color_plan(revision["source_color_plan"])
    for design in (run.get("generation_design"), options.get("generation_design")):
        if isinstance(design, dict) and "source_color_plan" in (design.get("design_brief") or {}):
            return normalize_source_color_plan(design["design_brief"]["source_color_plan"])
    return normalize_source_color_plan(None)


def selection_mask(size: tuple[int, int], selection: dict) -> Image.Image:
    """Return exact-size L mask; brush radius and feather use image short side."""
    selection = _selection(selection)
    width, height = size
    short = min(width, height)
    mask = Image.new("L", size, 0)
    if selection["base"] == "mask":
        encoded = selection["mask"]
        flat = np.repeat(np.arange(len(encoded["counts"]), dtype=np.uint8) % 2, encoded["counts"])
        native = Image.fromarray(flat.reshape(encoded["size"][1], encoded["size"][0]) * 255, mode="L")
        mask = native.resize(size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(mask)
    if selection["base"] == "ellipse":
        cx, cy = selection["center"]
        rx, ry = selection["radius"]
        angle = math.radians(selection["rotation_deg"])
        cosine, sine = math.cos(angle), math.sin(angle)
        points = []
        for step in range(129):
            theta = step / 128 * math.tau
            dx, dy = rx * width * math.cos(theta), ry * height * math.sin(theta)
            points.append((cx * (width - 1) + cosine * dx - sine * dy,
                           cy * (height - 1) + sine * dx + cosine * dy))
        draw.polygon(points, fill=255)
    for stroke in selection["strokes"]:
        points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in stroke["points"]]
        if not points:
            continue
        radius = max(.5, stroke["radius"] * short)
        fill = 255 if stroke["mode"] == "add" else 0
        if len(points) > 1:
            draw.line(points, fill=fill, width=max(1, round(radius * 2)))
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    if selection["feather"]:
        mask = mask.filter(ImageFilter.GaussianBlur(selection["feather"] * short))
    return mask


def _display_rgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= .0031308, linear * 12.92,
                    1.055 * np.power(linear, 1 / 2.4) - .055).astype(np.float32)


def _linear_rgb(display):
    display = np.asarray(display, dtype=np.float32)
    return np.where(display <= .04045, display / 12.92,
                    ((display + .055) / 1.055) ** 2.4).astype(np.float32)


def _guard_weight(source, guard):
    if not guard["enabled"]:
        return np.ones(source.shape[:2], dtype=np.float32)
    display = _display_rgb(source)
    chroma = display / np.maximum(display.sum(axis=-1, keepdims=True), 1e-6)
    reference = np.asarray(guard["reference_rgb"], dtype=np.float32)
    reference /= max(float(reference.sum()), 1e-6)
    distance = np.sqrt(np.sum((chroma - reference) ** 2, axis=-1))
    # Full weight near the reference, smooth zero by tolerance. Unlike RGB
    # distance this does not reject the same color merely because it is shaded.
    ramp = np.clip((distance / guard["tolerance"] - .35) / .65, 0, 1)
    guard_weight = 1 - ramp * ramp * (3 - 2 * ramp)
    # Normalized chromaticity near sensor black is dominated by channel noise.
    # Suppress only that very dark range; ordinary shaded skin remains active.
    lowlight = np.clip((np.sum(source * _LUMA, axis=-1) - .0001) / .0019, 0, 1)
    return (guard_weight * lowlight * lowlight * (3 - 2 * lowlight)).astype(np.float32)


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] * .5)]])


def _source_representative(source, weights):
    luminance = np.sum(source * _LUMA, axis=-1)
    fit_weights = weights * np.sqrt(np.clip(luminance / .08, 0, 1))
    valid = fit_weights > 1e-4
    if not np.any(valid):
        return None
    pixels, used_weights = source[valid], fit_weights[valid]
    chroma = pixels / np.maximum(pixels.sum(axis=-1, keepdims=True), 1e-6)
    chroma = np.asarray([_weighted_median(chroma[:, channel], used_weights) for channel in range(3)], dtype=np.float32)
    chroma /= max(float(chroma.sum()), 1e-6)
    rgb = np.asarray([_weighted_median(pixels[:, channel], used_weights) for channel in range(3)], dtype=np.float32)
    return chroma, _display_rgb(rgb), int(valid.sum())


def apply_source_color_plan(source_linear: np.ndarray, rendered_linear: np.ndarray,
                            plan: dict | None, silhouette=None) -> tuple[np.ndarray, dict]:
    """Fit one robust rendered-to-target color gain per source-selected region.

    Selection and color guard always read immutable source colors. Constant
    regional gains preserve natural chromatic variation; no pixel is painted
    with the target RGB. Source/reference/target colors are never semantic skin
    detection, and the optional silhouette is deliberately not a color mask.
    """
    plan = normalize_source_color_plan(plan)
    execution = {"schema_version": PLAN_VERSION, "enabled": plan["enabled"], "applied": False,
                 "regions_applied": [], "regions_skipped": [], "region_count": 0,
                 "geometry": "source_only", "candidate_pixels_copied": False,
                 "color_guard_source": "immutable_source", "representative": "weighted_median_rendered_chromaticity_in_source_selection",
                 "guard_deep_shadow_taper": "source_linear_luma_0.0001_to_0.002",
                 "working_space": "linear-sRGB-float32", "luminance_preserved_before_exposure": True}
    if not plan["enabled"]:
        return rendered_linear, execution
    if source_linear.shape != rendered_linear.shape:
        raise ValueError("Source and rendered image dimensions must match")
    height, width = source_linear.shape[:2]
    stride = max(1, math.ceil(max(height, width) / 768))
    source_proxy = source_linear[::stride, ::stride]
    result = rendered_linear
    for region in plan["regions"]:
        correction = region["correction"]
        if not region["enabled"] or (correction["strength"] == 0 and correction["exposure_ev"] == 0):
            continue
        mask = selection_mask((width, height), region["selection"])
        proxy_mask = np.asarray(mask)[::stride, ::stride].astype(np.float32) / 255
        proxy_mask *= _guard_weight(source_proxy, region["color_guard"])
        source_representative = _source_representative(source_proxy, proxy_mask)
        rendered_representative = _source_representative(result[::stride, ::stride], proxy_mask)
        if source_representative is None or rendered_representative is None:
            execution["regions_skipped"].append({"id": region["id"], "reason": "no_selected_source_color"})
            continue
        measured, measured_display, sample_count = rendered_representative
        target = _linear_rgb(correction["target_rgb"])
        target /= max(float(target.sum()), 1e-6)
        gains = np.clip(target / np.maximum(measured, 1e-6), .25, 4)
        log_gain = np.log2(gains).astype(np.float32) * np.float32(correction["strength"])
        output = np.empty_like(result, dtype=np.float32)
        for top in range(0, height, _TILE):
            bottom = min(height, top + _TILE)
            for left in range(0, width, _TILE):
                right = min(width, left + _TILE)
                source_tile = source_linear[top:bottom, left:right]
                weight = np.asarray(mask.crop((left, top, right, bottom)), dtype=np.float32) / 255
                weight *= _guard_weight(source_tile, region["color_guard"])
                original = result[top:bottom, left:right]
                colored = original * np.exp2(weight[..., None] * log_gain)
                original_luma = np.sum(original * _LUMA, axis=-1, keepdims=True)
                color_luma = np.sum(colored * _LUMA, axis=-1, keepdims=True)
                # Exactly retain each pixel's existing luminance before the
                # independently requested local exposure; preserve pure black.
                colored *= np.divide(original_luma, color_luma, out=np.ones_like(original_luma), where=color_luma > 0)
                if correction["exposure_ev"]:
                    exposure = np.exp2(np.float32(correction["exposure_ev"]) * weight[..., None])
                    colored = colored * exposure
                # Scene-linear colors may exceed display gamut. The shared
                # output transform handles that, after all edits are applied.
                output[top:bottom, left:right] = np.where(weight[..., None] > 0,
                                                         colored, original)
        result = output
        execution["regions_applied"].append({"id": region["id"], "label": region["label"],
            "source_sample_rgb": [round(float(value), 5) for value in source_representative[1]],
            "rendered_sample_rgb": [round(float(value), 5) for value in measured_display],
            "target_rgb": correction["target_rgb"], "channel_gains": gains.tolist(),
            "strength": correction["strength"], "exposure_ev": correction["exposure_ev"],
            "color_guard": region["color_guard"], "selected_proxy_samples": sample_count})
    execution.update(applied=bool(execution["regions_applied"]), region_count=len(execution["regions_applied"]))
    return result, execution


def diagnose_source_colors(source: np.ndarray | Path, *, grid=(8, 12), regions=None, max_edge=768) -> dict:
    """Compact source color grid, plus optional plan-region representatives.

    ``regions`` is an optional list of source_color_plan region dictionaries;
    each may include selection/color_guard. RGB values are display sRGB, not
    recommended target colors or a quality score. Grid index gives location.
    """
    if isinstance(source, (str, Path)):
        from raw_support import open_linear
        path = Path(source)
        values, precision = open_linear(path)
        source_record = {"path": str(path.resolve()), "precision": precision["source_precision"]}
    else:
        values = np.asarray(source, dtype=np.float32)
        source_record = {"path": None, "precision": "provided_linear_array"}
    height, width = values.shape[:2]
    stride = max(1, math.ceil(max(height, width) / max_edge))
    proxy = values[::stride, ::stride]
    display = _display_rgb(proxy)
    ph, pw = proxy.shape[:2]
    columns, rows = grid
    cells = []
    for row in range(rows):
        entries = []
        y0, y1 = round(row * ph / rows), max(round(row * ph / rows) + 1, round((row + 1) * ph / rows))
        for column in range(columns):
            x0, x1 = round(column * pw / columns), max(round(column * pw / columns) + 1, round((column + 1) * pw / columns))
            patch = display[min(y0, ph - 1):min(y1, ph), min(x0, pw - 1):min(x1, pw)]
            linear_patch = proxy[min(y0, ph - 1):min(y1, ph), min(x0, pw - 1):min(x1, pw)]
            entries.append({"rgb": [round(float(value), 3) for value in np.median(patch.reshape(-1, 3), axis=0)],
                            "luma": round(float(np.median(np.sum(linear_patch * _LUMA, axis=-1))), 4)})
        cells.append(entries)
    local = []
    for region in regions or []:
        selection = _selection(region.get("selection") or {})
        mask = np.asarray(selection_mask((width, height), selection))[::stride, ::stride].astype(np.float32) / 255
        raw_guard = region.get("color_guard") or {}
        guard = {"enabled": bool(raw_guard.get("enabled", bool(raw_guard))),
                 "reference_rgb": _rgb(raw_guard.get("reference_rgb", [.5, .35, .25])),
                 "tolerance": _number(raw_guard.get("tolerance", .25), .001, 1)}
        representative = _source_representative(proxy, mask * _guard_weight(proxy, guard))
        local.append({"id": region.get("id"), "reference_rgb": None if representative is None else
                      [round(float(value), 4) for value in representative[1]],
                      "selected_samples": 0 if representative is None else representative[2]})
    return {"schema_version": "source_color_diagnostic.v1", "source": source_record,
            "source_size": [width, height], "coordinate_space": "oriented_source_normalized",
            "rgb_space": "display_sRGB_0_1", "grid": {"columns": columns, "rows": rows,
            "cell_location": "x=(column+.5)/columns,y=(row+.5)/rows", "cells": cells},
            "local_regions": local, "semantic_masks_inferred": False}
