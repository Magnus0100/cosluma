"""Editable photographic light fields in the oriented source coordinate space.

Only the source's geometry and rendered source pixels participate. These are
local exposure/color corrections, not generated-person composites or 3D lights.
"""
from __future__ import annotations

import math

import numpy as np

PLAN_VERSION = "source_render_plan.v1"
_TILE = 512


def normalize_source_render_plan(plan: dict | None) -> dict:
    """Canonicalize the small public plan contract, including explicit off."""
    if not plan:
        return {"schema_version": PLAN_VERSION, "coordinate_space": "oriented_source_normalized",
                "enabled": False, "summary": "", "light_fields": []}

    def number(value, low, high):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Light field parameters must be finite numbers")
        return float(np.clip(value, low, high))

    fields = []
    for index, field in enumerate(plan.get("light_fields", [])):
        center = field.get("center", [.5, .5])
        radius = field.get("radius", [.15, .15])
        if len(center) != 2 or len(radius) != 2:
            raise ValueError("Light field center and radius must each have two coordinates")
        scope = str(field.get("scope", "all"))
        if scope not in ("all", "subject", "background"):
            raise ValueError(f"Unsupported light field scope: {scope}")
        fields.append({"id": str(field.get("id") or f"light-{index + 1}"),
                       "label": str(field.get("label") or f"局部光 {index + 1}"),
                       "center": [number(value, 0, 1) for value in center],
                       "radius": [number(value, .001, 1) for value in radius],
                       "rotation_deg": number(field.get("rotation_deg", 0), -360, 360),
                       "exposure_ev": number(field.get("exposure_ev", 0), -2, 2),
                       "temperature": number(field.get("temperature", 0), -1, 1),
                       "tint": number(field.get("tint", 0), -1, 1),
                       "saturation": number(field.get("saturation", 0), -1, 1),
                       "opacity": number(field.get("opacity", 1), 0, 1),
                       "scope": scope, "enabled": bool(field.get("enabled", True))})
    return {"schema_version": PLAN_VERSION, "coordinate_space": "oriented_source_normalized",
            "enabled": bool(plan.get("enabled", True)), "summary": str(plan.get("summary", "")),
            "light_fields": fields}


def source_render_plan_for_run(run: dict) -> dict:
    """User revision, even disabled/empty, wins over automatic source planning."""
    options = run.get("options") or {}
    revision = options.get("appearance_revision") or {}
    if "source_render_plan" in revision:
        return normalize_source_render_plan(revision["source_render_plan"])
    for design in (run.get("generation_design"), options.get("generation_design")):
        if isinstance(design, dict):
            brief = design.get("design_brief") or {}
            if "source_render_plan" in brief:
                return normalize_source_render_plan(brief["source_render_plan"])
    return normalize_source_render_plan(None)


def apply_source_render_plan(source_linear: np.ndarray, rendered_linear: np.ndarray,
                             plan: dict | None, silhouette=None) -> tuple[np.ndarray, dict]:
    """Apply smooth scene-linear gain fields in float32.

    The same full-image normalized coordinates are used in every tile. Gains
    retain source texture and exact black; the display transform handles
    highlights once at output. Optional subject scopes use only the source
    silhouette and are not interpreted as face/skin/material segmentation.
    """
    plan = normalize_source_render_plan(plan)
    summary = {"schema_version": PLAN_VERSION, "enabled": plan["enabled"], "applied": False,
               "coordinate_space": plan["coordinate_space"], "geometry": "source_only",
               "candidate_pixels_copied": False, "working_space": "linear-sRGB-float32",
               "highlight_protection": "deferred_to_output_display_transform",
               "fields_applied": [], "fields_skipped": []}
    if not plan["enabled"]:
        return rendered_linear, summary
    active = []
    for field in plan["light_fields"]:
        if not field["enabled"] or field["opacity"] == 0 or not any(
                field[key] for key in ("exposure_ev", "temperature", "tint", "saturation")):
            continue
        if field["scope"] != "all" and silhouette is None:
            summary["fields_skipped"].append({"id": field["id"], "reason": "source_silhouette_unavailable"})
            continue
        active.append(field)
    if not active:
        return rendered_linear, summary
    height, width = rendered_linear.shape[:2]
    if source_linear.shape != rendered_linear.shape:
        raise ValueError("Source and rendered image dimensions must match")
    if silhouette is not None:
        from PIL import Image
        silhouette = silhouette.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    result = np.empty_like(rendered_linear, dtype=np.float32)
    # Rotate in image pixels so a given angle looks identical on portrait and
    # landscape images; centers/radii still use normalized x/y coordinates.
    aspect = width / max(1, height)
    for top in range(0, height, _TILE):
        bottom = min(height, top + _TILE)
        yy = np.arange(top, bottom, dtype=np.float32)[:, None] / max(1, height - 1)
        for left in range(0, width, _TILE):
            right = min(width, left + _TILE)
            xx = np.arange(left, right, dtype=np.float32)[None, :] / max(1, width - 1)
            current = np.asarray(rendered_linear[top:bottom, left:right], dtype=np.float32).copy()
            subject = None
            if any(field["scope"] != "all" for field in active):
                subject = np.asarray(silhouette.crop((left, top, right, bottom)), dtype=np.float32) / 255
            for field in active:
                cx, cy = field["center"]
                rx, ry = field["radius"]
                angle = math.radians(field["rotation_deg"])
                cosine, sine = math.cos(angle), math.sin(angle)
                dx, dy = (xx - cx) * aspect, yy - cy
                u = (cosine * dx + sine * dy) / (rx * aspect)
                v = (-sine * dx + cosine * dy) / ry
                weight = np.exp(-2 * (u * u + v * v)) * np.float32(field["opacity"])
                if field["scope"] == "subject":
                    weight *= subject
                elif field["scope"] == "background":
                    weight *= 1 - subject
                temperature, tint = field["temperature"], field["tint"]
                stops = np.asarray([.35 * temperature + .12 * tint,
                                    -.12 * tint, -.35 * temperature + .12 * tint], dtype=np.float32)
                stops += np.float32(field["exposure_ev"])
                effective = weight[..., None] * stops
                gains = np.exp2(effective)
                current = current * gains
                if field["saturation"]:
                    luminance = (current[..., 0:1] * .2126 + current[..., 1:2] * .7152 + current[..., 2:3] * .0722)
                    current = luminance + (current - luminance) * (1 + field["saturation"] * weight[..., None])
            result[top:bottom, left:right] = current
    summary.update(applied=True, fields_applied=active, field_count=len(active))
    return result, summary
