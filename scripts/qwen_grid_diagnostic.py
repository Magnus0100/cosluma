"""Read-only Qwen image grid-artifact diagnostic.

This helper deliberately does not contact ComfyUI or rewrite any production
asset.  It compares already existing raster inputs/outputs using the same
high-pass diagonal-lag proxy used by the held-out Qwen A/B report.  It is
intended for readable texture diagnostics and corroborated periodic-artifact
checks on already returned images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    left = left - float(left.mean())
    right = right - float(right.mean())
    denominator = float(np.sqrt(np.mean(left * left) * np.mean(right * right)))
    return 0.0 if denominator <= 1e-12 else float(np.mean(left * right) / denominator)


def _lag_correlation(values: np.ndarray, dy: int, dx: int) -> float:
    height, width = values.shape
    top = max(0, dy)
    bottom = min(height, height + dy)
    left = max(0, dx)
    right = min(width, width + dx)
    if bottom <= top or right <= left:
        return 0.0
    return _correlation(values[top:bottom, left:right],
                        values[top - dy:bottom - dy, left - dx:right - dx])


def _highpass(values: np.ndarray, radius: float = 1.5) -> np.ndarray:
    # The held-out report used scipy's reflect-padded Gaussian.  Keep that
    # definition when scipy is available so historical values remain
    # comparable; Pillow is a deterministic fallback for minimal installs.
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8), mode="L")
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    else:
        blurred = gaussian_filter(values, sigma=radius, mode="reflect")
    return values - blurred


def _gray_metrics(gray: np.ndarray) -> dict:
    highpass = _highpass(gray)
    diagonal = {str(lag): _lag_correlation(highpass, lag, lag)
                for lag in (1, 2, 4, 8, 16)}
    neighbors = {str(lag): _lag_correlation(highpass, lag, lag)
                 for lag in (3, 5, 7, 9, 15, 17)}
    anti_diagonal = {str(lag): _lag_correlation(highpass, lag, -lag)
                     for lag in (1, 2, 4, 8, 16)}
    peak = max(diagonal[str(lag)] for lag in (4, 8, 16))
    return {
        "mean_rgb8": float(gray.mean() * 255.0),
        "std_rgb8": float(gray.std() * 255.0),
        "highpass_std": float(highpass.std() * 255.0),
        "diag_lag": diagonal,
        "diag_lag_neighbors": neighbors,
        "anti_diag_lag": anti_diagonal,
        "diag_peak_lag_4_16": float(peak),
    }


def grid_metrics(path: Path, *, content_sha256: str | None = None) -> dict:
    """Inspect visible colour; zero-alpha RGB can never affect the verdict.

    Opaque rasters retain the original luminance calculation exactly. An
    associated linear composite on both black and white tests opacity as well
    as colour without assuming the eventual photographic background.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[..., 3]
        if np.all(alpha == 255):
            metrics = _gray_metrics(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)
            metrics["evaluation_domain"] = "opaque_encoded_luminance"
        else:
            from float_appearance import srgb_to_linear, linear_to_srgb
            opacity = alpha.astype(np.float32) / 255.0
            associated = srgb_to_linear(rgba[..., :3].astype(np.float32) / 255.0) * opacity[..., None]
            projections = {}
            for name, background in (("over_black", 0.), ("over_white", 1.)):
                linear = associated + background * (1 - opacity[..., None])
                encoded = np.uint8(np.clip(linear_to_srgb(linear), 0, 1) * 255 + .5)
                gray = np.asarray(Image.fromarray(encoded).convert("L"), dtype=np.float32) / 255.0
                projections[name] = _gray_metrics(gray)
            primary = max(projections, key=lambda name: projections[name]["diag_peak_lag_4_16"])
            metrics = dict(projections[primary])
            metrics.update(evaluation_domain="associated_linear_rgb_over_neutral_backgrounds",
                           primary_projection=primary, visible_projections=projections)
        metrics["opacity"] = {"minimum": int(alpha.min()), "maximum": int(alpha.max()),
            "zero_pixels": int(np.count_nonzero(alpha == 0)),
            "nonopaque_pixels": int(np.count_nonzero(alpha != 255)),
            "hidden_rgb_ignored": True}
    return {"path": str(path),
        "sha256": content_sha256 if content_sha256 is not None else sha256_file(path),
        "bytes": path.stat().st_size, "size": [width, height], **metrics}


def periodic_grid_evidence(metrics: dict, *, threshold: float) -> dict:
    """Require two harmonic peaks distinct from ordinary neighboring detail.

    Positive correlation at one lag also occurs in natural material and edges.
    A periodic lattice has repeated peaks at a period and twice that period,
    with both peaks above their immediate neighbors. The historical absolute
    threshold stays unchanged; each independent peak must corroborate it.
    """
    if metrics.get("visible_projections"):
        projections = metrics["visible_projections"]
        evidence = {name: periodic_grid_evidence(value, threshold=threshold)
                    for name, value in projections.items()}
        corroborated = [name for name, value in evidence.items() if value["corroborated"]]
        primary = max(corroborated or list(evidence),
                      key=lambda name: projections[name]["diag_peak_lag_4_16"])
        return {**evidence[primary], "method": "two_harmonic_peaks_per_visible_projection.v1",
                "visible_projections": evidence, "primary_projection": primary,
                "corroborated": bool(corroborated)}
    scores = metrics.get("diag_lag") or {}
    neighbors = metrics.get("diag_lag_neighbors") or {}
    pairs = []
    for period in (4, 8):
        lags = (period, period * 2)
        values = [scores.get(str(lag)) for lag in lags]
        nearby = [[neighbors.get(str(lag - 1)), neighbors.get(str(lag + 1))]
                  for lag in lags]
        complete = all(isinstance(value, (int, float)) and np.isfinite(value)
                       for value in values + [value for pair in nearby for value in pair])
        prominence = [float(value) - max(pair) for value, pair in zip(values, nearby)] if complete else None
        corroborated = bool(complete and all(value > threshold for value in values)
                            and all(value > threshold for value in prominence))
        pairs.append({"lags": list(lags), "correlations": values,
                      "neighbor_correlations": nearby, "peak_prominences": prominence,
                      "corroborated": corroborated})
    return {"method": "two_harmonic_peaks_with_neighbor_prominence.v1",
            "threshold": threshold, "tested_pairs": pairs,
            "corroborated": any(pair["corroborated"] for pair in pairs)}


def _rows(values: Iterable[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("Images must use NAME=PATH syntax")
        if name.strip() in result:
            raise ValueError(f"Duplicate image name: {name.strip()}")
        result[name.strip()] = grid_metrics(Path(raw_path.strip()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True,
                        metavar="NAME=PATH",
                        help="Existing raster to inspect; repeat for clean/input/candidate images")
    parser.add_argument("--json", type=Path,
                        help="Optional diagnostic JSON path; never writes production assets")
    args = parser.parse_args()
    result = {
        "schema_version": "qwen_grid_diagnostic.v1",
        "read_only": True,
        "images": _rows(args.image),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        target = args.json.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
