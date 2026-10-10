"""Optional camera-RAW input support.

The production pipeline keeps camera originals untouched.  Pillow cannot
decode Sony ``.ARW`` files, so this small adapter uses ``rawpy`` only when a
RAW path is actually opened.  The dependency remains optional: existing JPEG
and PNG workflows do not import it and continue to work without rawpy.
"""

from __future__ import annotations

from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageOps


RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng"}
SOURCE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng"}
RGB_DECODE_VERSION = "oriented-srgb-icc-linearraw-v3"


@lru_cache(maxsize=1)
def deterministic_srgb_profile() -> bytes:
    """Canonical current sRGB metadata, independent of wall-clock creation time."""
    profile=bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    profile[24:36]=bytes.fromhex('07d000010001000000000000')
    profile[84:100]=bytes(16)  # ICC permits an unassigned profile ID.
    return bytes(profile)


def is_raw(path: Path) -> bool:
    return path.suffix.casefold() in RAW_SUFFIXES


def _require_rawpy():
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "RAW input requires the optional rawpy package. Install rawpy in "
            "the bundled Python environment, then retry the command."
        ) from exc
    return rawpy


def open_rgb(path: Path) -> Image.Image:
    """Decode an oriented sRGB image; untagged raster inputs explicitly assume sRGB."""
    if not is_raw(path):
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            profile_bytes = image.info.get("icc_profile")
            srgb = ImageCms.ImageCmsProfile(BytesIO(deterministic_srgb_profile()))
            if profile_bytes:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
                cms_input = oriented if oriented.mode in {"RGB", "RGBA", "CMYK", "LAB", "L"} else oriented.convert("RGB")
                result = ImageCms.profileToProfile(cms_input, source_profile, srgb, outputMode="RGB")
                management = "embedded_icc_to_sRGB"
                source_profile_name = ImageCms.getProfileName(source_profile).strip()
            else:
                result = oriented.convert("RGB")
                management = "assumed_sRGB"
                source_profile_name = None
            result.info.update({"icc_profile": srgb.tobytes(), "color_space": "sRGB",
                                "color_management": management, "source_profile_name": source_profile_name,
                                "decode_version": RGB_DECODE_VERSION})
            return result
    linear, _precision = open_linear(path)
    result = linear_preview(linear)
    result.info.update({"color_space": "sRGB", "color_management": "LibRaw_linear_to_IEC_sRGB",
                        "decode_version": RGB_DECODE_VERSION})
    return result


def open_linear(path: Path):
    """Decode camera RAW16 into linear sRGB primaries, or linearize a raster."""
    import numpy as np
    from float_appearance import srgb_to_linear

    if is_raw(path):
        rawpy = _require_rawpy()
        with rawpy.imread(str(path)) as raw:
            values = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                                    output_bps=16, gamma=(1, 1),
                                    output_color=rawpy.ColorSpace.sRGB, half_size=False)
        linear = np.asarray(values, dtype=np.float32) / np.float32(65535.0)
        precision = "raw16_linear"
    else:
        image = open_rgb(path)
        linear = srgb_to_linear(np.asarray(image, dtype=np.float32) / np.float32(255.0))
        precision = "raster8_srgb"
    return linear, {"source_precision": precision, "working_precision": "float32_linear",
                    "decode_version": RGB_DECODE_VERSION}


def linear_preview(linear, *, display_transform=None) -> Image.Image:
    """Encode a preview from the same float result used for the stored master."""
    import numpy as np
    from float_appearance import linear_to_srgb
    from display_transform import encode_display

    height, width = linear.shape[:2]
    values = np.empty((height, width, 3), dtype=np.uint8)
    for top in range(0, height, 256):
        encoded = (linear_to_srgb(linear[top:top + 256]) if display_transform is None
                   else encode_display(linear[top:top + 256], display_transform))
        values[top:top + 256] = np.uint8(encoded * 255.0 + 0.5)
    result = Image.fromarray(values)
    result.info.update({"icc_profile": deterministic_srgb_profile(),
                        "color_space": "sRGB", "decode_version": RGB_DECODE_VERSION,
                        "display_transform": display_transform})
    return result


def save_render_master(linear, path: Path, *, display_transform=None) -> None:
    """Persist the render once as ICC-tagged uint16 nonlinear sRGB TIFF."""
    import numpy as np
    import tifffile
    import zlib
    from float_appearance import linear_to_srgb
    from display_transform import encode_display

    height, width = linear.shape[:2]
    def encoded_strips():
        for top in range(0, height, 256):
            display = (linear_to_srgb(linear[top:top + 256]) if display_transform is None
                       else encode_display(linear[top:top + 256], display_transform))
            encoded = np.asarray(display * 65535.0 + 0.5, dtype='<u2')
            # TIFF byte iterators contain already-compressed strips; ndarray
            # iterators describe whole pages and would reassemble the canvas.
            yield zlib.compress(encoded.tobytes())
    profile = deterministic_srgb_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), encoded_strips(), shape=(height, width, 3), dtype='<u2', byteorder='<',
                     rowsperstrip=256, photometric="rgb", metadata=None,
                     compression="deflate", extratags=[(34675, "B", len(profile), profile, False)])


def save_scene_master(linear, path: Path, *, working_domain) -> None:
    """Store linear working RGB with its actual scene or appearance domain."""
    import numpy as np
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    if working_domain not in {'scene_linear_sRGB', 'display_linear_sRGB'}:
        raise ValueError('Unknown linear working domain')
    appearance = working_domain == 'display_linear_sRGB'
    tifffile.imwrite(str(path), np.asarray(linear, np.float32), photometric='rgb',
        compression='deflate', metadata={'working_space': 'linear-sRGB',
            'working_domain': working_domain, 'precision': 'float32',
            'display_transform_applied': appearance,
            'meaning': 'display-linear appearance render, not camera RAW' if appearance else 'scene-linear render, not camera RAW'})


def dimensions(path: Path) -> tuple[int, int]:
    """Return oriented dimensions without requiring Pillow to parse a RAW."""
    if not is_raw(path):
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    rawpy = _require_rawpy()
    with rawpy.imread(str(path)) as raw:
        sizes: Any = getattr(raw, "sizes", None)
        width = getattr(sizes, "width", None)
        height = getattr(sizes, "height", None)
        if width and height:
            # rawpy applies the camera's 90-degree orientation during
            # postprocess.  LibRaw flip codes 5–8 swap the displayed axes.
            flip = getattr(sizes, "flip", 0)
            if int(flip or 0) in {5, 6, 7, 8}:
                return int(height), int(width)
            return int(width), int(height)
        shape = getattr(getattr(raw, "raw_image_visible", None), "shape", None)
        if shape and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
    # The fallback is intentionally simple and only runs on unusual rawpy
    # builds that do not expose sizes metadata.
    return open_rgb(path).size


def storage_info(path: Path) -> dict[str, Any]:
    """Metadata equivalent used by registration and output manifests."""
    if is_raw(path):
        width, height = dimensions(path)
        return {
            "format": path.suffix.lstrip(".").upper(),
            "mode": "RAW",
            "raw_size": [width, height],
            "oriented_size": [width, height],
            "orientation": None,
            # ARW files in this handoff do not expose a verified per-channel
            # depth through the common rawpy metadata path; do not guess 12/14
            # bit here and accidentally advertise a high-bit deliverable.
            "bits_per_channel": None,
            "bit_depth_source": "not reported",
            "icc_present": False,
            "decode_color_space": "sRGB", "decode_color_management": "LibRaw_linear_to_IEC_sRGB",
            "decode_version": RGB_DECODE_VERSION,
        }
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        mode = image.mode
        bits = 16 if ("16" in mode or mode in {"I", "I;16", "I;16B", "I;16L"}) else 8
        return {
            "format": image.format,
            "mode": mode,
            "raw_size": [image.width, image.height],
            "oriented_size": [oriented.width, oriented.height],
            "orientation": image.getexif().get(274),
            "bits_per_channel": bits,
            "icc_present": bool(image.info.get("icc_profile")),
            "decode_color_space": "sRGB",
            "decode_color_management": "embedded_icc_to_sRGB" if image.info.get("icc_profile") else "assumed_sRGB",
            "decode_version": RGB_DECODE_VERSION,
        }
