"""Optional camera-RAW input support.

The production pipeline keeps camera originals untouched.  Pillow cannot
decode Sony ``.ARW`` files, so this small adapter uses ``rawpy`` only when a
RAW path is actually opened.  The dependency remains optional: existing JPEG
and PNG workflows do not import it and continue to work without rawpy.
"""

from __future__ import annotations

from io import BytesIO
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from PIL import Image, ImageCms, ImageOps


RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng"}
SOURCE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng"}
RGB_DECODE_VERSION = "oriented-srgb-icc-linearraw-v3"

# RAW dimension probing opens the camera container through LibRaw.  That is
# materially slower than reading a normal raster header and was repeated on
# every import/retry even though the source is immutable.  Keep a tiny
# stat-bound metadata memo (separate from byte identities): a replacement or
# edit invalidates it through size/mtime/ctime, and a decode recipe change
# invalidates it through ``decode_version``.  Only RAW metadata is persisted;
# raster headers are already cheap and should not create sidecars for every
# preview asset.
_STORAGE_MEMO: dict[tuple[str, int, int, int], dict[str, Any]] = {}
_STORAGE_MEMO_LIMIT = 128
_STORAGE_CACHE_VERSION = 1


def _storage_cache_root() -> Path:
    override = os.environ.get('COSPLAY_SOURCE_METADATA_CACHE')
    if override:
        return Path(override).expanduser()
    # Direct worker invocations may not pass through the project launcher that
    # configures the cache environment.  Keep rebuildable RAW metadata on the
    # data volume in that case too; C:\TEMP is only the last-resort fallback
    # for machines without the configured D: volume.
    data_root = Path('D:/CosplayStudio')
    if data_root.is_dir():
        return data_root / 'outputs' / 'cache' / 'source-metadata'
    return Path(tempfile.gettempdir()) / 'cosplay-source-metadata-v1'


def _storage_cache_path(path: Path) -> Path:
    token = hashlib.sha256(str(path).encode('utf-8', 'surrogatepass')).hexdigest()
    return _storage_cache_root() / token[:2] / (token[2:] + '.json')


def _storage_stat(path: Path):
    stat = path.stat()
    return (str(path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)), stat


def _load_cached_storage(path: Path, key, stat) -> dict[str, Any] | None:
    cached = _STORAGE_MEMO.get(key)
    if cached is not None:
        return dict(cached)
    if os.environ.get('COSPLAY_DISABLE_SOURCE_METADATA_CACHE') == '1':
        return None
    try:
        record = json.loads(_storage_cache_path(path).read_text(encoding='utf-8'))
        if (record.get('cache_version') == _STORAGE_CACHE_VERSION
                and record.get('path') == str(path)
                and record.get('size') == int(stat.st_size)
                and record.get('mtime_ns') == int(stat.st_mtime_ns)
                and record.get('ctime_ns') == int(stat.st_ctime_ns)
                and record.get('decode_version') == RGB_DECODE_VERSION
                and isinstance(record.get('info'), dict)):
            info = dict(record['info'])
            _STORAGE_MEMO[key] = info
            return dict(info)
    except (OSError, ValueError, TypeError):
        pass
    return None


def _save_cached_storage(path: Path, key, stat, info: dict[str, Any]) -> None:
    if os.environ.get('COSPLAY_DISABLE_SOURCE_METADATA_CACHE') == '1':
        return
    _STORAGE_MEMO[key] = dict(info)
    while len(_STORAGE_MEMO) > _STORAGE_MEMO_LIMIT:
        _STORAGE_MEMO.pop(next(iter(_STORAGE_MEMO)))
    target = _storage_cache_path(path)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {'cache_version': _STORAGE_CACHE_VERSION, 'path': str(path),
                  'size': int(stat.st_size), 'mtime_ns': int(stat.st_mtime_ns),
                  'ctime_ns': int(stat.st_ctime_ns), 'decode_version': RGB_DECODE_VERSION,
                  'info': info}
        temporary = target.with_suffix(target.suffix + '.tmp-' + str(os.getpid()))
        temporary.write_text(json.dumps(record, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        os.replace(temporary, target)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _profile_decode_event(path: Path, operation: str, started: float, *, raw: bool) -> None:
    """Append opt-in decode timing without changing image decode behavior."""
    target = os.environ.get('COSPLAY_STAGE_PROFILE_PATH')
    if not target or started is None:
        return
    try:
        value = Path(path)
        size = value.stat().st_size
    except OSError:
        size = None
    try:
        with Path(target).open('a', encoding='utf8') as stream:
            stream.write(json.dumps({
                'schema': 'cosplay_stage_profile.v1', 'kind': 'decode',
                'operation': operation, 'path': str(Path(path).resolve()),
                'raw': bool(raw), 'bytes': size,
                'elapsed_seconds': round(time.perf_counter() - started, 6)},
                ensure_ascii=False, allow_nan=False) + '\n')
    except (OSError, TypeError, ValueError):
        # Profiling is strictly observational; an unavailable report must not
        # turn an otherwise valid image decode into a failed preparation.
        pass


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
    started = time.perf_counter() if os.environ.get('COSPLAY_STAGE_PROFILE_PATH') else None
    try:
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
    finally:
        _profile_decode_event(path, 'open_rgb', started, raw=is_raw(path))


def open_linear(path: Path):
    """Decode camera RAW16 into linear sRGB primaries, or linearize a raster."""
    import numpy as np
    from float_appearance import srgb_to_linear

    started = time.perf_counter() if os.environ.get('COSPLAY_STAGE_PROFILE_PATH') else None
    try:
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
    finally:
        _profile_decode_event(path, 'open_linear', started, raw=is_raw(path))


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
    return _preview_from_rgb8(values, display_transform)


def _preview_from_rgb8(values, display_transform):
    result = Image.fromarray(values)
    result.info.update({"icc_profile": deterministic_srgb_profile(),
                        "color_space": "sRGB", "decode_version": RGB_DECODE_VERSION,
                        "display_transform": display_transform})
    return result


def _write_render_master(linear, path: Path, display_transform, preview_values=None):
    """Encode bounded strips, optionally filling RGB8 from the same float strip."""
    import numpy as np
    import tifffile
    import zlib
    import os
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from float_appearance import linear_to_srgb
    from display_transform import encode_display

    height, width = linear.shape[:2]
    def encoded_strips():
        # zlib releases the GIL. Overlap compression with the next strip's
        # encoding, bounded to four strips rather than queuing a whole frame.
        # Ordered futures preserve identical TIFF strips and precision.
        workers=min(4,os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='master-deflate') as executor:
            pending=deque()
            for top in range(0, height, 256):
                display = (linear_to_srgb(linear[top:top + 256]) if display_transform is None
                           else encode_display(linear[top:top + 256], display_transform))
                if preview_values is not None:
                    # Round directly from the encoded float32 values for each
                    # output; deriving RGB8 from RGB16 can change threshold ties.
                    preview_values[top:top + 256] = np.uint8(display * 255.0 + 0.5)
                encoded = np.asarray(display * 65535.0 + 0.5, dtype='<u2')
                # TIFF byte iterators contain already-compressed strips.
                # Level 1 keeps the lossless TIFF byte contract while
                # avoiding the expensive high-ratio deflate pass on already
                # large photographic strips.  The pixel values, dtype and
                # strip order are unchanged; only the reversible container
                # compression work is reduced.
                pending.append(executor.submit(zlib.compress,encoded.tobytes(), 1))
                if len(pending)>=workers:yield pending.popleft().result()
            while pending:yield pending.popleft().result()
    profile = deterministic_srgb_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), encoded_strips(), shape=(height, width, 3), dtype='<u2', byteorder='<',
                     rowsperstrip=256, photometric="rgb", metadata=None,
                     compression="deflate", extratags=[(34675, "B", len(profile), profile, False)])


def save_render_master(linear, path: Path, *, display_transform=None) -> None:
    """Persist the render once as ICC-tagged uint16 nonlinear sRGB TIFF."""
    _write_render_master(linear, path, display_transform)


def save_render_preview_and_master(linear, path: Path, *, display_transform=None) -> Image.Image:
    """Return RGB8 preview and write RGB16 TIFF with one display encode per strip.

    Only the returned preview spans the whole frame. Float and uint16 encoding
    buffers stay at 256 rows, and the existing bounded zlib workers compress at
    most four strips in flight. Both outputs retain their original rounding and
    the deterministic sRGB profile.
    """
    import numpy as np
    height, width = linear.shape[:2]
    values = np.empty((height, width, 3), dtype=np.uint8)
    _write_render_master(linear, path, display_transform, values)
    return _preview_from_rgb8(values, display_transform)


def save_scene_master(linear, path: Path, *, working_domain) -> None:
    """Store linear working RGB with its actual scene or appearance domain."""
    import numpy as np
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    if working_domain not in {'scene_linear_sRGB', 'display_linear_sRGB'}:
        raise ValueError('Unknown linear working domain')
    appearance = working_domain == 'display_linear_sRGB'
    tifffile.imwrite(str(path), np.asarray(linear, np.float32), photometric='rgb',
        compression='deflate', compressionargs={'level': 1}, metadata={'working_space': 'linear-sRGB',
            'working_domain': working_domain, 'precision': 'float32',
            'display_transform_applied': appearance,
            'meaning': 'display-linear appearance render, not camera RAW' if appearance else 'scene-linear render, not camera RAW'})


def dimensions(path: Path) -> tuple[int, int]:
    """Return oriented dimensions without requiring Pillow to parse a RAW."""
    if not is_raw(path):
        with Image.open(path) as image:
            # ``size`` and EXIF orientation are header metadata.  Calling
            # exif_transpose here materializes a complete pixel image even
            # though callers only need the oriented canvas dimensions.
            orientation = image.getexif().get(274)
            width, height = image.size
            if orientation in {5, 6, 7, 8}:
                return int(height), int(width)
            return int(width), int(height)
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
    path = Path(path).expanduser().resolve()
    if is_raw(path):
        key, stat = _storage_stat(path)
        cached = _load_cached_storage(path, key, stat)
        if cached is not None:
            return cached
        width, height = dimensions(path)
        info = {
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
        _save_cached_storage(path, key, stat, info)
        return dict(info)
    with Image.open(path) as image:
        # Keep this path metadata-only: Pillow's mode/size/EXIF access does not
        # decode the raster, while exif_transpose would allocate a full pixel
        # copy solely to learn its dimensions.
        mode = image.mode
        orientation = image.getexif().get(274)
        width, height = image.size
        oriented_size = (height, width) if orientation in {5, 6, 7, 8} else (width, height)
        bits = 16 if ("16" in mode or mode in {"I", "I;16", "I;16B", "I;16L"}) else 8
        bit_depth_source = 'pillow_mode'
        # Pillow reports many RGB TIFFs as mode ``RGB`` regardless of the
        # actual BitsPerSample tag. Read only the mature TIFF header so source
        # registration records storage truth; the current non-RAW decode
        # contract remains raster8_srgb in open_linear.
        if path.suffix.casefold() in {'.tif', '.tiff'}:
            try:
                import tifffile
                with tifffile.TiffFile(str(path)) as resource:
                    page = resource.pages[0]
                    tag = page.tags.get('BitsPerSample')
                    values = tag.value if tag is not None else None
                    if isinstance(values, (tuple, list)):
                        values = tuple(int(value) for value in values)
                        if values:
                            bits = max(values)
                            bit_depth_source = 'tiff_bits_per_sample'
                    elif values is not None:
                        bits = int(values)
                        bit_depth_source = 'tiff_bits_per_sample'
            except (ImportError, OSError, ValueError, TypeError):
                # Header reporting is observational; retain Pillow's mode
                # fallback if an unusual TIFF cannot be inspected.
                pass
        return {
            "format": image.format,
            "mode": mode,
            "raw_size": [image.width, image.height],
            "oriented_size": [oriented_size[0], oriented_size[1]],
            "orientation": orientation,
            "bits_per_channel": bits,
            "bit_depth_source": bit_depth_source,
            "icc_present": bool(image.info.get("icc_profile")),
            "decode_color_space": "sRGB",
            "decode_color_management": "embedded_icc_to_sRGB" if image.info.get("icc_profile") else "assumed_sRGB",
            "decode_version": RGB_DECODE_VERSION,
        }
