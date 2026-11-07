"""Decode originals, fit canvases and save ordinary image files."""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from PySide6.QtGui import QColorSpace, QImage, QImageReader

Image.MAX_IMAGE_PIXELS = None
QImageReader.setAllocationLimit(0)
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
RAW_EXTENSIONS = {'.arw', '.cr2', '.cr3', '.nef', '.orf', '.raf', '.rw2', '.dng'}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}


def open_photo(path):
    """Read the oriented original as float32 sRGB without an 8-bit conversion."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            pixels = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB, output_bps=16, gamma=(2.4, 12.92))
        return pixels.astype(np.float32) / 65535.0
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        raise OSError(f'Cannot read {path}: {reader.errorString()}')
    if not image.colorSpace().isValid():
        image.setColorSpace(QColorSpace(QColorSpace.SRgb))
    image = image.convertedToColorSpace(QColorSpace(QColorSpace.SRgb), QImage.Format_RGBX32FPx4)
    if image.isNull():
        raise ValueError(f'Cannot convert the colour profile of {path} to sRGB')
    pixels = np.frombuffer(image.constBits(), np.float32).reshape(image.height(), image.bytesPerLine() // 4)
    return pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)[..., :3].copy()


def fit_photo(pixels, size):
    """Use Pillow's identical crop geometry on each floating-point channel."""
    channels = [np.asarray(ImageOps.fit(Image.fromarray(pixels[..., channel]), tuple(size),
        method=Image.Resampling.LANCZOS)) for channel in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0, 1)


def apply_tone_curve(pixels, points):
    """A monotone display-luminance curve; preserve RGB ratios and geometry."""
    from scipy.interpolate import PchipInterpolator
    points = np.asarray(points, np.float64)
    curve = PchipInterpolator(points[:, 0], points[:, 1])
    result = np.empty_like(pixels)
    weights = np.array([.2126, .7152, .0722], np.float32)
    for start in range(0, pixels.shape[0], 256):
        tile = pixels[start:start+256]
        luminance = tile @ weights
        mapped = np.clip(curve(luminance), 0, 1).astype(np.float32)
        ratios = np.ones_like(tile)
        np.divide(tile, luminance[..., None], out=ratios, where=luminance[..., None]>0)
        result[start:start+256] = np.clip(ratios * mapped[..., None], 0, 1)
    return result


def to_preview(pixels, max_edge=None):
    """Quantize only model inputs and previews, independently of the source RGB."""
    if max_edge is not None:
        height, width = pixels.shape[:2]
        factor = min(1.0, max_edge / max(width, height))
        if factor < 1:
            size = (max(1, round(width * factor)), max(1, round(height * factor)))
            pixels = fit_photo(pixels, size)
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255.0 + .5))


def open_rgb(path, *, max_edge=None):
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return to_preview(open_photo(path),max_edge)
    with Image.open(path) as opened:
        if max_edge is not None:
            opened.draft('RGB',(max_edge,max_edge))
        image = ImageOps.exif_transpose(opened)
        if max_edge is not None:
            image.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
        profile = image.info.get('icc_profile')
        if profile:
            return ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(BytesIO(profile)),
                ImageCms.createProfile('sRGB'), outputMode='RGB')
        return image.convert('RGB')


def save_image(image, path):
    """Save float RGB(A) as 16-bit PNG; keep PIL model images and masks at 8 bits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if isinstance(image, np.ndarray):
        pixels = np.uint16(np.clip(image, 0, 1) * 65535.0 + .5)
        if pixels.shape[2] == 3:
            pixels = np.dstack((pixels, np.full(pixels.shape[:2], 65535, np.uint16)))
            image_format = QImage.Format_RGBX64
        else:
            image_format = QImage.Format_RGBA64
        encoded = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], image_format)
        encoded.setColorSpace(QColorSpace(QColorSpace.SRgb))
        if not encoded.save(str(temporary), 'PNG'):
            raise OSError(f'Cannot save {path}')
    else:
        image.save(temporary, format='PNG', icc_profile=SRGB if image.mode in ('RGB', 'RGBA') else None)
    os.replace(temporary, path)
    return str(path.resolve())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save_jpeg(pixels, path):
    """Save a viewing copy alongside the 16-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    to_preview(pixels).save(temporary,format='JPEG',quality=95,subsampling=0,icc_profile=SRGB)
    os.replace(temporary,path)
    return str(path.resolve())


def fit_generated(image, size):
    """Centre-crop the model return and downsample; never enlarge RGB here."""
    if image.width < size[0] or image.height < size[1]:
        raise ValueError(f'Model output {image.size} does not cover the requested {tuple(size)} canvas')
    return ImageOps.fit(image, tuple(size), method=Image.Resampling.LANCZOS)


def preview_data_url(image, max_edge=1600):
    import base64
    if isinstance(image, np.ndarray):
        preview = to_preview(image, max_edge)
    else:
        preview = image.copy()
        preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    stream = BytesIO()
    preview.save(stream, format='JPEG', quality=93, subsampling=0, icc_profile=SRGB)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
