"""Decode originals, fit canvases and save ordinary image files."""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColorSpace, QImage, QImageIOHandler, QImageReader

Image.MAX_IMAGE_PIXELS = None
QImageReader.setAllocationLimit(0)
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
RAW_EXTENSIONS = {'.arw', '.crw', '.cr2', '.cr3', '.nef', '.nrw', '.orf', '.raf',
    '.rw2', '.rwl', '.dng', '.pef', '.srw', '.sr2', '.srf'}
HEIF_EXTENSIONS = {'.heic', '.heif', '.hif'}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | {
    '.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.tif', '.tiff', '.webp', '.bmp'}


def _heif_image(path):
    """Decode the primary HEIF image once, retaining 10/12-bit camera data."""
    import pillow_heif
    decoded = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, hdr_to_16bit=True)
    pixels = np.asarray(decoded)
    if pixels.ndim == 2:
        image_format = QImage.Format_Grayscale16 if pixels.dtype == np.uint16 else QImage.Format_Grayscale8
    elif pixels.dtype == np.uint16:
        if pixels.shape[2] == 3:
            pixels = np.dstack((pixels, np.full(pixels.shape[:2], 65535, np.uint16)))
            image_format = QImage.Format_RGBX64
        else:
            image_format = QImage.Format_RGBA64
    else:
        image_format = QImage.Format_RGB888 if pixels.shape[2] == 3 else QImage.Format_RGBA8888
    image = QImage(pixels.data, decoded.size[0], decoded.size[1], pixels.strides[0], image_format).copy()
    profile = decoded.info.get('icc_profile')
    nclx = decoded.info.get('nclx_profile')
    if profile:
        image.setColorSpace(QColorSpace.fromIccProfile(profile))
    elif nclx and nclx['color_primaries'] != 2 and nclx['transfer_characteristics'] != 2:
        transfer = {1:QColorSpace.TransferFunction.Bt2020, 6:QColorSpace.TransferFunction.Bt2020,
            8:QColorSpace.TransferFunction.Linear, 13:QColorSpace.TransferFunction.SRgb,
            14:QColorSpace.TransferFunction.Bt2020, 15:QColorSpace.TransferFunction.Bt2020,
            16:QColorSpace.TransferFunction.St2084, 18:QColorSpace.TransferFunction.Hlg}[nclx['transfer_characteristics']]
        points = [QPointF(nclx['color_primary_'+name+'_x'], nclx['color_primary_'+name+'_y'])
            for name in ('white', 'red', 'green', 'blue')]
        image.setColorSpace(QColorSpace(*points, transfer))
    return image


def _read_image(path, max_edge=None):
    path = Path(path)
    if path.suffix.lower() in HEIF_EXTENSIONS:
        image = _heif_image(path)
        if max_edge is not None and max(image.width(), image.height()) > max_edge:
            image = image.scaled(max_edge, max_edge, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    else:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if max_edge is not None and size.isValid() and max(size.width(), size.height()) > max_edge:
            size.scale(max_edge, max_edge, Qt.KeepAspectRatio)
            reader.setScaledSize(size)
        image = reader.read()
        if image.isNull():
            raise OSError(f'Cannot read {path}: {reader.errorString()}')
    if not image.colorSpace().isValid():
        if image.format() == QImage.Format_CMYK8888:
            image = image.convertToFormat(QImage.Format_RGB888)
        image.setColorSpace(QColorSpace(QColorSpace.SRgb))
    return image


def image_size(path):
    """Read a model image's oriented dimensions without decoding its RGB."""
    reader = QImageReader(str(path))
    size = reader.size()
    if not size.isValid():
        raise OSError(f'Cannot read {path}: {reader.errorString()}')
    if reader.transformation() & QImageIOHandler.TransformationRotate90:
        size.transpose()
    return size.width(),size.height()


def open_photo(path):
    """Read the oriented original as float32 sRGB without an 8-bit conversion."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            pixels = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB, output_bps=16, gamma=(2.4, 12.92))
        return pixels.astype(np.float32) / 65535.0
    image = _read_image(path)
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
    encoded = np.empty(pixels.shape,np.uint8)
    for start in range(0,pixels.shape[0],256):
        encoded[start:start+256] = np.clip(pixels[start:start+256],0,1)*255.0+.5
    return Image.fromarray(encoded)


def display_image(pixels, *, max_edge=None):
    pixels = np.asarray(to_preview(pixels, max_edge))
    image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format_RGB888).copy()
    image.setColorSpace(QColorSpace(QColorSpace.SRgb))
    return image


def open_display(path, *, max_edge=None):
    """Read oriented, colour-managed display pixels; never upscale a preview."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return display_image(open_photo(path), max_edge=max_edge)
    image = _read_image(path, max_edge)
    image = image.convertedToColorSpace(QColorSpace(QColorSpace.SRgb), QImage.Format_RGB888)
    if image.isNull():
        raise ValueError(f'Cannot convert the colour profile of {path} to sRGB')
    return image


def open_rgb(path, *, max_edge=None):
    image = open_display(path, max_edge=max_edge)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return Image.fromarray(pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy())


def save_image(image, path):
    """Save float RGB(A) as 16-bit PNG; keep PIL model images and masks at 8 bits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        if isinstance(image, np.ndarray):
            pixels = np.empty((*image.shape[:2],4),np.uint16)
            for start in range(0,image.shape[0],256):
                pixels[start:start+256,:,:image.shape[2]] = np.clip(image[start:start+256],0,1)*65535.0+.5
            if image.shape[2] == 3:
                pixels[...,3] = 65535
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
    finally:
        temporary.unlink(missing_ok=True)
    return str(path.resolve())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_jpeg(pixels, path):
    """Save a viewing copy alongside the 16-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    try:
        to_preview(pixels).save(temporary,format='JPEG',quality=95,subsampling=0,icc_profile=SRGB)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
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
