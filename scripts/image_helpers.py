"""Image I/O and comparison sheets."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
from PIL import Image, ImageDraw, ImageOps
from raw_support import open_rgb
RESAMPLE = Image.Resampling

def rgb_image(path: Path) -> Image.Image:
    """Open an image as RGB while respecting camera orientation metadata."""
    return open_rgb(path)


def save_png(im: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", compress_level=1)


def comparison_sheet(images: Iterable[Tuple[str, Image.Image]], path: Path):
    thumb_w = 420
    items = [(label, ImageOps.contain(im, (thumb_w, 630), RESAMPLE.BILINEAR)) for label, im in list(images)[:4]]
    thumb_h = max(thumb.height for _, thumb in items)
    rows = (len(items) + 1) // 2
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + 34) * rows), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(items):
        x, y = (i % 2) * thumb_w, (i // 2) * (thumb_h + 34)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        draw.text((x + 8, y + thumb_h + 8), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


