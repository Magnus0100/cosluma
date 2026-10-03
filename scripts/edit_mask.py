#!/usr/bin/env python3
"""Small interactive alpha-mask painter for the cosplay demo.

The normal mode opens a Tkinter window with a fit-to-window preview.  Left
mouse adds opaque alpha, right mouse erases alpha, and the mouse wheel changes
the brush diameter.  Painting is done against the full-resolution L image and
only the preview is downscaled, so the saved mask can be passed directly to
``run_demo.py --alpha``.

On machines without Tkinter or a display, ``--cli`` (and the ``--add`` /
``--erase`` point options) provides a small points painter with the same output
format.  Point coordinates are normalized by default; use ``--coords pixels``
for image-pixel coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageOps


RESAMPLE = Image.Resampling


def default_output(image: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "outputs" / "mask_editor" / image.stem / "alpha.png"


def load_source(path: Path) -> Image.Image:
    """Read an image and normalize orientation before returning RGB pixels."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_alpha(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("L", size, 0)
    with Image.open(path) as source:
        alpha = source.convert("L")
    if alpha.size != size:
        alpha = alpha.resize(size, RESAMPLE.LANCZOS)
    return alpha


def save_alpha(alpha: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha.convert("L").save(path, format="PNG", compress_level=1)


def parse_point(token: str) -> tuple[float, float]:
    try:
        x, y = token.split(",", 1)
        return float(x), float(y)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"point must be x,y (received {token!r})"
        ) from exc


def point_from_json(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [tuple(map(float, point)) for point in raw], []
    if not isinstance(raw, dict):
        raise ValueError("points JSON must be a list or an object with add/erase lists")

    def points(key: str) -> list[tuple[float, float]]:
        values = raw.get(key, [])
        return [tuple(map(float, point)) for point in values]

    return points("add"), points("erase")


def to_pixel(point: tuple[float, float], size: tuple[int, int], coords: str) -> tuple[float, float]:
    if coords == "normalized":
        return point[0] * (size[0] - 1), point[1] * (size[1] - 1)
    return point


def apply_points(
    alpha: Image.Image,
    add_points: Sequence[tuple[float, float]],
    erase_points: Sequence[tuple[float, float]],
    brush: float,
    coords: str,
) -> None:
    draw = ImageDraw.Draw(alpha)
    radius = max(1.0, float(brush) / 2.0)
    w, h = alpha.size

    def paint(points: Sequence[tuple[float, float]], value: int) -> None:
        for point in points:
            x, y = to_pixel(point, (w, h), coords)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)

    paint(add_points, 255)
    paint(erase_points, 0)


def cli_paint(args: argparse.Namespace, *, reason: str | None = None) -> int:
    image = load_source(args.image)
    alpha = load_alpha(args.alpha, image.size)
    adds = list(args.add or [])
    erases = list(args.erase or [])
    if args.points_file:
        file_adds, file_erases = point_from_json(args.points_file)
        adds.extend(file_adds)
        erases.extend(file_erases)
    if not adds and not erases:
        message = "CLI painting needs --add x,y, --erase x,y, or --points-file points.json"
        if reason:
            message = f"{reason}. {message}"
        print(message, file=sys.stderr)
        return 2
    apply_points(alpha, adds, erases, args.brush, args.coords)
    save_alpha(alpha, args.out)
    print(json.dumps({
        "path": str(args.out),
        "size": list(alpha.size),
        "brush_px": args.brush,
        "added": len(adds),
        "erased": len(erases),
        "backend": "points",
    }, ensure_ascii=False, indent=2))
    return 0


def launch_gui(args: argparse.Namespace) -> int:
    # Imports stay local so the points painter remains usable on headless hosts.
    import tkinter as tk
    from tkinter import TclError, ttk
    from PIL import ImageTk

    image = load_source(args.image)
    alpha = load_alpha(args.alpha, image.size)
    w, h = image.size
    scale = min(1.0, float(args.preview_side) / max(w, h))
    preview_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    base_preview = image.resize(preview_size, RESAMPLE.LANCZOS)
    preview_alpha = alpha.resize(preview_size, RESAMPLE.LANCZOS)

    root = tk.Tk()
    root.title(f"Mask painter — {args.image.name}")
    root.minsize(480, 360)
    controls = ttk.Frame(root, padding=6)
    controls.pack(fill="x")
    brush_value = tk.IntVar(value=max(1, int(args.brush)))
    status = tk.StringVar(value="左键增加 alpha，右键擦除；滚轮调整笔刷")

    ttk.Label(controls, text="笔刷(px)").pack(side="left")
    brush_box = ttk.Spinbox(controls, from_=1, to=10000, width=7, textvariable=brush_value)
    brush_box.pack(side="left", padx=(4, 10))
    ttk.Button(controls, text="保存 (S)", command=lambda: save_current()).pack(side="left")
    ttk.Button(controls, text="清空", command=lambda: clear_current()).pack(side="left", padx=5)
    ttk.Label(controls, textvariable=status, anchor="w").pack(side="left", padx=10, fill="x", expand=True)

    canvas = tk.Canvas(root, background="#202020", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    display_image: ImageTk.PhotoImage | None = None
    canvas_image = canvas.create_image(0, 0, anchor="nw")
    cursor_id: int | None = None
    last_preview: tuple[float, float] | None = None

    def preview_render() -> Image.Image:
        overlay_alpha = preview_alpha.point(lambda value: int(value * 0.40))
        overlay = Image.new("RGBA", preview_size, (255, 50, 40, 0))
        overlay.putalpha(overlay_alpha)
        return Image.alpha_composite(base_preview.convert("RGBA"), overlay).convert("RGB")

    def refresh() -> None:
        nonlocal display_image
        display_image = ImageTk.PhotoImage(preview_render())
        canvas.itemconfigure(canvas_image, image=display_image)
        canvas.configure(scrollregion=(0, 0, preview_size[0], preview_size[1]))

    def save_current() -> None:
        save_alpha(alpha, args.out)
        status.set(f"已保存：{args.out}")

    def clear_current() -> None:
        alpha.paste(0)
        preview_alpha.paste(0)
        refresh()
        status.set("已清空 alpha")

    def event_point(event: tk.Event) -> tuple[float, float]:
        # The image is anchored at (0,0); account for a possible canvas scroll.
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        return x, y

    def paint(event: tk.Event, add: bool) -> None:
        nonlocal last_preview
        px, py = event_point(event)
        if not (0 <= px < preview_size[0] and 0 <= py < preview_size[1]):
            return
        value = 255 if add else 0
        brush = max(1.0, float(brush_value.get()))
        preview_radius = max(1.0, brush * scale / 2.0)
        full_radius = brush / 2.0
        full_point = (px / scale, py / scale)
        draw_full = ImageDraw.Draw(alpha)
        draw_preview = ImageDraw.Draw(preview_alpha)
        if last_preview is None:
            last_preview = (px, py)
        draw_full.line((last_preview[0] / scale, last_preview[1] / scale, full_point[0], full_point[1]), fill=value, width=max(1, round(brush)))
        draw_preview.line((last_preview[0], last_preview[1], px, py), fill=value, width=max(1, round(brush * scale)))
        for draw, cx, cy, radius in ((draw_full, full_point[0], full_point[1], full_radius), (draw_preview, px, py, preview_radius)):
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=value)
        last_preview = (px, py)
        refresh()
        status.set(f"{('增加' if add else '擦除')}  brush={int(brush)}px  x={int(full_point[0])}, y={int(full_point[1])}")

    def end_stroke(_event: tk.Event) -> None:
        nonlocal last_preview
        last_preview = None

    def move_cursor(event: tk.Event) -> None:
        nonlocal cursor_id
        px, py = event_point(event)
        radius = max(1.0, float(brush_value.get()) * scale / 2.0)
        if cursor_id is not None:
            canvas.delete(cursor_id)
        cursor_id = canvas.create_oval(px - radius, py - radius, px + radius, py + radius, outline="#ffe066", width=1)

    def wheel(event: tk.Event) -> None:
        delta = 10 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -10
        brush_value.set(max(1, min(10000, brush_value.get() + delta)))
        move_cursor(event)

    canvas.bind("<ButtonPress-1>", lambda event: paint(event, True))
    canvas.bind("<B1-Motion>", lambda event: paint(event, True))
    canvas.bind("<ButtonRelease-1>", end_stroke)
    canvas.bind("<ButtonPress-3>", lambda event: paint(event, False))
    canvas.bind("<B3-Motion>", lambda event: paint(event, False))
    canvas.bind("<ButtonRelease-3>", end_stroke)
    canvas.bind("<Motion>", move_cursor)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<Button-4>", wheel)
    canvas.bind("<Button-5>", wheel)
    root.bind("<KeyPress-s>", lambda _event: save_current())
    root.bind("<KeyPress-S>", lambda _event: save_current())
    root.bind("<KeyPress-Escape>", lambda _event: root.destroy())
    refresh()
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="原图，用于预览并确定输出尺寸")
    parser.add_argument("--alpha", type=Path, help="已有 L/灰度 alpha；尺寸不符时自动缩放")
    parser.add_argument("--out", type=Path, default=None, help="保存的 L PNG（默认 outputs/mask_editor/<image>/alpha.png）")
    parser.add_argument("--brush", type=float, default=160, help="全分辨率笔刷直径，默认 160px")
    parser.add_argument("--preview-side", type=int, default=1400, help="预览最长边，默认 1400px")
    parser.add_argument("--cli", action="store_true", help="跳过窗口，用点坐标直接绘制")
    parser.add_argument("--coords", choices=("normalized", "pixels"), default="normalized", help="点坐标格式，默认 0..1 归一化")
    parser.add_argument("--points-file", type=Path, help="JSON：[[x,y],...] 或 {add:[[x,y]],erase:[[x,y]]}")
    parser.add_argument("--add", action="append", type=parse_point, default=[], metavar="X,Y", help="CLI 增加 alpha 的点；可重复")
    parser.add_argument("--erase", action="append", type=parse_point, default=[], metavar="X,Y", help="CLI 擦除 alpha 的点；可重复")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.image = args.image.resolve()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    args.alpha = args.alpha.resolve() if args.alpha else None
    args.points_file = args.points_file.resolve() if args.points_file else None
    args.out = (args.out or default_output(args.image)).resolve()
    if args.cli:
        return cli_paint(args)
    try:
        return launch_gui(args)
    except (ImportError, OSError) as exc:
        return cli_paint(args, reason=f"Tkinter 窗口不可用 ({exc})")
    except Exception as exc:
        # tkinter.TclError is not consistently an OSError across Python builds.
        # Only turn that known headless/display failure into the CLI fallback;
        # unexpected implementation errors should still be visible to the user.
        if exc.__class__.__name__ == "TclError":
            return cli_paint(args, reason=f"Tkinter 窗口不可用 ({exc})")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
