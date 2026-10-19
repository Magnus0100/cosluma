"""Erase supplied candidate occupancy with the local IOPaint BigLaMa model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from asset_identity import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", "--removal-mask", dest="removal_mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=root / ".cache/local-models/lama/big-lama.pt")
    parser.add_argument("--max-edge", type=int, default=1536)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    original = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    width, height = original.size
    original_rgb = np.asarray(original)
    mask = Image.open(args.removal_mask).convert("L").resize(original.size, Image.Resampling.NEAREST)
    removal = np.asarray(mask) > 127
    scale = min(1.0, args.max_edge / max(width, height))
    inference_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = np.asarray(original.resize(inference_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
    remove_small = np.asarray(mask.resize(inference_size, Image.Resampling.NEAREST)) > 127
    infer_height, infer_width = remove_small.shape
    padding = ((0, (-infer_height) % 8), (0, (-infer_width) % 8))
    rgb = np.pad(rgb, (*padding, (0, 0)), mode="reflect")
    remove_small = np.pad(remove_small, padding, mode="constant")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
    loaded = time.perf_counter()
    image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
    mask_tensor = torch.from_numpy(remove_small.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        prediction = model(image_tensor, mask_tensor)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    predicted = time.perf_counter()
    filled = prediction[0].permute(1, 2, 0).float().cpu().numpy()[:infer_height, :infer_width]
    filled_image = Image.fromarray(np.clip(filled * 255, 0, 255).astype(np.uint8)).resize(original.size, Image.Resampling.LANCZOS)
    # LaMa predicts all pixels; retain the immutable candidate everywhere outside the supplied removal mask.
    clean = original_rgb.copy()
    clean[removal] = np.asarray(filled_image)[removal]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = (args.output_dir / "clean-background.png").resolve()
    arrays_path = (args.output_dir / "clean-background.npz").resolve()
    Image.fromarray(clean).save(clean_path)
    Image.fromarray((removal * 255).astype(np.uint8)).save(args.output_dir / "removal-mask.png")
    np.savez_compressed(arrays_path, rgb_srgb=clean.astype(np.float32) / 255, removal_mask=removal)
    checkpoint_sha = sha256_file(args.checkpoint)
    result = {
        "clean_png": str(clean_path), "clean_npz": str(arrays_path),
        "input": str(args.input.resolve()), "removal_mask_input": str(args.removal_mask.resolve()),
        "size": [width, height], "inference_size": list(inference_size),
        "model": "BigLaMa TorchScript via IOPaint", "device": args.device,
        "provenance": {"model": "BigLaMa TorchScript", "repository": "https://github.com/advimman/lama",
                       "deployment_reference": "https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py",
                       "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
                       "official_md5": "e3aa4aaa15225a33ec84f9f4bc47e500", "license": "Apache-2.0",
                       "release": "2021-11-15", "precision": "float32"},
        "timings_seconds": {"load_and_import": round(loaded-started, 3), "inference": round(predicted-loaded, 3),
                            "total": round(time.perf_counter()-started, 3)},
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
        "limitations": ["LaMa is an established 2021 eraser, not a 2026 SOTA generative scene model.",
                        "Erases only the supplied mask; undetected reflections or accessories remain.",
                        "Large holes can have blurred or repeated textures; examine the region visible outside source alpha.",
                        "Preserves all candidate pixels outside removal mask exactly."],
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
