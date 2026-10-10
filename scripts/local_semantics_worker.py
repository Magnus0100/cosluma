"""Run a pinned Sapiens2 human parser on an oriented sRGB image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

CLASS_NAMES = [
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
    "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg",
    "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm",
    "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip",
    "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
]
MODEL_REVISION = "449b3c5335e6722bb94990abdd1aa6e612432f22"
CHECKPOINT_SHA256 = "b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=root / ".cache/local-models/sapiens2/repo")
    parser.add_argument("--model-profile", default="sapiens2_seg_0.4b",
                        choices=("sapiens2_seg_0.4b",))
    parser.add_argument("--residual-pass", action="store_true", help="Search outside the primary parser mask with neutral-filled full-image and fixed overlapping crops")
    args = parser.parse_args()
    started = time.perf_counter()
    from sapiens_model_profiles import verified_checkpoint
    specification, checkpoint_sha = verified_checkpoint(args.checkpoint, args.model_profile, 'segmentation')
    sys.path.insert(0, str(args.repo_dir.resolve()))
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as functional
    from sapiens.dense.models import init_model

    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = True
    image = Image.open(args.input).convert("RGB")
    width, height = image.size
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    config = args.repo_dir / specification['config']
    print("Loading", args.model_profile, "on", torch.cuda.get_device_name(), flush=True)
    model = init_model(config, str(args.checkpoint), device=device)
    loaded = time.perf_counter()
    data = model.data_preprocessor(model.pipeline(dict(img=bgr)))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(data["inputs"])
    if logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError('Pinned Sapiens2 parser must return the complete 29-class taxonomy')
    torch.cuda.synchronize()
    predicted = time.perf_counter()
    native_labels = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    # Resize probabilities before selecting classes; avoids blocky label upscaling.
    labels = functional.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = (args.output_dir / "labels.npz").resolve()
    np.savez_compressed(labels_path, labels=labels, native_labels=native_labels)
    Image.fromarray(labels).save(args.output_dir / "labels.png")
    groups = {
        "face_neck": [3], "left_arm": [7, 11], "right_arm": [16, 20],
        "left_hand": [6], "right_hand": [15],
        "skin": [3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21, 22],
        "garment": [1, 9, 10, 13, 18, 19, 23], "hair": [4],
        "foreground": list(range(1, 29)), "lips": [24, 25],
    }
    masks = {}
    for name, ids in groups.items():
        path = (args.output_dir / f"mask-{name}.png").resolve()
        Image.fromarray((np.isin(labels, ids) * 255).astype(np.uint8)).save(path)
        masks[name] = str(path)
    palette = np.random.default_rng(31415).integers(40, 255, (29, 3), dtype=np.uint8)
    palette[0] = 0
    rgb = np.asarray(image)
    overlay = rgb.copy()
    active = labels > 0
    overlay[active] = (0.55 * rgb[active] + 0.45 * palette[labels[active]]).astype(np.uint8)
    Image.fromarray(overlay).save(args.output_dir / "overlay.png")
    residual_path = None
    residual_pixels = 0
    if args.residual_pass:
        primary = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        neutral = bgr.copy()
        neutral[primary] = 127
        residual = np.zeros((height, width), dtype=bool)
        crop_width, crop_height = round(width * .6), round(height * .6)
        windows = [(0, 0, width, height)] + [(x, y, x + crop_width, y + crop_height)
                  for y in (0, height-crop_height) for x in (0, width-crop_width)]
        residual_body_ids = [3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21]
        for left, top, right, bottom in windows:
            crop = neutral[top:bottom, left:right]
            crop_data = model.data_preprocessor(model.pipeline(dict(img=crop)))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                crop_logits = model(crop_data["inputs"])
            probabilities = functional.interpolate(crop_logits.float(), size=crop.shape[:2], mode="bilinear", align_corners=False).softmax(1)
            confidence, classes = probabilities.max(1)
            crop_labels = classes.squeeze(0).cpu().numpy()
            crop_confidence = confidence.squeeze(0).cpu().numpy()
            residual[top:bottom, left:right] |= np.isin(crop_labels, residual_body_ids) & (crop_confidence >= .65)
        residual &= ~primary
        residual_path = (args.output_dir / "residual-human-mask.png").resolve()
        Image.fromarray((residual*255).astype(np.uint8)).save(residual_path)
        residual_overlay = np.asarray(image).copy()
        residual_overlay[residual] = (residual_overlay[residual]*.5 + np.array([255, 40, 40])*.5).astype(np.uint8)
        Image.fromarray(residual_overlay).save(args.output_dir / "residual-overlay.png")
        residual_pixels = int(residual.sum())
    taxonomy_sha = hashlib.sha256(json.dumps(CLASS_NAMES, separators=(',', ':')).encode()).hexdigest()
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    result = {
        "labels_path": str(labels_path), "labels_png": str((args.output_dir / "labels.png").resolve()),
        "original_input_size": [width, height],
        "label_size": [width, height], "native_label_size": list(native_labels.shape[::-1]),
        "class_names": CLASS_NAMES, "class_pixel_counts": np.bincount(labels.ravel(), minlength=29).tolist(),
        "masks": masks, "input": str(args.input.resolve()),
        "residual_human_mask": str(residual_path) if residual_path else None, "residual_human_pixels": residual_pixels,
        "residual_pass": {"enabled": args.residual_pass, "primary_fill": "neutral_gray_127",
                          "windows": "full_image_plus_fixed_2x2_60percent_overlap", "body_probability_threshold": .65},
        "model": specification['repository'], "model_revision": specification['revision'],
        "model_profile": args.model_profile, "taxonomy_sha256": taxonomy_sha,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "official_checkpoint_sha256": specification['checkpoint_sha256'],
        "repository": "https://github.com/facebookresearch/sapiens2",
        "repo_dir": str(args.repo_dir.resolve()), "license": "Sapiens2 License",
        "config": str(config.resolve()), "config_sha256": config_sha,
        "preprocess": {"input_color": "sRGB", "network_size": [768, 1024], "keep_ratio": False,
                       "mean_rgb": [123.675, 116.28, 103.53], "std_rgb": [58.395, 57.12, 57.375],
                       "label_restore": "bilinear_logits_then_argmax", "precision": "bfloat16_autocast"},
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "provenance": {"model": specification['repository'], "model_revision": specification['revision'],
                       "model_profile": args.model_profile, "taxonomy_sha256": taxonomy_sha,
                       "config_sha256": config_sha,
                       "checkpoint_sha256": checkpoint_sha, "repository": "https://github.com/facebookresearch/sapiens2",
                       "license": "Sapiens2 License", "device": device, "precision": "bfloat16_autocast"},
        "timings_seconds": {"load_and_import": round(loaded-started, 3), "inference": round(predicted-loaded, 3),
                            "total": round(time.perf_counter()-started, 3)},
        "limitations": ["Predicted masks may be imperfect.", "Eyes have no separate label; face masks do not independently protect eyes.",
                        "Labels describe the original input, without generated character pixels."],
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
