"""Offline SegFace face parsing; masks describe regions, never edit source pixels."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

CLASS_NAMES = (
    "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow", "r_brow",
    "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip", "hair", "eye_g",
    "hat", "ear_r", "neck_l",
)
GROUPS = {
    "skin_weight": (1, 2, 4, 5, 10),
    "features_gate": (6, 7, 8, 9, 11, 12, 13, 15),
    "eyes": (8, 9), "lips": (12, 13), "brows": (6, 7), "hair": (14,),
    "inner_mouth": (11,), "face_skin": (2, 10), "neck": (1,),
}
MODEL_SHA256 = "3cd535bfcfab4e5c67b7df7015370c2b18cb9a4044ecb00c7c1fa8555558c507"
MODEL_REVISION = "5e093b03c0523f7f32a9845bbbc75ecb027c8bee"
REPO_COMMIT = "9416628e7b5a3e4b9a1068b8467bdc0cf0be7a7d"
# Ordered topology from MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL.
FACE_OVAL = (10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
             152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109)
FACE_SKIN_OWNERS = (2, 4, 5, 10)
FEATURE_POLYGONS = (
    (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246),
    (263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466),
    (61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185),
    (46,53,52,65,55,70,63,105,66,107),
    (276,283,282,295,285,300,293,334,296,336),
)
SUPPORT_SCHEMA = "portrait_texture_support.v4"
from asset_identity import sha256_file

# ``numpy.savez_compressed`` hard-codes zlib level 6.  Region assets are large
# native-resolution masks (often 60 MiB compressed), and level 3 gives the same
# NPZ contract while shaving roughly a second from a cold parse on the current
# workstation.  Keep the archive format standard so old consumers can continue
# to open it with ``numpy.load``; this only changes the compression level.
NPZ_COMPRESSION_LEVEL = 3


def save_npz(path: Path, **arrays):
    """Write a standard NumPy NPZ with a lower-cost compression level.

    ``ZipFile.open`` streams each NPY member, avoiding a second in-memory copy
    of the full region bundle while retaining ``allow_pickle=False`` semantics.
    The helper is deliberately local to this worker so cache/result contracts
    stay unchanged and no alternate reader is introduced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=NPZ_COMPRESSION_LEVEL) as archive:
        for name, values in arrays.items():
            if not isinstance(name, str) or "/" in name or "\\" in name:
                raise ValueError("NPZ member names must be simple strings")
            with archive.open(name + ".npy", mode="w") as member:
                # write_array accepts the writable ZipExtFile and streams large
                # arrays in chunks; object arrays are rejected by NumPy itself.
                np.lib.format.write_array(member, np.asanyarray(values), allow_pickle=False)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def context_box(landmarks, width: int, height: int, scale: float = 2.0):
    """Square head/neck context before clipping; coordinates are native pixels."""
    if not landmarks or width <= 0 or height <= 0:
        raise ValueError("Face landmarks and a positive source size are required")
    xs, ys = zip(*[(float(point[0]), float(point[1])) for point in landmarks])
    if not all(math.isfinite(v) for v in (*xs, *ys)):
        raise ValueError("Non-finite face landmarks")
    face_width, face_height = max(xs) - min(xs), max(ys) - min(ys)
    if min(face_width, face_height) <= 1:
        raise ValueError("Degenerate face landmarks")
    side = max(face_width, face_height) * scale
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2 + face_height * .05
    return [max(0, math.floor(cx - side / 2)), max(0, math.floor(cy - side / 2)),
            min(width, math.ceil(cx + side / 2)), min(height, math.ceil(cy + side / 2))]


def letterbox_transform(width: int, height: int, resolution: int = 512):
    if min(width, height, resolution) <= 0:
        raise ValueError("Positive ROI dimensions required")
    scale = resolution / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (resolution - resized_width) // 2, (resolution - resized_height) // 2
    return {"native_roi_size": [width, height], "model_size": [resolution, resolution],
            "resized_size": [resized_width, resized_height], "padding_left_top": [left, top],
            "valid_model_box": [left, top, left + resized_width, top + resized_height],
            "scale_xy": [resized_width / width, resized_height / height]}


def detection_tiles(width: int, height: int):
    """Overlapping scale-independent crops for small faces in full-body photographs."""
    edge = max(1, round(min(width, height) * .6))
    step = max(1, round(edge * .65))
    def starts(length):
        values = list(range(0, max(1, length - edge + 1), step))
        if values[-1] != length - edge:
            values.append(length - edge)
        return values
    return [[x, y, min(x + edge, width), min(y + edge, height)]
            for y in starts(height) for x in starts(width)]


def _same_face(a, b):
    """Deduplicate overlapping tile detections using landmark bounding-box IoU."""
    def bbox(face):
        xs, ys = zip(*face["landmarks"])
        return min(xs), min(ys), max(xs), max(ys)
    aa, bb = bbox(a), bbox(b)
    intersection = max(0, min(aa[2], bb[2]) - max(aa[0], bb[0])) * max(0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
    area_a = (aa[2] - aa[0]) * (aa[3] - aa[1])
    area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
    return intersection / max(1, area_a + area_b - intersection) > .4


def detect_faces(input_path: Path, model_path: Path, output_dir: Path, max_faces: int):
    import numpy as np
    from PIL import Image, ImageOps
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not model_path.is_file():
        raise FileNotFoundError(f"Offline face landmarker missing: {model_path}")
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path),
                                      delegate=python.BaseOptions.Delegate.CPU),
        num_faces=max_faces, min_face_detection_confidence=.3, min_face_presence_confidence=.3,
    )
    start = time.perf_counter()
    attempts = []
    faces = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for longest_side in (1536, 3072):
            resized = image.copy()
            resized.thumbnail((longest_side, longest_side), Image.Resampling.LANCZOS)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=np.asarray(resized)))
            attempts.append({"size": list(resized.size), "face_count": len(result.face_landmarks)})
            if result.face_landmarks:
                for points in result.face_landmarks:
                    landmarks = [[point.x * width, point.y * height] for point in points]
                    box = context_box(landmarks, width, height)
                    faces.append({"native_box": box, "landmarks": landmarks})
                break
        if not faces:
            # BlazeFace inside the task is a short-range detector. Tile full-body
            # photos to improve face scale without guessed photo-specific ROIs.
            for box in detection_tiles(width, height):
                tile = image.crop(box)
                tile.thumbnail((768, 768), Image.Resampling.LANCZOS)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(tile)))
                attempts.append({"size": list(tile.size), "native_crop": box,
                                 "face_count": len(result.face_landmarks), "strategy": "overlapping_tiles"})
                for points in result.face_landmarks:
                    landmarks = [[point.x * (box[2] - box[0]) + box[0],
                                  point.y * (box[3] - box[1]) + box[1]] for point in points]
                    candidate = {"native_box": context_box(landmarks, width, height), "landmarks": landmarks}
                    if not any(_same_face(candidate, existing) for existing in faces):
                        faces.append(candidate)
                    if len(faces) >= max_faces:
                        break
                if len(faces) >= max_faces:
                    break
    faces.sort(key=lambda face: (face["native_box"][0], face["native_box"][1]))
    for index, face in enumerate(faces):
        face["id"] = f"face-{index + 1:02d}"
    result = {"source_native_size": [width, height], "source_sha256": file_sha256(input_path), "faces": faces,
              "method": "mediapipe_offline_multi_face_landmarks", "attempts": attempts,
              "model_path": str(model_path.resolve()), "model_sha256": file_sha256(model_path),
              "mediapipe_version": importlib.metadata.version("mediapipe"),
              "seconds": time.perf_counter() - start,
              "missing_faces_policy": "explicit_empty_regions_no_synthetic_face_mask"}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detections.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_segface(repo: Path, checkpoint: Path, device: str):
    if not (repo / "network" / "models" / "segface_celeb.py").is_file():
        raise FileNotFoundError(f"Official SegFace source is missing: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SegFace checkpoint is missing: {checkpoint}")
    import torch
    import torchvision.models
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU substitution")
    sys.path.insert(0, str(repo.resolve()))
    module = importlib.import_module("network.models.segface_celeb")
    original_factory = module.convnext_base
    try:
        # Full checkpoint restores every parameter; ImageNet initialization is redundant.
        module.convnext_base = lambda *args, **kwargs: torchvision.models.convnext_base(weights=None)
        model = module.SegFaceCeleb(input_resolution=512, model="convnext_base")
    finally:
        module.convnext_base = original_factory
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict_backbone" not in state:
        raise ValueError("Expected official SegFace checkpoint key state_dict_backbone")
    model.load_state_dict(state["state_dict_backbone"], strict=True)
    del state
    return model.eval().to(device)


def _restore_field(field, transform, nearest=False):
    import numpy as np
    from PIL import Image
    left, top, right, bottom = transform["valid_model_box"]
    array = field[top:bottom, left:right]
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize(tuple(transform["native_roi_size"]), mode))


def qualify_support(raw_skin_weight, labels, landmarks, native_box):
    """Limit texture retouch to facial geometry and visibly parsed facial parts.

    Softmax confidence cannot repair a wrong semantic class. A paper occluder
    may confidently become skin; oval geometry rejects outside-face mistakes,
    and missing lip/mouth evidence withholds the uncertain lower face. A single
    visible eye remains useful for profile poses and never rejects the face.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    raw = np.clip(np.asarray(raw_skin_weight, np.float32), 0, 1)
    height, width = raw.shape
    l, t, r, b = native_box
    if raw.shape != (b - t, r - l) or labels.shape != raw.shape:
        raise ValueError("Qualification arrays must use native_box ROI dimensions")
    points = np.asarray(landmarks, np.float64)
    if points.ndim != 2 or points.shape[0] <= max(max(FACE_OVAL), 466) or points.shape[1] != 2:
        return {"skin_weight": np.zeros_like(raw), "face_oval_support": np.zeros_like(raw),
                "visible_face_support": np.zeros_like(raw), "material_skin_support": np.zeros_like(raw),
                "features_gate": np.isin(labels, GROUPS["features_gate"]).astype(np.float32)}, {
                    "status": "unavailable", "reason": "canonical_face_geometry_missing",
                    "policy": "no invented support; colour editing remains independent"}
    if not np.isfinite(points).all():
        raise ValueError("Non-finite canonical face geometry")
    polygon = points[list(FACE_OVAL)] - [l, t]
    canvas = Image.new("L", (width, height))
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=255)
    oval = np.asarray(canvas) > 0
    face_height = float(np.ptp(polygon[:, 1]))
    feather = max(1., face_height * .015)
    face_oval_support = np.clip(distance_transform_edt(oval) / feather, 0, 1).astype(np.float32)
    # A nonzero softmax probability is present in virtually every pixel. Only
    # the winner class owns material support; soft probabilities modulate that
    # ownership and never invent skin on clothing/background/other features.
    material_skin = np.isin(labels, FACE_SKIN_OWNERS)
    material_skin_support = np.clip(distance_transform_edt(material_skin) / feather, 0, 1).astype(np.float32)
    feature_canvas = Image.new("L", (width, height))
    drawing = ImageDraw.Draw(feature_canvas)
    for indices in FEATURE_POLYGONS:
        drawing.polygon([tuple(point) for point in points[list(indices)] - [l, t]], fill=255)
    feature_owner = (np.asarray(feature_canvas) > 0) | np.isin(labels, GROUPS["features_gate"])
    feature_owner &= oval
    feature_gate = np.clip(1 - distance_transform_edt(~feature_owner) / feather, 0, 1).astype(np.float32)
    feature_gate *= oval
    lip_or_mouth_pixels = int((np.isin(labels, (11, 12, 13)) & oval).sum())
    eyes = {"l_eye": int((labels == 8).sum()), "r_eye": int((labels == 9).sum())}
    visible = np.ones_like(raw)
    eligibility = {
        "status": "full_visible_face" if lip_or_mouth_pixels else "visible_upper_face_only",
        "face_oval_intersection": True, "lip_or_mouth_pixels": lip_or_mouth_pixels,
        "material_ownership": "argmax skin/nose/ears only; nonwinning material exactly zero; inward feather",
        "feature_ownership": "local hard feature labels plus canonical eye/lip/brow landmark polygons",
        "eye_pixels": eyes, "single_eye_policy": "retain visible skin; no bilateral-eye requirement",
        "confidence_policy": "softmax does not override semantic/geometric disagreement",
        "feather_native_pixels": feather,
        "scope": "texture retouch support only; no face-wide colour protection",
    }
    if not lip_or_mouth_pixels:
        eye_a = (points[33] + points[133]) * .5
        eye_b = (points[362] + points[263]) * .5
        axis = eye_b - eye_a
        length = float(np.linalg.norm(axis))
        if length <= 1:
            visible[:] = 0
            eligibility.update(status="unavailable", reason="degenerate_eye_axis")
        else:
            down = np.array([-axis[1], axis[0]]) / length
            if np.dot(points[152] - (eye_a + eye_b) * .5, down) < 0:
                down *= -1
            # When mouth is occluded, nose/chin positions can be hallucinated
            # under the object as well. The eye axis uses the still visible
            # upper-face anchors and safely retains forehead/temple skin.
            cut_point = (eye_a + eye_b) * .5
            yy, xx = np.ogrid[t:b, l:r]
            signed_lower = (xx + .5 - cut_point[0]) * down[0] + (yy + .5 - cut_point[1]) * down[1]
            visible = np.clip(-signed_lower / feather, 0, 1).astype(np.float32)
            eligibility.update(
                reason="lip_and_mouth_visibility_unverified; uncertain lower face withheld",
                lower_face_cut={"native_point": cut_point.tolist(), "toward_chin_unit_vector": down.tolist(),
                                "criterion": "retain only pixels above visible upper-face eye axis; nose/lower face unverified"},
            )
    qualified = raw * material_skin_support * face_oval_support * visible
    eligibility["raw_support_sum"] = float(raw.sum())
    eligibility["qualified_support_sum"] = float(qualified.sum())
    eligibility["retained_support_fraction"] = float(qualified.sum() / max(1., raw.sum()))
    return {"skin_weight": qualified, "face_oval_support": face_oval_support,
            "visible_face_support": visible, "material_skin_support": material_skin_support,
            "features_gate": feature_gate}, eligibility


def _support_overlays(crop, raw, qualified, features_gate, directory):
    import numpy as np
    from PIL import Image
    preview = crop.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    source_rgb = np.asarray(preview, np.float32)
    overlays = []
    # Qualified view shows final texture eligibility after the local feature gate.
    for values, name in ((raw, "raw-support-overlay.jpg"),
                         (qualified * (1 - features_gate), "qualified-support-overlay.jpg")):
        mask = np.asarray(Image.fromarray(np.asarray(values, np.float32)).resize(preview.size, Image.Resampling.BILINEAR))
        overlay = np.uint8(np.clip(source_rgb * (1 - .5 * mask[..., None]) + np.array([255, 140, 35]) * .5 * mask[..., None], 0, 255))
        Image.fromarray(overlay).save(directory / name, quality=95)
        overlays.append(overlay)
    return overlays


def refine_existing(result_path: Path, output_dir: Path):
    """CPU-only qualification using immutable real SegFace output, no model rerun."""
    import numpy as np
    from PIL import Image, ImageOps

    started = time.perf_counter()
    original = json.loads(result_path.read_text(encoding="utf-8"))
    if output_dir.resolve() == result_path.resolve().parent:
        raise ValueError("Refinement needs a separate output directory; keep original parsing evidence")
    source_path = Path(original["source_path"])
    if file_sha256(source_path) != original["source_sha256"]:
        raise ValueError("Source differs from the real model parsing evidence")
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if list(image.size) != original["source_native_size"]:
        raise ValueError("Native source size changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    raw_review, qualified_review = np.asarray(preview).copy(), np.asarray(preview).copy()
    for region in original["regions"]:
        with np.load(region["npz_path"], allow_pickle=False) as parsed:
            fields = {key: parsed[key].copy() for key in parsed.files}
        raw = np.asarray(fields.get("raw_skin_weight", fields["skin_weight"]), np.float32)
        support, eligibility = qualify_support(raw, fields["labels"], region.get("landmarks", []), region["native_box"])
        fields["raw_skin_weight"] = raw.astype(np.float16)
        fields["raw_features_probability"] = fields.get("raw_features_probability", fields["features_gate"]).copy()
        fields.update({key: value.astype(np.float16) for key, value in support.items()})
        directory = output_dir / region["id"]
        directory.mkdir(parents=True, exist_ok=True)
        asset = directory / "regions.npz"
        save_npz(asset, **fields)
        for key in ("raw_skin_weight", "skin_weight", "face_oval_support", "visible_face_support", "material_skin_support", "features_gate"):
            Image.fromarray(np.uint8(np.clip(fields[key], 0, 1).astype(np.float32) * 255 + .5)).save(directory / f"{key}.png")
        overlays = _support_overlays(image.crop(region["native_box"]), raw, support["skin_weight"],
                                     support["features_gate"], directory)
        l, t, r, b = region["native_box"]
        pl, pt, pr, pb = (round(l * preview.width / image.width), round(t * preview.height / image.height),
                          round(r * preview.width / image.width), round(b * preview.height / image.height))
        if pr > pl and pb > pt:
            for target, overlay in zip((raw_review, qualified_review), overlays):
                target[pt:pb, pl:pr] = np.asarray(Image.fromarray(overlay).resize((pr - pl, pb - pt)))
        region.update(
            original_npz_path=region["npz_path"], npz_path=str(asset.resolve()),
            raw_skin_mask=str((directory / "raw_skin_weight.png").resolve()),
            skin_mask=str((directory / "skin_weight.png").resolve()),
            protection_mask=str((directory / "features_gate.png").resolve()),
            raw_support_overlay=str((directory / "raw-support-overlay.jpg").resolve()),
            qualified_support_overlay=str((directory / "qualified-support-overlay.jpg").resolve()),
            texture_eligibility=eligibility,
        )
    Image.fromarray(raw_review).save(output_dir / "raw-support-overlay.jpg", quality=95)
    Image.fromarray(qualified_review).save(output_dir / "qualified-support-overlay.jpg", quality=95)
    original["parsing_review_overlay"] = original.get("parsing_review_overlay", original["review_overlay"])
    original["review_overlay"] = str((output_dir / "qualified-support-overlay.jpg").resolve())
    original["qualified_support_overlay"] = original["review_overlay"]
    original["raw_support_overlay"] = str((output_dir / "raw-support-overlay.jpg").resolve())
    original["mask_contract"].update(
        raw_skin_weight="original sum probabilities of neck/skin/ears/nose",
        skin_weight="argmax skin/nose/ear material ownership feathered inward, times raw probability, face oval and visible-face support",
        features_gate="finite local hard-label/landmark geometry; exact zero away from features",
        raw_features_probability="original grouped feature softmax probabilities, never used as material ownership",
        support_schema=SUPPORT_SCHEMA,
        qualified_support_overlay="skin_weight times (1-features_gate), shown on untouched source for review",
        body_policy="face beauty support excludes neck/arms; separate body retouch needs its own qualification",
    )
    original["support_refinement"] = {
        "schema_version": SUPPORT_SCHEMA, "method": "argmax_material_ownership_face_oval_and_visible_eye_axis",
        "source_result": str(result_path.resolve()), "source_result_sha256": file_sha256(result_path),
        "device": "cpu", "model_rerun": False, "seconds": time.perf_counter() - started,
        "worker_sha256": file_sha256(Path(__file__)),
        "limitation": "Geometric qualification reduces known errors; it is not learned occluder recognition.",
    }
    (output_dir / "result.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    return original


def run(input_path: Path, repo: Path, checkpoint: Path, output_dir: Path, registry: Path,
        device: str, max_faces: int = 8, detections_path: Path | None = None):
    if not input_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Source image and real SegFace checkpoint must both exist")
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    started = time.perf_counter()
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != MODEL_SHA256:
        raise ValueError("Checkpoint SHA256 differs from the pinned official 512 ConvNext weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if detections_path:
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
    else:
        from current_model_registry import load_registry
        settings = load_registry(registry_path=registry)
        detector_python = Path(settings["face_python"])
        landmarker = Path(settings["face_model_path"])
        if not detector_python.is_file():
            raise FileNotFoundError(f"CPU MediaPipe runtime is missing: {detector_python}")
        process = subprocess.run([str(detector_python), str(Path(__file__).resolve()), "--detect-only",
                                  "--input", str(input_path), "--output-dir", str(output_dir),
                                  "--landmarker-model", str(landmarker), "--max-faces", str(max_faces)],
                                 timeout=180, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"Offline face detection failed: {process.stderr[-4000:]}")
        detections = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    if detections["source_native_size"] != list(image.size):
        raise ValueError("Detector coordinates do not match the source image size")
    source_hash = file_sha256(input_path)
    if detections.get("source_sha256") != source_hash:
        raise ValueError("Detection data belongs to a different source image")
    faces = detections["faces"]
    model = None
    model_load_seconds = 0.0
    if faces:
        load_start = time.perf_counter()
        model = load_segface(repo, checkpoint, device)
        model_load_seconds = time.perf_counter() - load_start
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    palette = np.array([
        [0, 0, 0], [255, 180, 80], [255, 210, 160], [80, 100, 180], [255, 160, 130],
        [230, 130, 110], [70, 210, 240], [70, 170, 220], [60, 230, 80], [80, 200, 40],
        [255, 220, 60], [180, 30, 60], [240, 80, 150], [240, 50, 100], [170, 80, 230],
        [200, 200, 200], [170, 170, 70], [200, 150, 230], [110, 230, 210],
    ], dtype=np.uint8)
    preview = image.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    review = np.asarray(preview).copy()
    raw_review, qualified_review = review.copy(), review.copy()
    regions = []
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    for face in faces:
        face_start = time.perf_counter()
        box = list(map(int, face["native_box"]))
        l, t, r, b = box
        if not (0 <= l < r <= image.width and 0 <= t < b <= image.height):
            raise ValueError(f"Invalid native face ROI: {box}")
        crop = image.crop(box)
        transform = letterbox_transform(crop.width, crop.height)
        # Letterbox padding uses ImageNet mean, which becomes zero after normalization.
        model_rgb = np.broadcast_to(mean, (512, 512, 3)).copy()
        resized = np.asarray(crop.resize(tuple(transform["resized_size"]), Image.Resampling.BICUBIC),
                             np.float32) / 255.0
        x0, y0, x1, y1 = transform["valid_model_box"]
        model_rgb[y0:y1, x0:x1] = resized
        tensor = torch.from_numpy(((model_rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor, None, None).float(), dim=1)[0].cpu().numpy()
        labels = _restore_field(probabilities.argmax(0).astype(np.uint8), transform, nearest=True)
        confidence = _restore_field(probabilities.max(0).astype(np.float32), transform)
        fields = {name: _restore_field(probabilities[list(classes)].sum(0), transform)
                  for name, classes in GROUPS.items()}
        fields["raw_skin_weight"] = fields["skin_weight"].copy()
        fields["raw_features_probability"] = fields["features_gate"].copy()
        qualified_fields, eligibility = qualify_support(fields["skin_weight"], labels, face.get("landmarks", []), box)
        fields.update(qualified_fields)
        native_source = np.asarray(crop)
        colour = palette[labels]
        overlay = native_source.copy()
        support = labels != 0
        overlay[support] = (native_source[support] * .55 + colour[support] * .45).astype(np.uint8)
        directory = output_dir / face["id"]
        directory.mkdir(parents=True, exist_ok=True)
        asset = directory / "regions.npz"
        save_npz(asset, labels=labels, confidence=confidence.astype(np.float16),
                 **{name: np.clip(field, 0, 1).astype(np.float16) for name, field in fields.items()})
        for name, values in fields.items():
            Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255 + .5)).save(directory / f"{name}.png")
        Image.fromarray(labels).save(directory / "labels.png")
        Image.fromarray(colour).save(directory / "labels-colour.png")
        Image.fromarray(overlay).save(directory / "review-overlay.jpg", quality=95)
        # This untouched ROI is review evidence only.  Keep PNG lossless but
        # use the cheaper level 3 encoder; it saves about 0.6s for a 4k ROI
        # while changing the bytes only (the pixels and evidence contract stay
        # identical).
        crop.save(directory / "source-roi.png", compress_level=3)
        support_overlays = _support_overlays(crop, fields["raw_skin_weight"], fields["skin_weight"],
                                             fields["features_gate"], directory)
        review_box = [round(l * preview.width / image.width), round(t * preview.height / image.height),
                      round(r * preview.width / image.width), round(b * preview.height / image.height)]
        pl, pt, pr, pb = review_box
        if pr > pl and pb > pt:
            review[pt:pb, pl:pr] = np.asarray(Image.fromarray(overlay).resize((pr - pl, pb - pt)))
            for target, support_overlay in zip((raw_review, qualified_review), support_overlays):
                target[pt:pb, pl:pr] = np.asarray(Image.fromarray(support_overlay).resize((pr - pl, pb - pt)))
        active = np.isin(labels, GROUPS["skin_weight"] + GROUPS["features_gate"])
        regions.append({"id": face["id"], "native_box": box, "landmarks": face.get("landmarks", []),
                        "masks_coordinate_space": "native_box_roi", "mask_size": list(crop.size),
                        "npz_path": str(asset.resolve()),
                        "skin_mask": str((directory / "skin_weight.png").resolve()),
                        "raw_skin_mask": str((directory / "raw_skin_weight.png").resolve()),
                        "protection_mask": str((directory / "features_gate.png").resolve()),
                        "texture_eligibility": eligibility,
                        "review_overlay": str((directory / "review-overlay.jpg").resolve()),
                        "raw_support_overlay": str((directory / "raw-support-overlay.jpg").resolve()),
                        "qualified_support_overlay": str((directory / "qualified-support-overlay.jpg").resolve()),
                        "transform": transform,
                        "parsing_confidence": {
                            "definition": "uncalibrated max softmax probability; not a correctness guarantee",
                            "mean_max_probability": float(confidence.mean()),
                            "skin_features_mean_max_probability": float(confidence[active].mean()) if active.any() else None,
                            "skin_features_fraction_below_0_6": float((confidence[active] < .6).mean()) if active.any() else None,
                            "label_pixel_counts": {name: int((labels == index).sum()) for index, name in enumerate(CLASS_NAMES)},
                        }, "seconds": time.perf_counter() - face_start})
    review_path = output_dir / "review-overlay.jpg"
    Image.fromarray(review).save(review_path, quality=95)
    Image.fromarray(raw_review).save(output_dir / "raw-support-overlay.jpg", quality=95)
    Image.fromarray(qualified_review).save(output_dir / "qualified-support-overlay.jpg", quality=95)
    result = {
        "schema_version": "portrait_regions.v1", "status": "ready" if regions else "no_faces_detected",
        "source_path": str(input_path.resolve()), "source_sha256": source_hash,
        "source_native_size": list(image.size), "regions": regions,
        "coordinates": "original source pixels, zero-based xy; bbox right/bottom exclusive",
        "mask_contract": {"space": "native_box_roi", "raw_skin_weight": "sum probabilities of neck/skin/ears/nose",
                          "skin_weight": "argmax skin/nose/ear ownership feathered inward, times raw probability, face oval and visible-face support",
                          "features_gate": "finite local hard-label/landmark geometry; exact zero away from features; texture gate only",
                          "raw_features_probability": "original grouped feature softmax probabilities; evidence only",
                          "support_schema": SUPPORT_SCHEMA,
                          "qualified_support_overlay": "skin_weight times (1-features_gate), shown on untouched source for review",
                          "colour_policy": "no colour editing or global face colour protection",
                          "native_detail": "512px learned parsing interpolated to ROI; no recovered native detail"},
        "model": {"name": "SegFace ConvNext-Base CelebAMask-HQ 512", "class_names": list(CLASS_NAMES),
                  "groups": {name: list(value) for name, value in GROUPS.items()},
                  "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                  "checkpoint_repo": "kartiknarayan/SegFace", "checkpoint_revision": MODEL_REVISION,
                  "repo_path": str(repo.resolve()), "repo_commit": REPO_COMMIT, "license": "MIT",
                  "model_source_sha256": file_sha256(repo / "network/models/segface_celeb.py"),
                  "initialization": "ConvNext weights=None, strict full SegFace checkpoint restoration",
                  "loading": "torch.load weights_only=True; strict state_dict_backbone"},
        "preprocessing": {"resize": "aspect-preserving PIL bicubic letterbox", "resolution": 512,
                          "mean": mean.tolist(), "std": std.tolist(), "padding": "ImageNet mean"},
        "detector": {key: value for key, value in detections.items() if key != "faces"},
        "runtime": {"python": sys.executable, "torch": torch.__version__,
                    "torchvision": importlib.metadata.version("torchvision"), "device": device,
                    "model_load_seconds": model_load_seconds,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if model and device == "cuda" else 0,
                    "seconds": time.perf_counter() - started},
        "parsing_review_overlay": str(review_path.resolve()),
        "review_overlay": str((output_dir / "qualified-support-overlay.jpg").resolve()),
        "raw_support_overlay": str((output_dir / "raw-support-overlay.jpg").resolve()),
        "qualified_support_overlay": str((output_dir / "qualified-support-overlay.jpg").resolve()),
        "worker_sha256": file_sha256(Path(__file__)),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--registry", type=Path,
                        default=Path(__file__).resolve().parents[1] / "providers/local-models.current.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-faces", type=int, default=8)
    parser.add_argument("--detections", type=Path, help="Existing detection JSON for repeatable research")
    parser.add_argument("--refine-existing", type=Path, help="CPU-only support qualification from real cached parsing")
    parser.add_argument("--detect-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--landmarker-model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.max_faces <= 32:
        parser.error("max-faces must be between 1 and 32")
    if args.refine_existing:
        if args.input:
            cached_source = json.loads(args.refine_existing.read_text(encoding="utf-8"))["source_path"]
            if args.input.resolve() != Path(cached_source).resolve():
                parser.error("--input must match the cached parsing source when refining")
        result = refine_existing(args.refine_existing, args.output_dir)
        print(json.dumps({"status": result["status"], "face_count": len(result["regions"]),
                          "result": str((args.output_dir / "result.json").resolve()),
                          "support_refinement": result["support_refinement"]}))
    elif args.detect_only:
        if not args.input:
            parser.error("detect-only needs --input")
        if not args.landmarker_model:
            parser.error("detect-only needs an offline landmarker model")
        result = detect_faces(args.input, args.landmarker_model, args.output_dir, args.max_faces)
        print(json.dumps({"face_count": len(result["faces"]), "seconds": result["seconds"]}))
    else:
        if not args.input:
            parser.error("normal inference needs --input")
        if not args.model_root and not (args.repo and args.checkpoint):
            parser.error("Provide --model-root or both --repo and --checkpoint")
        result = run(args.input, args.repo or args.model_root / "repo",
                     args.checkpoint or args.model_root / "checkpoints/convnext_celeba_512/model_299.pt",
                     args.output_dir, args.registry, args.device, args.max_faces, args.detections)
        print(json.dumps({"status": result["status"], "face_count": len(result["regions"]),
                          "result": str((args.output_dir / "result.json").resolve()), "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
