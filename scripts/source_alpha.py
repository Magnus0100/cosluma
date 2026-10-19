#!/usr/bin/env python3
"""Source-only subject silhouette from the current BiRefNet ONNX model."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from cache_store import ContentCache
from raw_support import RGB_DECODE_VERSION, is_raw


def navigation_crop_regions(silhouette: Image.Image) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, Any]]:
    """Broad body-relative review windows; no pose/landmark inference or quality gate."""
    bbox = silhouette.point(lambda value: 255 if value >= 128 else 0).getbbox()
    method = "source_silhouette_bbox_relative"
    if bbox is None:
        bbox = (0, 0, silhouette.width, silhouette.height)
        method = "empty_silhouette_full_frame_relative"
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    relative = {
        "hair_halo": (-.10, -.05, 1.10, .28),
        "face_eyes": (.12, -.02, .88, .23),
        "hands": (-.15, .25, 1.15, .68),
        "garment_texture": (-.10, .19, 1.10, .93),
    }
    boxes = {name: (max(0, round(left + x0 * width)), max(0, round(top + y0 * height)),
                    min(silhouette.width, round(left + x1 * width)), min(silhouette.height, round(top + y1 * height)))
             for name, (x0, y0, x1, y1) in relative.items()}
    boxes["background_light"] = (0, 0, silhouette.width, silhouette.height)
    return boxes, {"method": method, "source_bbox": list(bbox), "silhouette_bbox_threshold": 128,
                   "regions_pixels": {key: list(value) for key, value in boxes.items()},
                   "relative_body_regions": {key: list(value) for key, value in relative.items()},
                   "purpose": "visual_navigation_only", "landmark_detection": False,
                   "note": "Broad head/arms-bag/garment windows relative to source silhouette; inspect full frame for unusual poses."}


def _selected_runtime_root(root: Path, *, registry_path=None):
    from current_model_registry import load_registry
    return Path(load_registry(
        root,
        registry_path=registry_path,
        required_keys=('birefnet_model_path', 'onnx_runtime_root'),
        required_profiles=(),
    )['onnx_runtime_root']).resolve()


def _runtime(root: Path, *, registry_path=None):
    """Load the ONNX runtime from the selected route registry only.

    The desktop interpreter is deliberately not a model runtime boundary:
    importing whichever ``onnxruntime`` happens to be on ``sys.path`` can mix
    providers and native DLLs across installations.  Every source silhouette
    inference therefore binds the route's explicit ONNX runtime directory.
    """
    runtime_root = _selected_runtime_root(root, registry_path=registry_path)
    for name, module in tuple(sys.modules.items()):
        module_file = getattr(module, '__file__', None)
        if not name.startswith('onnxruntime.capi.') or not isinstance(module_file, (str, os.PathLike)):
            continue
        if not module_file:
            continue
        native_path = Path(module_file).resolve()
        if 'pybind' not in name and native_path.suffix.lower() not in ('.pyd', '.so', '.dll'):
            continue
        if not native_path.is_relative_to(runtime_root):
            raise RuntimeError('ONNX native module '+name+' is already loaded from '+str(native_path)
                +'; use an isolated process for selected runtime '+str(runtime_root))
    sys.modules.pop('onnxruntime', None)
    optional = str(runtime_root)
    sys.path.insert(0, optional)
    try:
        return importlib.import_module('onnxruntime')
    finally:
        sys.path.remove(optional)


class SilhouetteSessionScope:
    """Lazily own one selected session for a source/reference pair."""

    _providers = ('CPUExecutionProvider',)
    _options = (('log_severity_level', 3),)

    def __init__(self):
        self._session = self._ort = self._binding = None
        self._dll_directories = []
        self._cuda_libraries = []

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def close(self):
        self._session = self._ort = self._binding = None
        self._cuda_libraries.clear()
        for directory in self._dll_directories:
            directory.close()
        self._dll_directories.clear()

    def _prepare_cuda_libraries(self, runtime_root, ort):
        if os.name != 'nt':
            return
        # ORT 1.30 uses CUDA 13; the chosen worker's Torch uses CUDA 12.8.
        # Bind the installed matching toolkit and cuDNN explicitly without
        # importing Torch into the desktop or changing its CUDA installation.
        cuda_major = str(getattr(ort, 'cuda_version', '')).split('.')[0]
        toolkit = os.environ.get('CUDA_PATH_V'+cuda_major+'_0') or os.environ.get('CUDA_PATH')
        directories = [runtime_root/'torch'/'lib']
        if toolkit:
            directories += [Path(toolkit)/'bin'/'x64', Path(toolkit)/'bin']
        for directory in directories:
            if directory.is_dir():
                self._dll_directories.append(os.add_dll_directory(str(directory)))
        library_root = runtime_root/'torch'/'lib'
        if (library_root/'cudnn64_9.dll').is_file():
            import ctypes
            # cuDNN loads its components by bare filename. Preload each one
            # explicitly so its internal loader does not depend on PATH or
            # on a prior Torch import (which preloads the same components).
            for name in ('cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll',
                         'cudnn_graph64_9.dll', 'cudnn_ops64_9.dll',
                         'cudnn_cnn64_9.dll', 'cudnn_adv64_9.dll', 'cudnn_engines_precompiled64_9.dll',
                         'cudnn_engines_runtime_compiled64_9.dll', 'cudnn_heuristic64_9.dll', 'cudnn64_9.dll'):
                path = library_root/name
                if path.is_file():
                    self._cuda_libraries.append(ctypes.CDLL(str(path)))
                elif name.startswith('cudnn'):
                    raise RuntimeError('Selected BiRefNet CUDA runtime is missing cuDNN component: '+str(path))

    def get(self, root: Path, model: Path, *, registry_path=None, model_hash=None):
        runtime_root = _selected_runtime_root(root, registry_path=registry_path)
        runtime_key = os.path.normcase(str(runtime_root))
        if self._binding is not None and self._binding[0] != runtime_key:
            raise RuntimeError('Cannot change ONNX runtime within one silhouette pair')
        ort = self._ort or _runtime(root, registry_path=registry_path)
        providers = (('CUDAExecutionProvider', 'CPUExecutionProvider')
                     if 'CUDAExecutionProvider' in ort.get_available_providers() else self._providers)
        cuda_options = (('arena_extend_strategy', 'kSameAsRequested'),
                        ('cudnn_conv_algo_search', 'HEURISTIC')) if providers[0] == 'CUDAExecutionProvider' else ()
        binding = (runtime_key, os.path.normcase(str(model.resolve())),
                   model_hash or sha256_file(model), providers, cuda_options, self._options,
                   'other_session_options_use_selected_runtime_defaults')
        if self._session is None or self._binding != binding:
            self.close()
            options = ort.SessionOptions()
            for name, value in self._options:
                setattr(options, name, value)
            if cuda_options:
                self._prepare_cuda_libraries(runtime_root, ort)
            selected = [(providers[0], dict(cuda_options)), providers[1]] if cuda_options else list(providers)
            session = ort.InferenceSession(str(model), sess_options=options, providers=selected)
            if cuda_options:
                session.disable_fallback()
                if session.get_providers()[0] != 'CUDAExecutionProvider':
                    raise RuntimeError('Selected BiRefNet CUDA provider failed to initialize; CPU fallback refused')
            self._ort, self._session, self._binding = ort, session, binding
        return self._ort, self._session


def _infer(image: Image.Image, model: Path, backend: str, root: Path, *, registry_path=None,
           session_scope=None, model_hash=None) -> tuple[Image.Image, dict[str, Any]]:
    with (nullcontext(session_scope) if session_scope is not None else SilhouetteSessionScope()) as scope:
        ort, session = scope.get(root, model, registry_path=registry_path, model_hash=model_hash)
        input_info = session.get_inputs()[0]
        size = 1024
        pixels = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        if backend != "birefnet_lite_onnx":
            raise ValueError("Unsupported silhouette inference backend: "+str(backend))
        pixels = (pixels - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
        tensor = np.transpose(pixels, (2, 0, 1))[None].astype(np.float16 if input_info.type == "tensor(float16)" else np.float32)
        output_info = session.get_outputs()[0]
        score = np.asarray(session.run([output_info.name], {input_info.name: tensor})[0], dtype=np.float32).squeeze()
        if score.ndim != 2:
            raise RuntimeError(f"unexpected silhouette output dimensions: {score.shape}")
        if not np.isfinite(score).all():
            raise RuntimeError('BiRefNet returned nonfinite silhouette logits')
        score = 1. / (1. + np.exp(-np.clip(score, -80, 80)))
        postprocess = "sigmoid logits, uint8 rounded, bilinear resize to source"
        mask = Image.fromarray(np.uint8(np.clip(score, 0, 1) * 255 + .5), mode="L")
        return mask, {"onnxruntime": ort.__version__, "execution_provider": session.get_providers()[0],
                      "session_providers": session.get_providers(),
                      "input_name": input_info.name, "input_dtype": str(tensor.dtype), "input_shape": list(tensor.shape),
                      "output_name": output_info.name, "output_shape": list(score.shape),
                      "preprocess": {"color": "RGB", "resize": [size, size], "interpolation": "bilinear",
                                     "range": "0..1", "mean": [.485, .456, .406],
                                     "std": [.229, .224, .225]},
                      "postprocess": postprocess}


def source_silhouette(root: Path, source: Path, image: Image.Image, *, sample: str = "",
                      backend: str = "auto", source_hash: str | None = None, registry_path=None,
                      session_scope=None) -> tuple[Image.Image, dict[str, Any]]:
    """Return geometry-only mask. Trust edits/structure diagnostics cannot alter it."""
    root = root.resolve(); source = source.resolve()
    source_hash = source_hash or sha256_file(source)
    base = {"schema_version": "source_silhouette.v1", "role": "source_subject_geometry",
            "requested_backend": backend, "source": str(source), "source_sha256": source_hash,
            "oriented_size": list(image.size), "candidate_used": False, "cache_hit": False,
            "source_decode": {"adapter": "raw_support.open_rgb", "output": "oriented RGB uint8",
                              "color_space": "sRGB", "decode_version": RGB_DECODE_VERSION,
                              "color_management": image.info.get("color_management"),
                              "source_profile_name": image.info.get("source_profile_name"),
                              "camera_wb": True if is_raw(source) else None,
                              "no_auto_bright": True if is_raw(source) else None}}
    if backend not in ('auto', 'birefnet'):
        raise ValueError('Unsupported silhouette backend: '+str(backend))
    override = os.environ.get('COSPLAY_ALPHA_MODEL', '').strip()
    if override:
        model = Path(override)
    else:
        from current_model_registry import load_registry
        model = Path(load_registry(
            root,
            registry_path=registry_path,
            required_keys=('birefnet_model_path', 'onnx_runtime_root'),
            required_profiles=(),
        )['birefnet_model_path'])
    model = model.expanduser()
    if not model.is_absolute():
        model = root/model
    if not model.is_file():
        raise RuntimeError('Current BiRefNet model is unavailable: '+str(model))
    model_backend = 'birefnet_lite_onnx'
    model_hash = sha256_file(model)
    cache = ContentCache(root, max_bytes=int(os.environ.get("COSPLAY_CACHE_MAX_BYTES", str(2 * 1024 ** 3))),
                         software_version="source-silhouette-v1")
    params = {"model_sha256": model_hash, "backend": model_backend, "preprocess_version": "srgb-bilinear-v2", "decode_version": RGB_DECODE_VERSION}
    key = cache.make_key(source_hash, "source_silhouette", params=params, dimensions=list(image.size))
    cached = cache.get(key, "silhouette.png"); metadata_path = cache.get(key, "provenance.json")
    if cached and metadata_path:
        with Image.open(cached) as existing:
            silhouette = existing.convert("L").resize(image.size, Image.Resampling.BILINEAR)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return silhouette, {**metadata, **base, "model": str(model.resolve()), "cache_hit": True, "cache_key": key}
    started = time.perf_counter()
    native, inference = _infer(image, model, model_backend, root, registry_path=registry_path,
                               session_scope=session_scope, model_hash=model_hash)
    metadata = {**base, **inference, "backend": model_backend, "model": str(model.resolve()),
                "model_sha256": model_hash,
                "model_source": "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX",
                "declared_upstream_code_license": "MIT",
                "weight_license_verification": "not_verified_locally",
                "elapsed_seconds": round(time.perf_counter() - started, 3), "cache_key": key}
    cache_record = {"source_sha256": source_hash, "operation": "source_silhouette", "params": params, "dimensions": list(image.size)}
    cache.put_image(key, native, artifact="silhouette.png", metadata=cache_record)
    cache.put_bytes(key, "provenance.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), metadata=cache_record)
    return native.resize(image.size, Image.Resampling.BILINEAR), metadata
