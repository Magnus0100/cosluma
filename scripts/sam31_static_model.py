"""Shared, complete SAM3.1 static Tri detector construction; no tracker."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

from sam31_model_profile import CHECKPOINT, DETECTOR_TENSORS, PREPROCESS, image_state, model_identity


def load_model(directory, device, *, identity=None):
    import torch
    from sam3 import model_builder as b
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_image_processor import Sam3Processor
    directory = Path(directory)
    identity = identity if identity is not None else model_identity(directory)
    if Path(sys.executable).resolve() != Path(identity['runtime']['python']).resolve():
        raise ValueError('SAM3.1 worker interpreter differs from the pinned runtime')
    for package, expected_version in identity['runtime']['packages'].items():
        if importlib.metadata.version(package) != expected_version:
            raise ValueError('SAM3.1 runtime package version differs: '+package)
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('SAM3.1 static inference requires the explicit CUDA runtime')
    # Exact official Multiplex detector block. The ordinary SAM3 image builder
    # omits four 3.1 convolution tensors and must never be substituted here.
    backbone = SAM3VLBackboneTri(scalp=0,
        visual=b._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=False),
        text=b._create_text_encoder(str(directory/'code/sam3/assets/bpe_simple_vocab_16e6.txt.gz')))
    model = Sam3MultiplexDetector(num_feature_levels=1, backbone=backbone,
        transformer=b._create_sam3_transformer(use_fa3=False),
        segmentation_head=b._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None, input_geometry_encoder=b._create_geometry_encoder(),
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=b._create_dot_product_scoring(), supervise_joint_box_scores=True,
        is_multiplex=True).eval()
    expected = set(model.state_dict())
    if len(expected) != DETECTOR_TENSORS:
        raise ValueError('SAM3.1 static detector architecture differs from the pinned checkpoint')
    checkpoint = torch.load(directory/CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    selected = image_state(checkpoint, expected)
    loaded = model.load_state_dict(selected, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('SAM3.1 static detector did not load completely')
    model._cosplay_strict_checkpoint = {'loaded': True, 'detector_tensor_count': len(selected),
        'missing_keys': [], 'unexpected_keys': [], 'partially_initialized': False}
    del checkpoint, selected
    model = model.eval().to(device)
    return model, Sam3Processor(model, device=device, resolution=PREPROCESS['resolution'],
                               confidence_threshold=PREPROCESS['instance_confidence_threshold'])
