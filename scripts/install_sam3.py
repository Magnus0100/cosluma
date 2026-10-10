"""Verify the already downloaded current SAM3.1 static installation read-only.

This makes no network requests or GPU calls and never rewrites model manifests.
A current verification receipt is written only to an explicit external path.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from sam31_model_profile import CHECKPOINT, CODE_COMMIT, DETECTOR_TENSORS, DIRECTORY, IMAGE_PROFILE, model_identity


def verify_installation(directory=DIRECTORY):
    import torch
    directory = Path(directory).resolve()
    code = directory/'code'
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
    if commit != CODE_COMMIT or dirty:
        raise ValueError('SAM3.1 code must match the pristine pinned official commit')
    identity = model_identity(directory)
    if Path(sys.executable).resolve() != Path(identity['runtime']['python']).resolve():
        raise ValueError('Verification interpreter differs from the pinned SAM3.1 runtime')
    for package, expected in identity['runtime']['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('SAM3.1 runtime package differs: '+package)
    checkpoint = torch.load(directory/CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        checkpoint = checkpoint['model']
    if not checkpoint or any(not isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError('Checkpoint must be a safe tensor state dictionary')
    detector_count = sum(key.startswith('detector.') for key in checkpoint)
    if detector_count != DETECTOR_TENSORS:
        raise ValueError('SAM3.1 checkpoint detector tensor inventory is incomplete')
    return {'schema': 'sam31_static_installation_verification.current.v1', 'profile': IMAGE_PROFILE,
        'identity': identity, 'checkpoint_tensor_count': len(checkpoint),
        'detector_tensor_count': detector_count, 'weights_only_inspection': True,
        'model_files_modified': False, 'model_manifests_modified': False,
        'GPU_inference': False, 'network_requests': 0,
        'strict_architecture_and_real_inference_proof': 'separate_current_scene_worker_receipt'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(DIRECTORY))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.resolve().is_relative_to(Path(args.directory).resolve()):
        parser.error('Receipt must be outside the immutable model installation')
    receipt = verify_installation(args.directory)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'ok': True, 'profile': IMAGE_PROFILE, 'receipt': str(args.receipt.resolve())}))


if __name__ == '__main__':
    main()
