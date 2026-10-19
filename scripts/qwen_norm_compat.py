"""Pinned Qwen-Image 2.1 GGUF norm compatibility; never a global Tensor hook.

The external deployment has no Git metadata.  Accept only the inspected source
bytes (or our exact resulting bytes), so an upstream change requires review.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import tempfile

PROFILE = 'qwen21_nonquantized_norm_plain_tensor.v1'
TARGET = Path('comfy/ldm/qwen_image21/model.py')
SOURCE_SHA256 = '0b3d6e36460da4ef5d4b52851e68391904cd11b32f95cd075b8549d1f86b448c'
PATCHED_SHA256 = '2c0b5c0df88979cd3e74f557cbc0c9095b7c3724736f4bbc0c53ec2ac9a100b2'
CALL = '            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)'
INSERT = ('            q_scale = _cosplay_qwen_plain_norm_scale(q_scale)\n'
          '            k_scale = _cosplay_qwen_plain_norm_scale(k_scale)\n')
HELPER = '''

# Cosplay compatibility: qwen21_nonquantized_norm_plain_tensor.v1
def _cosplay_qwen_plain_norm_scale(value):
    # GGUF wrappers belong to stored weights, not floating Q/K activations.
    # Do not dequantize, reinterpret, or alter packed/unknown GGUF values here.
    if not isinstance(value, torch.Tensor):
        return value
    ggml = any(cls.__name__ == "GGMLTensor" for cls in type(value).__mro__)
    if not ggml:
        return value
    qtype = getattr(value, "tensor_type", None)
    if qtype is not None and getattr(qtype, "name", None) not in ("F32", "F16"):
        return value
    if not value.is_floating_point():
        return value
    return torch.Tensor(value)
'''


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _patch_bytes(original):
    text = original.decode('utf-8')
    tree = ast.parse(text)
    attention = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == 'Attention'), None)
    forward = next((n for n in getattr(attention, 'body', [])
                    if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None)
    if forward is None or text.count(CALL) != 1 or '_cosplay_qwen_plain_norm_scale' in text:
        raise RuntimeError('Unsupported Qwen norm call structure; compatibility was not installed')
    call_line = text.splitlines().index(CALL) + 1
    if not forward.lineno <= call_line <= forward.end_lineno:
        raise RuntimeError('Qwen norm call is outside Attention.forward')
    newline = '\r\n' if '\r\n' in text else '\n'
    patched = text.replace(CALL, INSERT.replace('\n', newline) + CALL)
    patched += HELPER.replace('\n', newline)
    compile(patched, str(TARGET), 'exec')
    return patched.encode('utf-8')


def ensure_qwen_norm_compat(comfy_root):
    """Install before owned service launch; reject unreviewed runtime versions.

    The caller must establish that this deployment's service is stopped.  No
    import of Torch, CUDA, ComfyUI, or user models occurs in this installer.
    """
    root = Path(comfy_root).resolve()
    path = root / TARGET
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise RuntimeError('Qwen norm compatibility target is missing or outside deployment')
    original = path.read_bytes()
    before = _sha(original)
    if before == PATCHED_SHA256:
        return {'schema': 'qwen_norm_compatibility.v1', 'profile': PROFILE,
                'target': str(path), 'source_sha256': SOURCE_SHA256,
                'installed_sha256': before, 'changed': False, 'verified': True}
    if before != SOURCE_SHA256:
        raise RuntimeError('Unsupported Qwen runtime source for ' + PROFILE + ': ' + before)
    patched = _patch_bytes(original)
    after = _sha(patched)
    if after != PATCHED_SHA256:
        raise RuntimeError('Qwen compatibility output identity differs; refusing installation')
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.cosplay-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(patched)
            stream.flush()
            os.fsync(stream.fileno())
        # Do not overwrite an external updater's concurrent change.
        if path.read_bytes() != original:
            raise RuntimeError('Qwen source changed during compatibility installation')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if _sha(path.read_bytes()) != after:
        raise RuntimeError('Installed Qwen compatibility identity differs')
    return {'schema': 'qwen_norm_compatibility.v1', 'profile': PROFILE,
            'target': str(path), 'source_sha256': before,
            'installed_sha256': after, 'changed': True, 'verified': True}


def validate_qwen_norm_compat(comfy_root, receipt):
    """Read-only disk/receipt check; caller separately attests live process identity."""
    if not isinstance(receipt, dict):
        return False
    path = Path(comfy_root).resolve() / TARGET
    try:
        return (receipt.get('schema') == 'qwen_norm_compatibility.v1'
                and receipt.get('profile') == PROFILE
                and receipt.get('source_sha256') == SOURCE_SHA256
                and receipt.get('installed_sha256') == PATCHED_SHA256
                and receipt.get('verified') is True
                and Path(receipt.get('target', '')).resolve() == path.resolve()
                and path.resolve().is_relative_to(Path(comfy_root).resolve())
                and _sha(path.read_bytes()) == PATCHED_SHA256)
    except (OSError, ValueError, TypeError):
        return False
