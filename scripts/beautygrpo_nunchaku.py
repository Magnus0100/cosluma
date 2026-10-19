"""Pinned SVDQuant CUDA execution and complete official Beauty LoRA binding."""
from pathlib import Path
import hashlib
import json

MODEL_REPO = 'nunchaku-ai/nunchaku-flux.1-kontext-dev'
MODEL_REVISION = '70dff7728491f3016e256137e8f7d87812af0b4f'
MODEL_FILE = 'svdq-int4_r32-flux.1-kontext-dev.safetensors'
MODEL_SHA = '5dceaae1666bd76402bb9e47a9431a51b2b528f921502ee92b90257717acbebb'
MODEL_BYTES = 6768310048
PACKAGE_VERSION = '1.2.1+cu12.8torch2.10'


def installed_model(config, *, installation=False):
    from asset_identity import sha256_file
    path = Path(config['config_path']).parent/'nunchaku'/MODEL_FILE
    receipt_path = path.with_suffix('.safetensors.download.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if (receipt.get('status') != 'verified' or receipt.get('sha256') != MODEL_SHA
            or receipt.get('expected_bytes') != MODEL_BYTES or path.stat().st_size != MODEL_BYTES):
        raise ValueError('Beauty requires the verified pinned Nunchaku Kontext INT4 model')
    stat = path.stat()
    # Bind the installation verification to the exact file stat; a subsequent
    # change requires a fresh explicit full SHA audit before execution.
    audit = path.with_suffix('.safetensors.installation.json')
    binding = {'sha256': MODEL_SHA, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
               'repo': MODEL_REPO, 'revision': MODEL_REVISION}
    if not audit.is_file() or json.loads(audit.read_text(encoding='utf8')) != binding:
        if not installation:
            raise ValueError('Nunchaku installation audit is missing or stale; run the explicit installation verification')
        if sha256_file(path) != MODEL_SHA:
            raise ValueError('Nunchaku model bytes differ from the official pinned SHA')
        temporary = audit.with_suffix('.tmp')
        temporary.write_text(json.dumps(binding, indent=2)+'\n', encoding='utf8');temporary.replace(audit)
    return path, binding


def load_transformer(path, *, device, torch_dtype, io_profile=None):
    """Use Rust's offset-ordered pread storage with the official model loader.

    Only the pinned mixin's storage/config construction hook is overridden.
    Quantized allocation, loadDict, module injection and forward stay in the
    inherited Nunchaku implementation. One CPU state dict owns at most the
    pinned file's payload; no second reader or whole-file bytes copy is made.
    """
    from importlib.metadata import version
    import time
    import torch

    if torch.device(device).type != 'cuda':
        raise ValueError('Beauty INT4 loader requires its explicit CUDA device')
    if version('nunchaku') != PACKAGE_VERSION:
        raise ValueError('Beauty loader requires the qualified Nunchaku package version')
    path = Path(path)
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise ValueError('Beauty loader requires the pinned local INT4 safetensors file')
    if torch_dtype != torch.bfloat16:
        raise ValueError('Beauty INT4 loader requires its fixed bfloat16 execution dtype')
    from safetensors import safe_open
    from nunchaku import NunchakuFluxTransformer2dModel

    class PreadTransformer(NunchakuFluxTransformer2dModel):
        @classmethod
        def _build_model(cls, model_path, **kwargs):
            started, cpu_started = time.perf_counter(), time.process_time()
            with safe_open(model_path, framework='pt', device='cpu', backend='pread') as opened:
                metadata = opened.metadata()
                state_dict = opened.get_tensors()
            if io_profile is not None:
                io_profile.append({'stage': 'rust_pread_all_tensors',
                    'seconds': time.perf_counter()-started,
                    'process_cpu_seconds': time.process_time()-cpu_started,
                    'backend': 'safetensors_pread', 'read_order': 'offset_keys',
                    'tensor_count': len(state_dict), 'file_bytes': path.stat().st_size})
            with torch.device('meta'):
                transformer = cls.from_config(json.loads(metadata['config'])).to(
                    kwargs.get('torch_dtype', torch.bfloat16))
            return transformer, state_dict, metadata

    return PreadTransformer.from_pretrained(str(path), offload=False,
                                           torch_dtype=torch_dtype, device=device)


def bind_lora(transformer, path, expected_keys):
    """Observe every source tensor used by the unmodified upstream converter."""
    import torch
    from safetensors.torch import load_file
    import nunchaku.lora.flux.nunchaku_converter as converter
    tensors = load_file(str(path), device='cpu')
    if sorted(tensors) != sorted(expected_keys):
        raise ValueError('Official Beauty LoRA tensor inventory changed')
    source_keys = {key.removeprefix('transformer.') for key in tensors}
    used = set()
    class Observed(dict):
        def __getitem__(self, key):
            result = super().__getitem__(key)
            if key in source_keys: used.add(key)
            return result
        def get(self, key, default=None):
            if key in self:
                return self[key]
            return default
        def pop(self, key, *default):
            if key in self and key in source_keys: used.add(key)
            return super().pop(key, *default)
    names = ('convert_to_nunchaku_transformer_block_lowrank_dict',
             'convert_to_nunchaku_flux_single_transformer_block_lowrank_dict')
    originals = {name: getattr(converter, name) for name in names}
    def wrap(original):
        def observed(*args, **kwargs):
            kwargs['extra_lora_dict'] = Observed(kwargs['extra_lora_dict'])
            return original(*args, **kwargs)
        return observed
    for name, original in originals.items(): setattr(converter, name, wrap(original))
    try:
        converted = converter.to_nunchaku(tensors, base_sd=transformer._quantized_part_sd,
                                          dtype=torch.bfloat16)
    finally:
        for name, original in originals.items(): setattr(converter, name, original)
    missing = sorted(source_keys-used)
    if missing:
        raise ValueError('Nunchaku did not consume every official Beauty tensor: '+', '.join(missing))
    if any(not bool(torch.isfinite(value).all()) for value in converted.values()):
        raise ValueError('Converted Beauty LoRA contains nonfinite values')
    transformer.update_lora_params(converted)
    transformer.set_lora_strength(1.)
    return {'schema': 'beautygrpo_nunchaku_lora_binding.v1', 'source_tensors': len(tensors),
            'consumed_source_tensors': len(used), 'unhandled_source_tensors': missing,
            'converted_tensors': len(converted), 'strength': 1.,
            'source_key_inventory_sha256': hashlib.sha256('\n'.join(sorted(source_keys)).encode()).hexdigest(),
            'converter': 'official_nunchaku_to_nunchaku', 'base_svd_rank': 32,
            'beauty_lora_rank': 32, 'source_lora_quantized': False}
