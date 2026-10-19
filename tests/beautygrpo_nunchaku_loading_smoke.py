"""Pread storage hook preserves official loading, metadata and failure semantics."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

import beautygrpo_nunchaku as backend


def main():
    calls=[]
    dtype=object()
    tensors={'z_int32': object(), 'm_bfloat16': object(), 'a_int8': object()}
    metadata={'config': json.dumps({'num_layers': 19, 'num_single_layers': 38}),
              'quantization_config': 'unchanged official SVDQuant config'}

    class FakeDevice:
        def __init__(self,selected):
            self.selected=selected
            self.type=str(selected).split(':')[0]
        def __enter__(self):
            calls.append(('device_context',self.selected))
            return self
        def __exit__(self,*args):pass

    class OfficialTransformer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(('official_from_pretrained', path, kwargs))
            result,state,meta=cls._build_model(path, **kwargs)
            assert state is tensors and meta is metadata
            # The storage adapter must inherit this entry unchanged, rather
            # than copy/reimplement quantization, injection or model forward.
            assert cls.from_pretrained.__func__ is OfficialTransformer.from_pretrained.__func__
            result._quantized_part_sd=state
            result.official_loadDict_and_injection=True
            return result

        @classmethod
        def from_config(cls, config):
            calls.append(('official_from_config', config))
            return cls()

        def to(self, selected):
            assert selected is dtype
            calls.append(('meta_dtype', selected))
            return self

    class Opened:
        def metadata(self): return metadata
        def get_tensors(self):
            calls.append(('rust_get_tensors',))
            return tensors
        def keys(self): raise AssertionError('Loader reverted to lexical per-tensor reads')
        def get_tensor(self,key): raise AssertionError('Loader bypassed Rust offset-ordered batch API')

    @contextmanager
    def safe_open(path, **kwargs):
        assert kwargs == {'framework':'pt','device':'cpu','backend':'pread'}
        calls.append(('safe_open',path,kwargs))
        yield Opened()

    torch=ModuleType('torch');torch.device=FakeDevice;torch.bfloat16=dtype
    safetensors=ModuleType('safetensors');safetensors.safe_open=safe_open
    nunchaku=ModuleType('nunchaku');nunchaku.NunchakuFluxTransformer2dModel=OfficialTransformer
    original_entry=OfficialTransformer.from_pretrained.__func__
    with tempfile.TemporaryDirectory(prefix='beauty-pread-hook-') as folder:
        path=Path(folder)/'pinned-fixture.safetensors';path.write_bytes(b'tiny fixture, not model weights')
        with patch.dict(sys.modules,{'torch':torch,'safetensors':safetensors,'nunchaku':nunchaku}), \
             patch('importlib.metadata.version',return_value=backend.PACKAGE_VERSION), \
             patch.object(backend,'MODEL_BYTES',path.stat().st_size):
            profile=[]
            result=backend.load_transformer(path,device='cuda:0',torch_dtype=dtype,io_profile=profile)
            assert isinstance(result,OfficialTransformer) and result.official_loadDict_and_injection
            assert result._quantized_part_sd is tensors
            assert profile[0]['read_order']=='offset_keys' and profile[0]['tensor_count']==3
            assert profile[0]['stage']=='rust_pread_all_tensors'
            assert profile[0]['seconds']>=0 and profile[0]['process_cpu_seconds']>=0
            assert OfficialTransformer.from_pretrained.__func__ is original_entry
            assert OfficialTransformer.__dict__.get('_build_model') is None
            assert nunchaku.NunchakuFluxTransformer2dModel is OfficialTransformer
            assert len([row for row in calls if row[0]=='safe_open'])==1
            assert ('device_context','meta') in calls
            reads=len([row for row in calls if row[0]=='safe_open'])
            for device in ('cpu','meta','mps'):
                try:backend.load_transformer(path,device=device,torch_dtype=dtype)
                except ValueError:pass
                else:raise AssertionError('Non-CUDA device read the quantized weights')
            assert len([row for row in calls if row[0]=='safe_open'])==reads
            with patch('importlib.metadata.version',return_value='unqualified'):
                try:backend.load_transformer(path,device='cuda:0',torch_dtype=dtype)
                except ValueError:pass
                else:raise AssertionError('Unpinned Nunchaku storage hook admitted')
            try:backend.load_transformer(path,device='cuda:0',torch_dtype=object())
            except ValueError:pass
            else:raise AssertionError('Changed execution dtype admitted')
            @contextmanager
            def fail_open(*args,**kwargs):
                raise OSError('storage read failed')
                yield
            with patch.object(safetensors,'safe_open',fail_open):
                try:backend.load_transformer(path,device='cuda:0',torch_dtype=dtype)
                except OSError:pass
                else:raise AssertionError('Storage failure silently retried on another backend')
            path.write_bytes(b'changed size')
            try:backend.load_transformer(path,device='cuda:0',torch_dtype=dtype)
            except ValueError:pass
            else:raise AssertionError('Changed local model size admitted')
    print(json.dumps({'ok':True,'official_from_pretrained_inherited':True,
        'scoped_hook_no_global_patch':True,'metadata_and_tensor_objects_preserved':True,
        'single_offset_ordered_pread':True,'package_size_dtype_and_read_failure_checked':True,
        'non_CUDA_rejected_before_storage_read':True,
        'model_inferences':0,'gpu_allocations':0,'actual_large_weight_reads':0}))


if __name__=='__main__':main()
