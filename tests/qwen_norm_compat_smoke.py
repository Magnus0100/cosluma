"""CPU-only managed install contracts; no Torch or external model installation."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import qwen_norm_compat as compat


FIXTURE = '''import torch
class OtherModel:
    def forward(self, x):
        return x
class Attention:
    def forward(self, q, k, pe, q_scale, k_scale):
        if True:
            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
        return q, k
'''


class Tensor:
    def __init__(self, value=None, *, floating=True):
        self.storage = value.storage if isinstance(value, Tensor) else object()
        self.shape = (128,)
        self.dtype = 'float32'
        self.floating = floating

    def is_floating_point(self):
        return self.floating


class GGMLTensor(Tensor):
    pass


def test_scale_contract():
    ns = {'torch': SimpleNamespace(Tensor=Tensor)}
    exec(compile(ast.parse(compat.HELPER), '<helper>', 'exec'), ns)
    convert = ns['_cosplay_qwen_plain_norm_scale']
    for qtype in (None, SimpleNamespace(name='F16'), SimpleNamespace(name='F32')):
        value = GGMLTensor()
        value.tensor_type = qtype
        result = convert(value)
        assert type(result) is Tensor and result.storage is value.storage
        assert result.shape == value.shape and result.dtype == value.dtype
    for qtype in ('Q6_K', 'BF16', 'FUTURE_FORMAT'):
        value = GGMLTensor()
        value.tensor_type = SimpleNamespace(name=qtype)
        assert convert(value) is value, 'Packed/unknown GGUF must not be reinterpreted'
    integer = GGMLTensor(floating=False)
    assert convert(integer) is integer
    ordinary = Tensor()
    assert convert(ordinary) is ordinary
    class OtherTensor(Tensor):
        pass
    other = OtherTensor()
    assert convert(other) is other


def test_install_contract():
    original = FIXTURE.encode('utf-8')
    expected = compat._patch_bytes(original)
    with tempfile.TemporaryDirectory(prefix='cosplay-qwen-norm-') as temporary:
        root = Path(temporary)
        target = root / compat.TARGET
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        unrelated = root / 'other_model.py'
        unrelated.write_bytes(b'unchanged\r\n')
        with patch.object(compat, 'SOURCE_SHA256', compat._sha(original)), \
             patch.object(compat, 'PATCHED_SHA256', compat._sha(expected)):
            receipt = compat.ensure_qwen_norm_compat(root)
            assert receipt['changed'] and compat.validate_qwen_norm_compat(root, receipt)
            assert target.read_bytes() == expected
            before = target.stat().st_mtime_ns
            again = compat.ensure_qwen_norm_compat(root)
            assert not again['changed'] and target.stat().st_mtime_ns == before
            assert compat.validate_qwen_norm_compat(root, again)
            assert unrelated.read_bytes() == b'unchanged\r\n'
            for key in ('profile', 'source_sha256', 'installed_sha256', 'target', 'schema'):
                assert not compat.validate_qwen_norm_compat(root, {**receipt, key: 'tampered'})
            target.write_bytes(expected + b'\n# external update\n')
            assert not compat.validate_qwen_norm_compat(root, receipt)
            try:
                compat.ensure_qwen_norm_compat(root)
            except RuntimeError as error:
                assert 'Unsupported Qwen runtime source' in str(error)
            else:
                raise AssertionError('Unsupported runtime was patched')
            assert target.read_bytes().endswith(b'# external update\n')
    # Both line-ending variants remain syntactically valid, with exactly two
    # inserted calls only inside Qwen Attention; other class math is untouched.
    for fixture in (FIXTURE, FIXTURE.replace('\n', '\r\n')):
        tree = ast.parse(compat._patch_bytes(fixture.encode()).decode())
        original_tree = ast.parse(fixture)
        assert ast.dump(tree.body[1]) == ast.dump(original_tree.body[1])
        calls = [node for node in ast.walk(tree.body[2]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == '_cosplay_qwen_plain_norm_scale']
        assert len(calls) == 2
    try:
        compat._patch_bytes(FIXTURE.replace('class Attention:', 'class DifferentModel:').encode())
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unknown model shape accepted')


def main():
    test_scale_contract()
    test_install_contract()
    print('Qwen norm compatibility: type/packed/noop/shape/install/idempotence/tamper contracts passed')


if __name__ == '__main__':
    main()
