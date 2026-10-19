"""Actual proxy writer calls retain every consumer member and integer owner."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_array_storage import save_proxy_arrays


def main():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    shape = (7, 9)
    weights = np.arange(7*9*10, dtype=np.float32).reshape(*shape, 10)/1000
    weights[0, 0, 0] = np.float32(-0.0)
    owners = (np.arange(63, dtype=np.int32).reshape(shape) % 29)
    colour = weights[..., :6]
    scene = weights[..., :4]
    baseline = np.asfortranarray(weights[..., :3])
    cases = [('evaluate_reference_colour.py', 'source-weights.npz',
              {'values': weights, 'chroma_values': weights[:, ::-1]}, {'weights': weights, 'chroma_weights': weights[:, ::-1]}),
             ('reference_colour_blend.py', 'reference-colour-layer.npz',
              {'values': colour, 'owners': owners}, {'field': colour, 'owners': owners}),
             ('reference_scene_balance.py', 'scene-balance-layer.npz',
              {'values': scene, 'baseline_rgb': baseline, 'owners': owners},
              {'field': scene, 'baseline': baseline, 'owners': owners})]
    with TemporaryDirectory(prefix='reference-proxy-storage-') as temporary:
        root = Path(temporary)
        renderer = ast.parse((scripts/'render_reference_colour_composite.py').read_text(encoding='utf8'))
        owner_reader = next(node for node in renderer.body if isinstance(node, ast.FunctionDef) and node.name == 'field_owners')
        reader_namespace = {'np': np}
        exec(compile(ast.Module(body=[owner_reader], type_ignores=[]), 'field_owners', 'exec'), reader_namespace)
        for filename, output_name, expected, values in cases:
            tree = ast.parse((scripts/filename).read_text(encoding='utf8'))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
            calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'save_proxy_arrays']
            assert len(calls) == 1, 'Expected the one actual proxy writer: '+filename
            target = root/output_name
            namespace = {'save_proxy_arrays': save_proxy_arrays, 'output': root,
                         'path': target, 'asset_path': target, **values}
            eval(compile(ast.Expression(body=calls[0]), filename, 'eval'), namespace)
            with zipfile.ZipFile(target) as archive:
                assert archive.namelist() == [name+'.npy' for name in expected]
                assert all(item.compress_type == (zipfile.ZIP_STORED if item.filename == 'baseline_rgb.npy'
                                                  else zipfile.ZIP_DEFLATED) for item in archive.infolist())
            with np.load(target, allow_pickle=False) as archive:
                assert archive.files == list(expected)
                for name, original in expected.items():
                    stored = archive[name]
                    assert stored.dtype.str == original.dtype.str and stored.shape == original.shape
                    assert stored.tobytes(order='C') == original.tobytes(order='C')
            if 'owners' in expected:
                actual = reader_namespace['field_owners'](
                    {'automatic_policy': True, 'asset': {'path': str(target)}}, shape)
                assert actual.dtype == np.dtype(np.int32)
                np.testing.assert_array_equal(actual, owners)
    print('OK: actual fit/fusion/scene save calls preserve all keys, dtype, shape and bits; '
          'integer owners retained for the real sampler; fixed sparse DEFLATE1/dense RGB STORED; no models/GPU')


if __name__ == '__main__':
    main()
