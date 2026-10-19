#!/usr/bin/env python3
"""Current BiRefNet errors must surface; content-bound masks may be cached."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import source_alpha


def _native_runtime_boundary_contract(root):
    selected = root/'selected-runtime'
    foreign_root = SimpleNamespace(__file__=str(root/'foreign-runtime/onnxruntime/__init__.py'))
    native = SimpleNamespace(__file__=str(root/'foreign-runtime/onnxruntime/capi/onnxruntime_pybind11_state.pyd'))
    with patch.object(source_alpha,'_selected_runtime_root',return_value=selected), \
            patch.dict(source_alpha.sys.modules, {'onnxruntime':foreign_root,
                'onnxruntime.capi.onnxruntime_pybind11_state':native}), \
            patch.object(source_alpha.importlib,'import_module',side_effect=AssertionError('Foreign native module was reimported')):
        try:
            source_alpha._runtime(root)
        except RuntimeError as error:
            assert 'ONNX native module' in str(error) and 'isolated process' in str(error)
        else:
            raise AssertionError('A foreign native ONNX runtime was mixed with the selected runtime')
        assert source_alpha.sys.modules['onnxruntime'] is foreign_root
        assert str(selected) not in source_alpha.sys.path
    chosen = SimpleNamespace(__version__='selected-fixture')
    with patch.object(source_alpha,'_selected_runtime_root',return_value=selected), \
            patch.dict(source_alpha.sys.modules, {'onnxruntime':foreign_root}), \
            patch.object(source_alpha.importlib,'import_module',return_value=chosen) as importer:
        assert source_alpha._runtime(root) is chosen
        importer.assert_called_once_with('onnxruntime')
        assert str(selected) not in source_alpha.sys.path


def _paired_session_contract(root):
    source, candidate, model = root/'paired-source.png', root/'paired-candidate.png', root/'paired.onnx'
    source_image = Image.new('RGB', (51, 79), (60, 120, 180))
    candidate_image = Image.new('RGB', (73, 37), (170, 100, 55))
    source_image.save(source); candidate_image.save(candidate); model.write_bytes(b'paired-model')
    initialized = []; runs = []

    class Session:
        def __init__(self, path, *, sess_options, providers):
            assert providers == ['CPUExecutionProvider'] and sess_options.log_severity_level == 3
            initialized.append(path)

        def get_inputs(self):
            return [SimpleNamespace(name='pixels',type='tensor(float)')]

        def get_providers(self):
            return ['CPUExecutionProvider']

        def get_outputs(self):
            return [SimpleNamespace(name='logits')]

        def run(self, names, values):
            assert names == ['logits']
            tensor = values['pixels']; assert tensor.shape == (1,3,1024,1024)
            runs.append(tensor.copy())
            return [tensor[:, :1]]

    runtime = SimpleNamespace(__version__='offline-fixture',SessionOptions=SimpleNamespace,InferenceSession=Session,
        get_available_providers=lambda: ['CPUExecutionProvider'])
    runtime_root = lambda _root,registry_path=None: root/('runtime-b' if registry_path == 'b' else 'runtime-a')
    with patch.dict(source_alpha.os.environ, {'COSPLAY_ALPHA_MODEL':str(model)}), \
            patch.object(source_alpha,'_selected_runtime_root',side_effect=runtime_root), \
            patch.object(source_alpha,'_runtime',return_value=runtime):
        with source_alpha.SilhouetteSessionScope() as scope:
            first, source_receipt = source_alpha.source_silhouette(root,source,source_image,session_scope=scope)
            second, candidate_receipt = source_alpha.source_silhouette(root,candidate,candidate_image,session_scope=scope)
            assert len(initialized) == 1 and len(runs) == 2
        assert scope._session is None and scope._ort is None and scope._binding is None
        assert not source_receipt['cache_hit'] and not candidate_receipt['cache_hit']
        with patch.object(source_alpha,'_runtime',side_effect=AssertionError('Cached pair initialized ONNX')), \
                patch.object(source_alpha,'_selected_runtime_root',side_effect=AssertionError('Cached pair queried ONNX')):
            with source_alpha.SilhouetteSessionScope() as cached_scope:
                cached_first, cached_source = source_alpha.source_silhouette(root,source,source_image,session_scope=cached_scope)
                cached_second, cached_candidate = source_alpha.source_silhouette(root,candidate,candidate_image,session_scope=cached_scope)
                assert cached_scope._session is None
        assert cached_source['cache_hit'] and cached_candidate['cache_hit'] and len(initialized) == 1
        np.testing.assert_array_equal(np.asarray(first),np.asarray(cached_first))
        np.testing.assert_array_equal(np.asarray(second),np.asarray(cached_second))
        with source_alpha.SilhouetteSessionScope() as changed_scope:
            source_alpha._infer(source_image,model,'birefnet_lite_onnx',root,session_scope=changed_scope)
            model.write_bytes(b'changed-paired-model')
            source_alpha._infer(source_image,model,'birefnet_lite_onnx',root,session_scope=changed_scope)
            try:
                source_alpha._infer(source_image,model,'birefnet_lite_onnx',root,registry_path='b',session_scope=changed_scope)
            except RuntimeError as error:
                assert 'Cannot change ONNX runtime' in str(error)
            else:
                raise AssertionError('Pair attempted an in-process ONNX runtime switch')
            assert len(initialized) == 3, 'Runtime switch initialized another session'
        assert changed_scope._session is None
        try:
            with source_alpha.SilhouetteSessionScope() as failed_scope:
                with patch.object(Session,'run',side_effect=RuntimeError('session inference fixture')):
                    source_alpha._infer(source_image,model,'birefnet_lite_onnx',root,session_scope=failed_scope)
        except RuntimeError as error:
            assert 'session inference fixture' in str(error)
        else:
            raise AssertionError('Session inference failure was hidden')
        assert failed_scope._session is None and failed_scope._ort is None


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_source_alpha_") as temp:
        root = Path(temp); source = root / "source.png"
        image = Image.new("RGB", (96, 64), (75, 105, 145)); image.save(source)
        with patch.dict(source_alpha.os.environ, {"COSPLAY_ALPHA_MODEL": str(root/".cache/models/birefnet_lite_fp16.onnx")}):
            try:
                source_alpha.source_silhouette(root, source, image)
            except RuntimeError as error:
                assert 'BiRefNet model is unavailable' in str(error)
            else:
                raise AssertionError('Missing BiRefNet silently produced a mask')
            for backend in ('ellipse', 'polygon', 'u2net'):
                try: source_alpha.source_silhouette(root, source, image, backend=backend)
                except ValueError: pass
                else: raise AssertionError('Removed silhouette route was accepted')
            model = root / ".cache/models/birefnet_lite_fp16.onnx"
            model.parent.mkdir(parents=True); model.write_bytes(b"cache-contract-fixture")
            with patch.object(source_alpha, "_infer", return_value=(Image.new("L", (32, 32), 255), {"preprocess": {"color": "RGB"}})) as infer:
                _, first = source_alpha.source_silhouette(root, source, image)
                _, second = source_alpha.source_silhouette(root, source, image)
                assert infer.call_count == 1 and not first["cache_hit"] and second["cache_hit"]
                model.write_bytes(b"new-model-fixture")
                _, third = source_alpha.source_silhouette(root, source, image)
                assert third["cache_key"] != first["cache_key"] and infer.call_count == 2
            # Runtime failures must surface their real cause.
            model.write_bytes(b"failure-fixture")
            with patch.object(source_alpha, "_infer", side_effect=RuntimeError("runtime fixture")):
                try: source_alpha.source_silhouette(root, source, image)
                except RuntimeError as error: assert 'runtime fixture' in str(error)
                else: raise AssertionError('Inference failure silently produced a mask')
            model.unlink()
        _paired_session_contract(root)
        _native_runtime_boundary_contract(root)
        _cuda_provider_contract(root)
    print("source alpha smoke ok: current route, explicit errors, content-bound cache, paired init once, cached init zero, scoped release, foreign native runtime rejected")
    return 0


def _cuda_provider_contract(root):
    configured = []
    class Session:
        actual = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        def __init__(self, path, *, sess_options, providers):
            configured.append(providers)
        def disable_fallback(self):
            self.fallback_disabled = True
        def get_providers(self):
            return self.actual
    runtime = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session,
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model = root/'gpu-model.onnx'; model.write_bytes(b'gpu-fixture')
    with patch.object(source_alpha, '_selected_runtime_root', return_value=root/'gpu-runtime'), \
         patch.object(source_alpha, '_runtime', return_value=runtime), \
         patch.object(source_alpha.SilhouetteSessionScope, '_prepare_cuda_libraries') as prepare_dlls:
        with source_alpha.SilhouetteSessionScope() as scope:
            _, session = scope.get(root, model)
            assert session.fallback_disabled
            assert configured[-1][0][0] == 'CUDAExecutionProvider'
            assert configured[-1][0][1]['arena_extend_strategy'] == 'kSameAsRequested'
            prepare_dlls.assert_called_once()
        with patch.object(Session, 'actual', ['CPUExecutionProvider']):
            try:
                with source_alpha.SilhouetteSessionScope() as scope:
                    scope.get(root, model)
            except RuntimeError as error:
                assert 'CPU fallback refused' in str(error)
            else:
                raise AssertionError('CUDA initialization silently fell back to CPU')


if __name__ == "__main__":
    raise SystemExit(main())
