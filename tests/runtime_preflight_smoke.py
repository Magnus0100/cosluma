#!/usr/bin/env python3
"""Smoke isolated module availability and the per-run execution lock."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import current_model_registry
import runtime_preflight
import single_source_workflow as workflow


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay-runtime-gate-") as folder:
        root = Path(folder)
        selected = root / "sapiens" / "runtime" / "Scripts" / "python.exe"
        selected.parent.mkdir(parents=True)
        selected.write_bytes(b"fixture executable")
        config = {"python": str(selected), "registry_path": root / "local-models.current.json"}
        config["registry_path"].write_text('{}', encoding='utf8')
        runtime_preflight._PROBE_CACHE.clear()

        with patch.object(runtime_preflight, "_missing_imports", return_value=[]), \
             patch.object(runtime_preflight, "_probe_runtime", return_value=None) as probe, \
             patch.object(current_model_registry, "registry_location",
                          return_value=(Path(config["registry_path"]), "source")), \
             patch.object(current_model_registry, "load_registry", return_value=config):
            result = runtime_preflight.preflight_for_run(root, require_local_models=True)
            assert result["status"] == "ready"
            assert result["model_python"] == str(selected.resolve())
            assert result["model_registry_origin"] == "source"
            # Repeating the same layered task reuses its successful probe.
            result = runtime_preflight.preflight_for_run(
                root, require_local_models=True, mode="existing_reference",
                execution_strategy="layered_reference_transfer")
            assert result["model_module_probe"] == "cached_success"
            assert probe.call_count == 1
            config["registry_path"].write_text('{"changed":true}', encoding='utf8')
            probe.return_value = "fixture runtime unavailable"
            try:
                runtime_preflight.preflight_for_run(root, require_local_models=True)
                raise AssertionError("changed runtime registry reused a stale probe")
            except runtime_preflight.RuntimePreflightError as exc:
                assert "fixture runtime unavailable" in str(exc)
            assert probe.call_count == 2

        # The lightweight child still reports absent modules. Real imports
        # occur in the selected worker, where DLL/ABI errors retain context.
        assert runtime_preflight._probe_runtime(Path(sys.executable), ("json",)) is None
        missing = runtime_preflight._probe_runtime(
            Path(sys.executable), ("_cosplay_missing_runtime_dependency",))
        assert "_cosplay_missing_runtime_dependency" in missing
        with patch.dict(os.environ, {"PYTHONPATH": "foreign-packages",
                                     "PYTHONHOME": "foreign-python",
                                     "PYTHONUSERBASE": "foreign-user"}), \
             patch.object(runtime_preflight.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0, stderr="")) as child:
            assert runtime_preflight._probe_runtime(selected, ("torch",)) is None
            assert child.call_args.args[0][1] == "-I"
            environment = child.call_args.kwargs["env"]
            assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"} & environment.keys()
            assert environment["PYTHONNOUSERSITE"] == "1"
        with patch.object(runtime_preflight.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("fixture", 60)):
            assert "did not finish" in runtime_preflight._probe_runtime(selected, ("torch",))

        # Route-specific contracts must not pull in the full layered registry.
        silhouette_model = root / "birefnet.onnx"
        silhouette_model.write_bytes(b"fixture")
        onnx_root = root / "onnx"
        (onnx_root / "onnxruntime/capi").mkdir(parents=True)
        (onnx_root / "onnxruntime/__init__.py").write_text("__version__='fixture'\n", encoding="utf8")
        (onnx_root / "onnxruntime/capi/onnxruntime_pybind11_state.pyd").write_bytes(b"fixture")
        silhouette_config = {"birefnet_model_path": str(silhouette_model),
                             "onnx_runtime_root": str(onnx_root), "registry_path": root / "sparse.json"}
        with patch.object(runtime_preflight, "_missing_imports", return_value=[]), \
             patch.object(current_model_registry, "registry_location",
                          return_value=(Path(config["registry_path"]), "source")), \
             patch.object(current_model_registry, "load_registry", return_value=silhouette_config) as load:
            result = runtime_preflight.preflight_for_run(
                root, require_local_models=True, mode="local_portrait",
                execution_strategy="layered_reference_transfer")
            assert result["route"] == "local_portrait_source_silhouette"
            assert result["required_registry_keys"] == ["birefnet_model_path", "onnx_runtime_root"]
            assert result["model_python"] is None
            assert load.call_args.kwargs["required_keys"] == ("birefnet_model_path", "onnx_runtime_root")

        with patch.object(runtime_preflight, "_missing_imports", return_value=["scipy"]):
            try:
                runtime_preflight.preflight_for_run(root, require_local_models=False)
                raise AssertionError("missing host dependency was not rejected")
            except runtime_preflight.RuntimePreflightError as exc:
                assert "scipy" in str(exc)

        with workflow._run_id_lock(root, "same-run"):
            lock = root / workflow.RUNS_DIR / "same-run" / ".run.lock"
            assert lock.is_file()
            try:
                with workflow._run_id_lock(root, "same-run"):
                    raise AssertionError("concurrent run lock was not enforced")
            except FileExistsError as exc:
                assert "already in progress" in str(exc)
        # The marker remains as a stable inode; ownership is released by the
        # OS, so a later retry can acquire it without unlinking any run files.
        with workflow._run_id_lock(root, "same-run"):
            assert lock.exists()

        stale = root / workflow.RUNS_DIR / "stale-run"
        stale.mkdir(parents=True)
        (stale / ".run.lock").write_text(json.dumps({"pid": 999999, "stamp": "dead"}), encoding="utf8")
        with workflow._run_id_lock(root, "stale-run"):
            assert (stale / ".run.lock").is_file()
        with workflow._run_id_lock(root, "stale-run"):
            assert (stale / ".run.lock").is_file()

    print('{"ok": true, "runtime_module_availability": true, "selected_registry_python": true, "run_id_lock": true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
