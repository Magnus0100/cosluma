"""Fast runtime checks for the source workflow.

The desktop interpreter owns orchestration, while the model registry owns the
Python runtime used by the expensive local workers.  Historically a direct
``project_cli one-click`` call could get as far as the in-process renderer
before SciPy (or the selected model runtime) failed.  This module performs a
small, side-effect-free check before generation/rendering starts.

Only module availability and executable paths are checked here. The selected
worker performs real imports once, with its normal readable error reporting.
Do not import Torch in a throwaway preflight process: on a cold Windows/HDD
installation that repeats expensive DLL reads before the real worker starts.
Model weights are left to the existing installation and execution adapters.
"""
from __future__ import annotations

from runtime_resources import external_python_run

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from runtime_resources import isolated_python_environment


class RuntimePreflightError(RuntimeError):
    """A required interpreter or import is unavailable."""


HOST_IMPORTS = ("scipy",)
MODEL_RUNTIME_IMPORTS = ("numpy", "scipy", "PIL", "torch", "transformers", "cv2", "safetensors")
LOCAL_SILHOUETTE_HOST_IMPORTS = ("numpy", "PIL")
GENERATED_APPEARANCE_HOST_IMPORTS = ('numpy','PIL','scipy','tifffile')

# Resolve top-level import specs in an isolated interpreter, without loading
# Torch/OpenCV or their native libraries. The actual worker owns ABI validation.
# The installation is process-scoped and immutable during a desktop session;
# cache only successful probes and invalidate them when the selected registry
# or runtime executable changes.  Failures are deliberately never cached so a
# repaired runtime can be retried immediately.
_PROBE_CACHE: dict[tuple, bool] = {}


def _file_identity(path: Path | str) -> tuple | None:
    try:
        target = Path(path).expanduser().resolve()
        stat = target.stat()
    except OSError:
        return None
    return (str(target), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def _probe_cache_key(python: Path, names: Iterable[str], registry: Path | None) -> tuple:
    return (
        _file_identity(python),
        tuple(names),
        _file_identity(registry) if registry is not None else None,
    )


def _missing_imports(names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append(name)
    return missing


def _probe_runtime(python: Path, names: Iterable[str]) -> str | None:
    """Resolve required top-level modules without repeating worker imports."""
    names = tuple(names)
    expression = ("import importlib.util, sys; "
                  f"missing = [name for name in {names!r} "
                  "if importlib.util.find_spec(name) is None]; "
                  "sys.stderr.write('Modules unavailable: ' + ', '.join(missing) if missing else ''); "
                  "sys.exit(bool(missing))")
    environment = isolated_python_environment()
    try:
        result = external_python_run(
            [str(python), "-I", "-c", expression],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return "Isolated module availability check did not finish within 60 seconds"
    except OSError as exc:
        return str(exc)
    if result.returncode:
        detail = (result.stderr or "").strip().splitlines()
        return detail[-1] if detail else f"exit code {result.returncode}"
    return None


def preflight_for_run(root: Path, *, registry_path=None, require_local_models: bool = False,
                      check_host: bool = True, mode: str | None = None,
                      execution_strategy: str | None = None, editing_scope=None) -> dict:
    """Validate the runtimes needed by the current source workflow.

    The route determines the smallest runtime contract.  ``local_portrait``
    uses the source silhouette adapter (BiRefNet/ONNX loaded from the selected
    registry); reference-enabled tasks use the complete worker runtime.
    A serializable receipt is returned for the run manifest.
    """
    root = Path(root).resolve()
    if mode is not None and mode not in {'local_portrait', 'existing_reference', 'create_reference'}:
        raise ValueError('Unsupported runtime preflight mode: ' + str(mode))
    if execution_strategy is not None and execution_strategy != 'layered_reference_transfer':
        raise ValueError('Unsupported runtime preflight execution strategy: ' + str(execution_strategy))
    from task_model_preflight import uses_generated_appearance,generated_registry_status
    generated=uses_generated_appearance(mode,editing_scope)
    local_silhouette = mode == 'local_portrait'
    if local_silhouette:
        required_keys = ('birefnet_model_path', 'onnx_runtime_root')
        required_profiles = ()
        route = 'local_portrait_source_silhouette'
        model_imports = ()
    else:
        required_keys = None
        required_profiles = None
        route = 'layered_reference_transfer'
        model_imports = MODEL_RUNTIME_IMPORTS
    host_imports = GENERATED_APPEARANCE_HOST_IMPORTS if generated else tuple(HOST_IMPORTS) + (LOCAL_SILHOUETTE_HOST_IMPORTS if local_silhouette else ())
    receipt = {
        "schema_version": "cosplay_runtime_preflight.v1",
        "host_python": str(Path(sys.executable).resolve()),
        "model_registry": None,
        "model_registry_origin": None,
        "model_python": None,
        "route": route,
        "required_registry_keys": list(required_keys or ()),
        "required_host_imports": list(host_imports if check_host else ()),
        "required_model_imports": list(model_imports if require_local_models else ()),
    }
    if check_host:
        missing = _missing_imports(host_imports)
        if missing:
            raise RuntimePreflightError(
                "Desktop renderer dependencies are missing before rendering: "
                + ", ".join(missing) + ". Run the workflow with the project desktop Python."
            )

    if generated:
        selection=generated_registry_status(root,registry_path)
        if selection.get('error'):
            raise RuntimePreflightError('Explicit unused native registry metadata is invalid: '+selection['error'])
        receipt.update(route='scoped_generated_appearance',status='ready',native_models_required=False,
            model_registry=selection['registry_path'],model_registry_origin=selection['origin'],
            native_registry_configured=selection['configured'],native_registry_metadata_valid=selection['configuration_valid'],
            required_registry_keys=[],required_model_imports=[],model_python=None,
            model_inferences=0,inference_verified=False,provider_validation='separate_checked_provider_health')
        return receipt

    try:
        from current_model_registry import load_registry, registry_location

        selected, origin = registry_location(root, registry_path=registry_path)
        receipt["model_registry"] = str(selected)
        receipt["model_registry_origin"] = origin
    except (OSError, ValueError, RuntimeError) as exc:
        if require_local_models:
            raise RuntimePreflightError("Local model runtime registry is unavailable: " + str(exc)) from exc
        # Source-only rendering does not consume model weights, but retaining
        # the selected path in the receipt makes the automatic registry choice
        # visible for a later local continuation.
        receipt["registry_warning"] = str(exc)
        receipt["status"] = "ready"
        return receipt

    if not require_local_models:
        receipt["status"] = "ready"
        return receipt

    try:
        config = load_registry(root, registry_path=selected,
                               required_keys=required_keys,
                               required_profiles=required_profiles)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimePreflightError("Local model runtime registry is unavailable: " + str(exc)) from exc

    if local_silhouette:
        # Preserve the configured absolute drive.  Resolving a Windows
        # junction here would hide a mixed C:/D: installation and report a
        # stale cache path instead of the model path the registry selected.
        onnx_value = Path(str(config.get('onnx_runtime_root', ''))).expanduser()
        model_value = Path(str(config.get('birefnet_model_path', ''))).expanduser()
        onnx_root = onnx_value if onnx_value.is_absolute() else (root / onnx_value).resolve()
        model = model_value if model_value.is_absolute() else (root / model_value).resolve()
        receipt['onnx_runtime_root'] = str(onnx_root)
        receipt['silhouette_model'] = str(model)
        if not model.is_file():
            raise RuntimePreflightError('Selected BiRefNet model is unavailable: ' + str(model))
        module = onnx_root / 'onnxruntime' / '__init__.py'
        if not module.is_file():
            raise RuntimePreflightError('Selected ONNX Runtime package is unavailable: ' + str(onnx_root))
        native = [*onnx_root.joinpath('onnxruntime', 'capi').glob('onnxruntime_pybind11_state*.pyd'),
                  *onnx_root.joinpath('onnxruntime', 'capi').glob('onnxruntime_pybind11_state*.so')]
        if len(native) != 1 or native[0].stat().st_size <= 0:
            raise RuntimePreflightError('Selected ONNX Runtime native module is missing or ambiguous: ' + str(onnx_root))
    else:
        python = Path(str(config.get("python", ""))).expanduser().resolve()
        receipt["model_python"] = str(python)
        if not python.is_file():
            raise RuntimePreflightError("Selected model runtime Python is unavailable: " + str(python))
        probe_key = _probe_cache_key(python, model_imports, selected)
        cached_probe = any(
            success and key[0] == probe_key[0] and key[2] == probe_key[2]
            and set(model_imports).issubset(key[1])
            for key, success in _PROBE_CACHE.items())
        if cached_probe is True:
            error = None
            receipt["model_module_probe"] = "cached_success"
        else:
            error = _probe_runtime(python, model_imports)
            if error is None:
                _PROBE_CACHE[probe_key] = True
                # Keep this bounded for long-lived desktop sessions that may
                # switch between a handful of model registries.
                if len(_PROBE_CACHE) > 16:
                    _PROBE_CACHE.pop(next(iter(_PROBE_CACHE)))
                receipt["model_module_probe"] = "executed_success"
        if error:
            raise RuntimePreflightError(
                "Selected model runtime module check failed ("
                + ", ".join(model_imports) + "): " + error
            )
    receipt["status"] = "ready"
    return receipt

