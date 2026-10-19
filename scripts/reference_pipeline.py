"""One local preparation entry point for reference matching and composition.

Preparation may use the model runtime. Reopening, editing and exporting an
owned appearance stack only replay frozen native assets in the desktop runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import sys
from material_solver_profiles import DEFAULT_PROFILE, solver_configuration
from asset_identity import sha256_file
from runtime_resources import isolated_python_environment
import workflow_timing
from cache_store import exclusive_file_lock
from reference_stage_worker import stage_session

from reference_recipe import RECIPE
PRODUCERS = ('reference_appearance_prepare.py','reference_colour_fit.py',
             'reference_colour_blend.py','reference_scene_balance.py','render_reference_colour_composite.py')
STAGES = ('fit','colour_fusion','native_baseline','scene_balance','freeze')
INPUT_CONTRACT = 'reference_pipeline_inputs.v1'
EXTRA_INPUTS = frozenset(('object_masks', 'authored_colour_edit', 'material_targets'))
CODE_DEPENDENCIES = PRODUCERS + ('colour_space.py', 'evaluate_reference_colour.py', 'display_transform.py',
    'reference_stage_worker.py', 'reference_recipe.py',
    'native_cuda_fields.py', 'native_cuda_matting.py', 'native_cuda_sampling.py',
    'raw_support.py', 'float_appearance.py', 'source_foreground.py', 'native_array_storage.py', 'local_photo_analysis.py',
    'native_source_baseline.py', 'portrait_beauty_layer.py', 'reference_pre_scene_cache.py',
    'local_semantics_worker.py', 'local_face_protection.py', 'local_face_protection_worker.py',
    'local_colour_brush.py', 'reference_support.py',
    'reference_appearance_controls.py', 'reference_local_correspondence.py',
    'reference_local_correspondence_worker.py', 'reference_adaptive_fusion.py',
    'reference_automatic_layers.py', 'source_skin_domains.py', 'reference_material_palette.py',
    'reference_material_evidence.py', 'reference_material_solver.py', 'material_solver_profiles.py',
    'material_anchor_review.py', 'qualified_material_finish.py', 'native_receiver_preservation.py')


def _stage_profiling_enabled():
    """Return whether the optional per-process stage profiler is requested.

    Process-tree sampling is diagnostic only.  Keeping it opt-in avoids a
    psutil/Windows API walk and JSONL write roughly every 250 ms during every
    production stage while retaining the explicit profile path used by smoke
    tests and performance investigations.
    """
    return os.environ.get('COSPLAY_PROFILE_STAGES', '').strip() == '1'


def _sha(path):
    return sha256_file(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    payload = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    tmp.write_text(payload, encoding='utf8')
    tmp.replace(path)
    # Callers that just persisted a durable JSON snapshot can reuse this
    # digest instead of immediately reopening the same file for a second full
    # read.  Existing callers ignore the return value, preserving the helper's
    # write-only contract.
    # ``Path.write_text`` uses the platform's universal-newline translation
    # (CRLF on Windows).  Hash the bytes that were actually committed while
    # still avoiding a second file read.
    written_payload = payload.replace('\n', os.linesep).encode('utf8')
    return hashlib.sha256(written_payload).hexdigest()


def _python(root, registry_path=None):
    from current_model_registry import load_registry
    candidate=Path(load_registry(root, registry_path=registry_path)['python'])
    if not candidate.is_file():
        raise RuntimeError('Local reference preparation runtime is unavailable; saved versions remain editable')
    return candidate


def _current_alpha(plan):
    return plan.get('mask_roles',{}).get('foreground_alpha',{})


def _resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _code_identity(root):
    # Numeric freeze changes require RECIPE changes, not packaging file hashes.
    from runtime_resources import code_path
    return {name: _sha(code_path(name)) for name in CODE_DEPENDENCIES
        if name != 'reference_appearance_prepare.py'}


def _code_file(filename):
    from runtime_resources import code_path
    return code_path(filename)




def _feature_identity(root, run):
    from reference_local_correspondence import local_correspondence_identity
    try:
        return {'available': True, 'identity': local_correspondence_identity(root,
            algorithm_profile=run.get('options', {}).get('algorithm_profile', 'dinov3_vitl16'),
            registry_path=run.get('options', {}).get('model_registry'))}
    except (OSError, RuntimeError) as error:
        # Identity records missing dependencies; it never substitutes a model
        # or converts unavailable inference into an automatic success.
        return {'available': False, 'reason': str(error)}


def _registry_identity(root, run):
    from local_photo_analysis import local_runtime_paths
    selected = run.get('options', {}).get('model_registry')
    runtime = local_runtime_paths(root, registry_path=_resolve(root, selected) if selected else None)
    registry = runtime['registry_path']
    if (selected or os.environ.get('COSPLAY_MODEL_REGISTRY')) and not registry.is_file():
        raise RuntimeError('Selected local model registry is missing: '+str(registry))
    return {'sha256': _sha(registry) if registry.is_file() else None,
            'semantics_profile': runtime['semantics_profile'],
            'matting_profile': runtime['matting_profile']}


def _asset_identity(root, asset):
    if _sha(_resolve(root, asset['path'])) != asset['sha256']:
        raise ValueError('Preparation input bytes differ from declared SHA: ' + str(asset['path']))
    return {key: copy.deepcopy(value) for key, value in asset.items()
        if key not in {'path', 'file', 'manifest', 'output'} and not key.endswith(('_path', '_file'))}


def _receipt_identity(root, value):
    """Bind JSON choices and its file closure, never live file locations."""
    if isinstance(value, list):
        return [_receipt_identity(root, item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {'previews', 'validation', 'run_id', 'manifest', 'output'}:
            continue
        if isinstance(item, str) and (key == 'path' or key.endswith(('_path', '_file'))):
            actual = _sha(_resolve(root, item))
            expected = value.get('sha256') if key == 'path' else value.get(key.rsplit('_', 1)[0] + '_sha256')
            if expected and expected != actual:
                raise ValueError('Photo-specific dependency bytes changed: ' + item)
            result[key] = {'content_sha256': actual}
        else:
            result[key] = _receipt_identity(root, item)
    return result


def _extra_identity(root, run):
    extra = run.get('reference_pipeline_inputs') or {}
    if not isinstance(extra, dict) or set(extra) - EXTRA_INPUTS:
        raise ValueError('Unsupported reference preparation inputs')
    return {key: _receipt_identity(root, _read(_resolve(root, path))) for key, path in extra.items() if path}


def _environment_identity(root, plan):
    assets = _background(plan)
    return {key: _asset_identity(root, assets[key]) for key in ('background', 'validity')}


def _input_identity(root, run):
    """The effective T/B/S inputs, excluding later manual edits/strength."""
    from display_transform import source_transform_for_run
    from native_source_baseline import retouch_identity
    from portrait_beauty_layer import identity_for_run
    from reference_colour_fit import preserve_source_hair_for_run, SOURCE_HAIR_POLICY
    source_retouch = retouch_identity(run, root=root)
    revision = run['options']['appearance_revision']
    plan = revision['design_reconstruction_plan']
    display = source_transform_for_run(run)
    canvas = plan['canvas']
    registry = _registry_identity(root, run)
    environment = _environment_identity(root, plan)
    stack = next((row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack'), None)
    if stack:
        saved_asset = stack.get('assets', {}).get('preparation_snapshot')
        if saved_asset:
            _asset_identity(root, saved_asset)
            saved = _read(_resolve(root, saved_asset['path']))
            # A repaired background is an output, not the original scene
            # input. Only an unchanged frozen output may retain that input.
            if saved.get('prepared_environment') == environment and saved.get('pipeline_inputs'):
                environment = saved['pipeline_inputs']['environment']
    return {'schema': INPUT_CONTRACT, 'recipe': RECIPE,
        'source': _asset_identity(root, {'path': run['source']['path'], 'sha256': run['source']['sha256']}),
        'reference': _asset_identity(root, {'path': run['reference']['path'], 'sha256': run['reference']['sha256']}),
        'alpha': _asset_identity(root, _current_alpha(plan)),
        'identity_protection': _asset_identity(root, plan['mask_roles']['identity_protection']),
        'canvas': {key: copy.deepcopy(canvas[key]) for key in ('source_oriented_size', 'source_proxy_size')},
        'mapping': copy.deepcopy(canvas['candidate_mapping']['candidate_original_to_final_original']),
        'source_display': display, 'portrait_beauty': identity_for_run(run),
        'source_hair_policy': SOURCE_HAIR_POLICY if preserve_source_hair_for_run(run) else None,
        'environment': environment, 'extra_inputs': _extra_identity(root, run),
        'scene_options': {'warm_colour': True, 'automatic_local_fields': True},
        'algorithm_profile': run.get('options', {}).get('algorithm_profile', 'dinov3_vitl16'),
        'model_registry_sha256': registry['sha256'],
        'model_registry_profiles': {key: value for key, value in registry.items()
            if key != 'sha256'},
        'local_feature_model': _feature_identity(root, run), 'code': _code_identity(root),
        **({'native_source_retouch': source_retouch} if source_retouch else {})}


def _stamp_inputs(root, result, identity):
    """Stamp newly produced assets only; never rewrite a saved parent."""
    plan = result['design_reconstruction_plan']
    layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    asset = layer['assets']['preparation_snapshot']
    snapshot = _read(_resolve(root, asset['path']))
    if snapshot.get('parameters', {}).get('source_hair_policy') != identity.get('source_hair_policy'):
        raise ValueError('Prepared hair policy differs from the checked preparation inputs')
    snapshot['pipeline_inputs'] = copy.deepcopy(identity)
    snapshot['prepared_environment'] = _environment_identity(root, plan)
    _write(_resolve(root, asset['path']), snapshot)
    asset['sha256'] = _sha(_resolve(root, asset['path']))
    layer['params']['provenance_identity_sha256'] = asset['sha256']
    plan.setdefault('provenance', {}).setdefault('reference_pipeline', {})['preparation_snapshot_sha256'] = asset['sha256']
    result['preparation_input_identity_sha256'] = _digest(identity)


def attach_reference_support(root, plan, output):
    """Publish source-bound preparation evidence, from the current prepared pair."""
    from reference_support import build_reference_support
    layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    snapshot_asset = layer['assets']['preparation_snapshot']
    _asset_identity(root, snapshot_asset)
    snapshot = _read(_resolve(root, snapshot_asset['path']))
    binding = {key: copy.deepcopy(layer['params'][key]) for key in
        ('source_sha256', 'reference_sha256', 'alpha_sha256', 'source_fp_sha256', 'canvas_size')}
    binding['automatic_delta_sha256'] = layer['assets']['automatic_delta']['sha256']
    report = build_reference_support(snapshot['parameters'], snapshot['colour_layer'],
        snapshot['scene_layer'], binding=binding, donor=snapshot.get('donor_support'))
    path = Path(output) / 'reference-support.json'
    _write(path, report)
    layer['params']['reference_support'] = report
    layer['assets']['reference_support'] = {'path': str(path.resolve()), 'sha256': _sha(path),
        'representation': 'path_independent_reference_support'}
    return plan




def _attach_independent_effects(root, run, result):
    from foreground_effects import prepare_effects
    effects = prepare_effects(root, run, result['design_reconstruction_plan'])
    result = {**result, 'design_reconstruction_plan': effects['plan'],
              'independent_effect_requests': effects['requests']}
    return result


def _prepared_matches(root, run, result, *, check_producer=True, input_identity=None):
    try:
        from reference_appearance_stack import validate_layer
        plan=result['design_reconstruction_plan']
        layer=next(x for x in plan['layers'] if x['kind']=='reference_appearance_stack')
        current=run['options']['appearance_revision']['design_reconstruction_plan']
        if (plan['source_sha256'] != run['source']['sha256'] or plan['candidate_sha256'] != run['reference']['sha256']
                or layer['params']['alpha_sha256'] != _current_alpha(current)['sha256']):
            return False
        snapshot_asset = layer['assets']['preparation_snapshot']
        snapshot_path = _resolve(root, snapshot_asset['path'])
        if _sha(snapshot_path) != snapshot_asset['sha256']:
            return False
        snapshot=_read(snapshot_path)
        if snapshot.get('recipe') != RECIPE:
            return False
        current_inputs = input_identity if input_identity is not None else _input_identity(root, run)
        if snapshot.get('parameters', {}).get('source_hair_policy') != current_inputs.get('source_hair_policy'):
            return False
        saved_inputs = snapshot.get('pipeline_inputs')
        if not isinstance(saved_inputs, dict):
            return False  # Saved versions replay, but unbound caches do not.
        if not check_producer:
            current_inputs = {key: value for key, value in current_inputs.items() if key != 'code'}
            saved_inputs = {key: value for key, value in saved_inputs.items() if key != 'code'}
        if saved_inputs != current_inputs:
            return False
        # The freeze format has an explicit recipe version. Metadata-only
        # packaging fixes must not refit already validated native pixels.
        # A numeric freeze change requires a new RECIPE; T/B/S code changes
        # separately invalidate preparation through their actual hashes.
        if check_producer:
            # When the caller already built the current identity, comparing
            # against that stamped code closure is equivalent and avoids
            # hashing the same dependency set a second time in one prepare
            # invocation.  Calls without an identity retain the full check.
            current_code = (current_inputs.get('code') if input_identity is not None
                             else _code_identity(root))
            if saved_inputs.get('code') != current_code:
                return False
        # Reject stale input/code snapshots before decoding gigabytes of native
        # values. A matching candidate still receives every consumer check.
        validate_layer(layer,plan,run)
        for asset in [*_background(plan).values(), *plan['source_baseline']['assets'].values()]:
            if _sha(_resolve(root, asset['path'])) != asset['sha256']:
                return False
        return True
    except (OSError,ValueError,KeyError,StopIteration,TypeError):
        return False


def _background(plan):
    return next(x for x in plan['layers'] if x['kind']=='background_design')['assets']


def _resume_fit_ready(root, run, directory, input_identity):
    """Validate the durable fit output before reusing an interrupted job."""
    try:
        directory = Path(directory)
        parameters_path = directory / 'parameters.json'
        weights_path = directory / 'source-weights.npz'
        if not parameters_path.is_file() or not weights_path.is_file():
            return False
        parameters = _read(parameters_path)
        binding = parameters['binding']
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        expected = {
            'source': run['source']['sha256'],
            'reference': run['reference']['sha256'],
            'alpha': _current_alpha(plan)['sha256'],
            'source_display': input_identity.get('source_display'),
            'portrait_beauty': input_identity.get('portrait_beauty'),
            'native_source_retouch': input_identity.get('native_source_retouch'),
        }
        for key, value in expected.items():
            if binding.get(key) != value:
                return False
        if (binding.get('algorithm_sha256') != _sha(_code_file('reference_colour_fit.py'))
                or binding.get('evaluation_sha256') != _sha(_code_file('evaluate_reference_colour.py'))
                or binding.get('source_weights_sha256') != _sha(weights_path)):
            return False
        return all((directory / name).is_file() for name in ('validation.json', 'anchors.jpg'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _resume_colour_ready(root, run, directory, input_identity, *, fit_validated=False):
    try:
        if not fit_validated and not _resume_fit_ready(root, run, directory, input_identity):
            return False
        from reference_colour_blend import validate_layer
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        native = tuple(plan['canvas']['source_oriented_size'])
        proxy = tuple(plan['canvas']['source_proxy_size'])
        parameters = _read(Path(directory) / 'parameters.json')
        validate_layer(Path(directory) / 'reference-colour-layer.json', parameters, native, proxy)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _resume_baseline_ready(root, run, directory, input_identity, *, colour_validated=False):
    try:
        if not colour_validated and not _resume_colour_ready(root, run, directory, input_identity):
            return False
        directory = Path(directory)
        parameters = _read(directory / 'parameters.json')
        receipt = _read(directory / 'composite-receipt.json')
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        native = tuple(plan['canvas']['source_oriented_size'])
        colour_path = directory / 'reference-colour-layer.json'
        master = directory / 'appearance-composite.tif'
        if (not master.is_file() or receipt.get('size') != list(native)
                or receipt.get('output_sha256') != _sha(master)
                or receipt.get('parameters_sha256') != _sha(directory / 'parameters.json')
                or receipt.get('renderer_sha256') != _sha(_code_file('render_reference_colour_composite.py'))):
            return False
        colour_receipt = receipt.get('reference_colour_layer') or {}
        if colour_receipt.get('receipt_sha256') != _sha(colour_path):
            return False
        if receipt.get('scene_balance'):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _resume_stage_ready(root, run, directory, stage, input_identity, *, validated_stages=()):
    if stage == 'fit':
        return _resume_fit_ready(root, run, directory, input_identity)
    if stage == 'colour_fusion':
        return _resume_colour_ready(root, run, directory, input_identity,
                                    fit_validated='fit' in validated_stages)
    if stage == 'native_baseline':
        return _resume_baseline_ready(root, run, directory, input_identity,
                                      colour_validated='colour_fusion' in validated_stages)
    if stage == 'scene_balance':
        return _resume_scene_ready(root, run, directory, input_identity,
                                   baseline_validated='native_baseline' in validated_stages)
    if stage == 'freeze':
        try:
            result = _read(Path(directory).parent / 'appearance' / 'prepared-plan.json')
            return _prepared_matches(root, run, result, input_identity=input_identity)
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return False


def _resume_scene_ready(root, run, directory, input_identity, *, baseline_validated=False):
    try:
        if not baseline_validated and not _resume_baseline_ready(root, run, directory, input_identity):
            return False
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        native = tuple(plan['canvas']['source_oriented_size'])
        proxy = tuple(plan['canvas']['source_proxy_size'])
        parameters = _read(Path(directory) / 'parameters.json')
        from reference_scene_balance import validate_scene_layer
        validate_scene_layer(Path(directory) / 'scene-balance-layer.json', parameters,
            Path(directory) / 'reference-colour-layer.json', plan, native, proxy)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _resume_completed_stages(root, run, work, state, input_identity, old_worker_sha):
    """Return a validated contiguous stage prefix from an interrupted job.

    A journal is evidence only when its effective identity and worker snapshot
    still match this invocation. Every skipped stage revalidates its own bound
    artifacts; already validated prerequisites are not parsed and hashed again.
    """
    try:
        if not isinstance(state, dict) or state.get('schema') != 'reference_pipeline_job.v1':
            return []
        if state.get('status') not in {'preparing', 'failed', 'cancelled'}:
            return []
        if state.get('identity') != input_identity:
            return []
        # ``state.identity == input_identity`` already covers the complete
        # current code closure. Re-hashing every producer here only repeated
        # the same check before stage-specific validation.
        if not old_worker_sha or state.get('input_sha256') != old_worker_sha:
            return []
        completed = state.get('completed_stages')
        metrics = state.get('stage_metrics')
        prefix = []
        if not isinstance(completed, list) or not isinstance(metrics, dict):
            return []
        for expected in STAGES:
            if expected not in completed:
                break
            if metrics.get(expected, {}).get('status') != 'completed':
                break
            if not _resume_stage_ready(root, run, work / 'matching', expected, input_identity,
                                       validated_stages=prefix):
                break
            prefix.append(expected)
        return prefix
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _profile_process_tree(process, state):
    """Best-effort child process metrics for the opt-in stage profiler.

    Profiling must never become a dependency of preparation.  ``psutil`` is
    optional and failures are recorded as unavailable instead of affecting the
    worker or its output.
    """
    try:
        import psutil
    except Exception:
        if os.name == 'nt':
            _profile_windows_process_tree(process.pid, state)
        else:
            state['metrics_available'] = False
        return
    state['metrics_available'] = True
    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return
    tree_rss = 0
    for child in processes:
        try:
            info = child.memory_info()
            state['peak_rss_bytes'] = max(state.get('peak_rss_bytes', 0), int(info.rss))
            tree_rss += int(info.rss)
            times = child.cpu_times()
            cpu = float(times.user) + float(times.system)
            state.setdefault('cpu_seconds_by_pid', {})[str(child.pid)] = cpu
            state['seen_pids'] = sorted({*state.get('seen_pids', []), int(child.pid)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    state['peak_tree_rss_bytes'] = max(state.get('peak_tree_rss_bytes', 0), tree_rss)


def _profile_windows_process_tree(pid, state):
    """Collect process-tree CPU/RSS using Windows APIs when psutil is absent."""
    try:
        import ctypes
        from ctypes import wintypes

        class _FileTime(ctypes.Structure):
            _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]

        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        class _ProcessEntry(ctypes.Structure):
            _fields_ = [
                ('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE,
                                             ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
                                             ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime)]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        invalid = ctypes.c_void_p(-1).value
        parent_map = {}
        snapshot_value = snapshot.value if hasattr(snapshot, 'value') else snapshot
        if snapshot_value not in (None, invalid):
            try:
                entry = _ProcessEntry(); entry.dwSize = ctypes.sizeof(_ProcessEntry)
                if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                    while True:
                        parent_map[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                        if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                            break
            finally:
                kernel32.CloseHandle(snapshot)
        process_ids = [int(pid)]
        changed = True
        while changed:
            changed = False
            for child, parent in parent_map.items():
                if parent in process_ids and child not in process_ids:
                    process_ids.append(child); changed = True

        query = 0x1000 | 0x0010  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        current_tree_rss = 0
        for process_id in process_ids:
            handle = kernel32.OpenProcess(query, False, process_id)
            if not handle:
                continue
            try:
                memory = _MemoryCounters(); memory.cb = ctypes.sizeof(_MemoryCounters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    rss = int(memory.WorkingSetSize)
                    current_tree_rss += rss
                    state['peak_rss_bytes'] = max(state.get('peak_rss_bytes', 0),
                                                   int(memory.PeakWorkingSetSize))
                creation = _FileTime(); exit_time = _FileTime(); kernel = _FileTime(); user = _FileTime()
                if kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                            ctypes.byref(kernel), ctypes.byref(user)):
                    def _seconds(value):
                        ticks = (int(value.high) << 32) | int(value.low)
                        return ticks / 10_000_000.0
                    state.setdefault('cpu_seconds_by_pid', {})[str(process_id)] = (
                        _seconds(kernel) + _seconds(user))
                state['seen_pids'] = sorted({*state.get('seen_pids', []), process_id})
            finally:
                kernel32.CloseHandle(handle)
        state['metrics_available'] = True
        state['metrics_source'] = 'windows-api-process-tree'
        state['peak_tree_rss_bytes'] = max(state.get('peak_tree_rss_bytes', 0), current_tree_rss)
    except Exception:
        # A restricted process token or an older Windows API must not break
        # preparation; wall/decode timing remains available in that case.
        state['metrics_available'] = False


def _write_stage_profile(path, value):
    """Append one JSONL record without making profiling part of the contract."""
    try:
        with Path(path).open('a', encoding='utf8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    except (OSError, TypeError, ValueError):
        pass


def _decode_profile_summary(path):
    """Summarize opt-in decode events for quick stage-level comparisons."""
    count = 0
    seconds = 0.0
    by_operation = {}
    try:
        for line in Path(path).read_text(encoding='utf8').splitlines():
            try:
                value = json.loads(line)
                elapsed = float(value.get('elapsed_seconds', 0))
            except (ValueError, TypeError, AttributeError):
                continue
            if value.get('kind') != 'decode':
                continue
            operation = str(value.get('operation', 'unknown'))
            count += 1
            seconds += elapsed
            by_operation[operation] = round(by_operation.get(operation, 0.0) + elapsed, 6)
    except OSError:
        pass
    return {'decode_event_count': count, 'decode_seconds': round(seconds, 6),
            'decode_seconds_by_operation': by_operation}


def _process(root, command, directory, cancel=None, *, stage=None, source_cache=None,
             profile_path=None, registry_path=None):
    from reference_stage_worker import execute_stage
    return execute_stage(_python(root, registry_path=registry_path), root, command, directory, cancel,
        stage=stage, source_cache=source_cache, profile_path=profile_path,
        registry_path=registry_path)


def prepare_material_finish(root, run, inputs_path, *, output=None, algorithm_profile=None,
                            cancel=None, progress=None):
    """Derive reviewed material colours from an owned complete saved endpoint.

    This records inheritance of previous T/B/S execution rather than assigning
    current producer identities to historical pixels. The caller saves a child.
    """
    from foreground_effects import prepare_effects
    root = Path(root).resolve()
    inputs_path = _resolve(root, inputs_path)
    profile = algorithm_profile or run.get('options', {}).get('material_finish_profile', DEFAULT_PROFILE)
    adapter = solver_configuration(profile)
    parent = run['options']['appearance_revision']['design_reconstruction_plan']
    layer = next(row for row in parent['layers'] if row['kind'] == 'reference_appearance_stack')
    identity = {'input_sha256': _sha(inputs_path),
        'parent_delta_sha256': layer['assets']['automatic_delta']['sha256'],
        'source_fp_sha256': layer['params']['source_fp_sha256'],
        'alpha_sha256': layer['params']['alpha_sha256'],
        'producer_sha256': _sha(_code_file('qualified_material_finish.py')),
        'algorithm_profile': profile,
        'application_dependencies_sha256': {name: _sha(_code_file(name)) for name in
            ('reference_colour_blend.py', 'render_reference_colour_composite.py')},
        'solver_dependencies_sha256': {name: _sha(_code_file(name)) for name in adapter['dependencies']}}
    if output is None:
        output = root/'outputs/cache/qualified-material-finish'/_digest(identity)[:20]
    output = Path(output).resolve()
    registry_path = (run.get('options') or {}).get('model_registry')
    if Path(sys.executable).resolve() != _python(root, registry_path=registry_path).resolve():
        # Desktop Python deliberately does not carry cv2 or the model runtime.
        # Only the CPU preparation worker imports the numeric producer.
        output.mkdir(parents=True, exist_ok=True)
        parent_snapshot = output/'material-finish-parent.json'
        _write(parent_snapshot, run)
        if progress:
            progress('material_colour', '准备原生材质颜色修正。', 0)
        _process(root, [_code_file('qualified_material_finish.py'), '--root', root,
            '--run', parent_snapshot, '--inputs', inputs_path, '--output', output,
            '--profile', profile], output, cancel, registry_path=registry_path)
        result = _read(output/'prepared-plan.json')
    else:
        from qualified_material_finish import finish_owned_appearance
        result = finish_owned_appearance(root, run, inputs_path, output,
            algorithm_profile=profile, cancel=cancel, progress=progress)
    effects = prepare_effects(root, run, result['design_reconstruction_plan'])
    result['design_reconstruction_plan'] = effects['plan']
    result.update(independent_effect_requests=effects['requests'], derived_from_saved_target=True,
                  upstream_T_B_S_refitted=False, project_pointer_updated=False)
    return result


@stage_session
def prepare(root, run, *, rebuild=False, cancel=None, progress=None):
    """Prepare a separate, frozen plan; never save a revision or call GPT.

    progress(stage, message, fraction) describes real stages, not a time ETA.
    Retry reuses a validated final cache or repeats computation; it does not
    promise arbitrary mid-stage resume. The durable input and journal survive.
    """
    root=Path(root).resolve(); run=copy.deepcopy(run)
    prepare_started = time.perf_counter()
    def report(stage,message,fraction=None):
        if cancel and cancel():
            raise InterruptedError('Reference preparation cancelled')
        if progress: progress(stage,message,fraction)
    report('reference_inputs','检查原片、参考和当前人物边缘。',0)
    plan=(run.get('options',{}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    extra_inputs = run.get('reference_pipeline_inputs') or {}
    if (set(extra_inputs) == {'material_targets'} and plan
            and any(row.get('kind') == 'reference_appearance_stack' for row in plan.get('layers', []))):
        return prepare_material_finish(root, run, extra_inputs['material_targets'],
            cancel=cancel, progress=progress)
    # The frozen input closure verifies the actual registry; missing redundant
    # scene provenance must not trigger scene preparation before that check.
    # Build the identity once for an already complete plan.  The former probe
    # discarded this value and rebuilt the complete input/code closure for the
    # cache key; retaining it preserves the fast saved-plan path without the
    # duplicate SHA work.  Incomplete plans fall through to scene preparation.
    initial_plan = plan
    initial_identity = None
    stack = next((row for row in (plan or {}).get('layers', [])
                  if row.get('kind') == 'reference_appearance_stack'), None)
    if stack and not rebuild:
        try:
            initial_identity = _input_identity(root, run)
        except (OSError, ValueError, KeyError, StopIteration, TypeError):
            initial_identity = None
        if initial_identity is not None and _prepared_matches(
                root, run, {'design_reconstruction_plan': plan},
                input_identity=initial_identity):
            report('prepared','已复用本版本的人物与场景融合资产；准备不会改动已保存版本。',1)
            return _attach_independent_effects(root,run,{'ok':True,'design_reconstruction_plan':plan,
                'cache_hit':True,'new_rendered_revision':False,'project_pointer_updated':False})
    selected_registry = _registry_identity(root, run)
    previous_registry = (plan or {}).get('provenance', {}).get('local_model_registry', {})
    registry_changed = (previous_registry.get('sha256') != selected_registry['sha256'] or
        any(previous_registry.get(key, default) != selected_registry[key] for key, default in
            (('semantics_profile', 'sapiens2_seg_0.4b'),
             ('matting_profile', 'vitmatte_small_distinctions'))))
    if (not plan or not _current_alpha(plan)
            or registry_changed
            or plan.get('source_sha256') != run['source']['sha256']
            or plan.get('candidate_sha256') != (run.get('reference') or {}).get('sha256')
            or not any(x.get('kind')=='background_design' for x in plan.get('layers',[]))):
        from design_reconstruction_pipeline import prepare_scene
        with workflow_timing.span('scene_preparation', model_request=None, http_request=False) as measured:
            scene = prepare_scene(root,run,cancel=cancel)
            if measured:
                measured.annotate(cache_hit=bool(scene.get('cache_hit')))
            plan=scene['design_reconstruction_plan']
        run.setdefault('options',{}).setdefault('appearance_revision',{})['design_reconstruction_plan']=plan
    # A complete existing plan has already passed `_input_identity` above;
    # that identity hashes and verifies the source, reference, and alpha
    # bindings.  Reusing it here avoids hashing the same large source files a
    # second time before launching workers.  Incomplete/rebuilt plans still
    # take the explicit binding checks below, and a scene-prepared plan is
    # treated as new because its matte may have changed.
    bindings_verified = initial_identity is not None and plan is initial_plan
    for role in ('source','reference'):
        path=Path(run[role]['path'])
        if not path.is_absolute():path=root/path
        run[role]['path']=str(path.resolve())
        if not bindings_verified:
            if _sha(path) != run[role]['sha256']:
                raise ValueError('Original/reference contents changed; import the current file as a new input')
    alpha=_current_alpha(plan)
    if not bindings_verified and _sha(_resolve(root, alpha['path'])) != alpha['sha256']:
        raise ValueError('Current matte bytes do not match this edit snapshot')
    identity = (initial_identity if initial_identity is not None and plan is initial_plan
                else _input_identity(root, run))
    cache_root=root/'outputs/cache/reference-pipeline'
    key=_digest(identity)[:20]
    cached_plan = cache_root/key/'appearance/prepared-plan.json'
    if not rebuild and cached_plan.is_file():
        try:
            result=_read(cached_plan)
        except (OSError, ValueError):
            result=None
        if result and _prepared_matches(root,run,result,input_identity=identity):
            report('prepared','已复用匹配当前原片、参考及人物边缘的准备结果。',1)
            return _attach_independent_effects(root,run,{**result,'cache_hit':True,'new_rendered_revision':False,'project_pointer_updated':False})
    work=cache_root/key;work.mkdir(parents=True,exist_ok=True)
    source_cache = work/'source-cache'
    worker_run=copy.deepcopy(run)
    # Diagnostic fitting reads this alpha role; the original production source
    # baseline remains immutable. New matte preparation estimates a fresh pair.
    worker_plan=worker_run['options']['appearance_revision']['design_reconstruction_plan']
    worker_plan['source_baseline']['assets']['foreground_alpha']=copy.deepcopy(alpha)
    worker_plan['mask_roles']['foreground_alpha']=copy.deepcopy(alpha)
    journal=work/'preparation-state.json'
    worker_path=work/'input-snapshot.json'
    previous_state = None
    previous_worker_sha = None
    if not rebuild and journal.is_file() and worker_path.is_file():
        try:
            previous_state = _read(journal)
            previous_worker_sha = _sha(worker_path)
        except (OSError, ValueError):
            previous_state = None
    # Re-fitting always decodes the original source and consumes its recorded
    # source transform. The prepared asset domain is never erased or relabelled.
    worker_sha = _write(worker_path, worker_run)
    resumed_stages = _resume_completed_stages(root, run, work, previous_state,
        identity, previous_worker_sha)
    if resumed_stages:
        state = copy.deepcopy(previous_state)
        # A journal can contain stale trailing entries when an external stop
        # happened after an earlier stage was invalidated.  Keep only the
        # contiguous, revalidated prefix; rerun stages must be appended once.
        state['completed_stages'] = list(resumed_stages)
        state['stage_metrics'] = {
            name: state.get('stage_metrics', {}).get(name, {})
            for name in resumed_stages
        }
        state.update(identity=identity, input_sha256=worker_sha, status='resuming',
                     source_cache=str((work/'source-cache').resolve()),
                     source_cache_policy='ephemeral_source_bound_variants_display_and_beauty',
                     stage_profile_schema='cosplay_stage_profile.v1',
                     stage_profile_path=str((work/'stage-profile.json').resolve()),
                     resumed_stages=list(resumed_stages))
    else:
        state={'schema':'reference_pipeline_job.v1','identity':identity,'input_sha256':worker_sha,
               'status':'preparing','completed_stages':[],'saved_version_modified':False,'generation_requested':False,
               'source_cache': str((work/'source-cache').resolve()),
               'source_cache_policy': 'ephemeral_source_bound_variants_display_and_beauty',
               'stage_profile_schema': 'cosplay_stage_profile.v1',
               'stage_profile_path': str((work/'stage-profile.json').resolve()),
               'stage_metrics': {}}
    _write(journal,state)
    match=work/'matching'
    extra={key: str(_resolve(root, path)) for key, path in (run.get('reference_pipeline_inputs') or {}).items() if path}
    resumed = set(resumed_stages)

    def reuse_stage(name, message):
        with workflow_timing.span('reference_' + name, cache_hit=True,
                                  model_request=False, http_request=False):
            report(name, message, STAGES.index(name)/len(STAGES))
        workflow_timing.record_worker('reference_' + name, cache_hit=True,
                                      reuse_reason='validated_current_input_prefix')
        metric = state['stage_metrics'].setdefault(name, {
            'schema': 'cosplay_stage_profile.v1', 'stage': name,
        })
        metric.update(cache_hit=True, status='completed', resumed=True,
                      reused_from=str(match.resolve()))
        if name not in state['completed_stages']:
            state['completed_stages'].append(name)
        state['elapsed_seconds'] = round(time.perf_counter() - prepare_started, 3)
        _write(work/'stage-profile.json', {
            'schema': 'cosplay_reference_stage_profile.v1',
            'recipe': RECIPE,
            'stages': copy.deepcopy(state['stage_metrics']),
            'completed_stages': list(state['completed_stages']),
            'resumed_stages': list(resumed_stages),
            'elapsed_seconds': state['elapsed_seconds'],
        })
        _write(journal,state)

    def run_stage(name, message, command):
        if name in resumed:
            reuse_stage(name, '复用同一输入身份已验证的阶段产物。')
        else:
            stage(name, message, command)

    def _stage_unlocked(name,message,command):
        report(name,message,STAGES.index(name)/len(STAGES))
        # Persist the stage boundary before launching the worker.  A user
        # cancellation or an external process stop can happen while the
        # worker is still running; without this checkpoint the durable
        # journal only says ``stage=name`` and the JSON ledger does not exist.
        # Process-tree sampling and JSONL decode events are diagnostic only and
        # stay disabled unless explicitly requested for a profiling run.
        profile_path = (work / f'{name}.profile.{os.getpid()}.{time.time_ns()}.jsonl'
                        if _stage_profiling_enabled() else None)
        started_stage = time.perf_counter()
        state.update(stage=name,status='preparing')
        state['stage_metrics'][name] = {
            'schema': 'cosplay_stage_profile.v1', 'stage': name,
            'status': 'running', 'cache_hit': False,
        }
        if profile_path is not None:
            state['stage_metrics'][name]['profile_path'] = str(profile_path.resolve())
        _write(journal,state)
        _write(work/'stage-profile.json', {
            'schema': 'cosplay_reference_stage_profile.v1',
            'recipe': RECIPE,
            'stages': copy.deepcopy(state['stage_metrics']),
            'completed_stages': list(state['completed_stages']),
            'active_stage': name,
            'elapsed_seconds': round(time.perf_counter() - prepare_started, 3),
        })
        try:
            with workflow_timing.span('reference_' + name, cache_hit=False,
                                      model_request=None if name == 'fit' else False,
                                      http_request=False):
                metrics = _process(root,command,work,cancel,stage=name,
                                   source_cache=source_cache,profile_path=profile_path,
                                   registry_path=(run.get('options') or {}).get('model_registry'))
                if metrics:
                    workflow_timing.record_worker('reference_' + name,
                        wall_seconds=metrics.get('wall_seconds'),
                        worker_timeline=metrics.get('worker_timeline'),
                        worker_pid=metrics.get('worker_pid'),
                        native_field_execution=metrics.get('native_field_execution'),
                        cpu_seconds=metrics.get('cpu_seconds'),
                        peak_tree_rss_bytes=metrics.get('peak_tree_rss_bytes'),
                        decode_seconds=metrics.get('decode_seconds'),
                        native_libraw_threading=metrics.get('native_libraw_threading'),
                        profile_path=str(profile_path.resolve()) if profile_path else None)
        except BaseException as error:
            failed = dict(state['stage_metrics'][name])
            failed.update({
                'status': 'cancelled' if isinstance(error, InterruptedError) else 'failed',
                'wall_seconds': round(time.perf_counter() - started_stage, 3),
                'error_type': type(error).__name__,
                'error': str(error),
            })
            state['stage_metrics'][name] = failed
            state['elapsed_seconds'] = round(time.perf_counter() - prepare_started, 3)
            _write(work/'stage-profile.json', {
                'schema': 'cosplay_reference_stage_profile.v1',
                'recipe': RECIPE,
                'stages': copy.deepcopy(state['stage_metrics']),
                'completed_stages': list(state['completed_stages']),
                'active_stage': name,
                'elapsed_seconds': state['elapsed_seconds'],
            })
            _write(journal,state)
            raise
        if metrics is not None:
            metrics = {key: value for key, value in metrics.items()
                       if key not in {'command', 'cpu_seconds_by_pid'}}
            metrics['cache_hit'] = False
            metrics['status'] = 'completed'
            state['stage_metrics'][name] = metrics
        else:
            # Keep a useful wall-time ledger when optional profiling is off.
            state['stage_metrics'][name].update({
                'status': 'completed',
                'wall_seconds': round(time.perf_counter() - started_stage, 3),
            })
        state['completed_stages'].append(name)
        state['elapsed_seconds'] = round(time.perf_counter() - prepare_started, 3)
        _write(work/'stage-profile.json', {
            'schema': 'cosplay_reference_stage_profile.v1',
            'recipe': RECIPE,
            'stages': copy.deepcopy(state['stage_metrics']),
            'completed_stages': list(state['completed_stages']),
            'elapsed_seconds': state['elapsed_seconds'],
        })
        _write(journal,state)

    def stage(name, message, command):
        """Run one heavy stage under a cross-process identity lock."""
        nonlocal state
        with exclusive_file_lock(work/(name+'.single-flight.lock')):
            try:
                latest = _read(journal)
                completed = tuple(latest.get('completed_stages') or ()) if isinstance(latest, dict) else ()
                if (isinstance(latest, dict) and latest.get('identity') == identity
                        and latest.get('input_sha256') == worker_sha
                        and _resume_stage_ready(root, run, match, name, identity,
                                                validated_stages=completed)):
                    state = latest
                    reuse_stage(name, '复用同一输入身份已完成的阶段产物。')
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
            _stage_unlocked(name, message, command)
    try:
        if not resumed:
            (match / 'pipeline-inputs.json').unlink(missing_ok=True)
        fit=[_code_file('evaluate_reference_colour.py'),'--root',root,'--run',worker_path,'--output',match]
        for key,flag in [('object_masks','--object-masks'),('authored_colour_edit','--authored-colour-edit')]:
            if extra.get(key):fit += [flag,extra[key]]
        run_stage('fit','按原片与参考准备人物仿色。',fit)
        run_stage('colour_fusion','准备同材质颜色融合和眼口衔接。',[
            _code_file('reference_colour_blend.py'),'--root',root,'--run',worker_path,'--match-dir',match])
        rendering=[_code_file('render_reference_colour_composite.py'),'--run',worker_path,'--match-dir',match,
            '--reference-blend',match/'reference-colour-layer.json']
        run_stage('native_baseline','建立真实原生仿色基线。',rendering)
        balance=[_code_file('reference_scene_balance.py'),'--root',root,'--run',worker_path,'--match-dir',match,'--warm-colour']
        run_stage('scene_balance','协调人物与环境的宽尺度明暗和光色。',balance)
        _write(match / 'pipeline-inputs.json', identity)
        if 'freeze' in resumed:
            reuse_stage('freeze', '复用同一输入身份已验证的冻结资产。')
            result=_read(work/'appearance/prepared-plan.json')
        else:
            stage('freeze','保存可编辑的人物基底与融合差值。',[
                _code_file('reference_appearance_prepare.py'),'--run',worker_path,'--match-dir',match,'--output',work/'appearance'])
            result=_read(work/'appearance/prepared-plan.json')
        if extra.get('material_targets'):
            derived = copy.deepcopy(worker_run)
            derived['options']['appearance_revision']['design_reconstruction_plan'] = result['design_reconstruction_plan']
            result = prepare_material_finish(root, derived, extra['material_targets'],
                output=work/'material-finish', cancel=cancel, progress=progress)
        _stamp_inputs(root, result, identity)
        attach_reference_support(root, result['design_reconstruction_plan'], work / 'appearance')
        result = _attach_independent_effects(root, run, result)
        _write(work / 'appearance/prepared-plan.json', result)
        if not _prepared_matches(root,run,result,input_identity=identity):
            raise ValueError('Frozen appearance assets did not satisfy their current source/reference/matte contract')
        state.update(status='prepared',stage='prepared',prepared_plan=str(work/'appearance/prepared-plan.json'));_write(journal,state)
        report('prepared','人物与场景融合已准备；预览后保存高清新版本。',1)
        return result
    except BaseException as error:
        state.update(status='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error));_write(journal,state)
        raise
