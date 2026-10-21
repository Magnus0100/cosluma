"""Read-only shaping inspection; byte and numeric checks run in a worker.

``observe_run`` only observes small in-memory metadata and file stats. Its
result is a browsing invalidation hint, never a capability or an IO receipt.
``inspect_*`` always re-enter the authoritative typed readers.

``delivery_fact`` reports the requested strength next to the fraction one
saved receipt actually delivered, so the GUI never presents a request as if it
had been met. Only a geometric receipt carrying the mapping statistics can say
that; every other receipt reports unknown.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


_READERS = ('shaping_inspection.py', 'qt_workbench.py', 'appearance_assets.py', 'appearance_read_scope.py',
    'photo_shaping.py', 'geometric_appearance.py', 'face_contour_deform.py',
    'photo_shaping_geometry.py', 'portrait_regions_worker.py', 'runtime_resources.py',
    'photo_appearance.py', 'photo_development.py', 'photo_edit_session.py',
    'photo_session_evidence.py', 'photo_editing.py', 'asset_identity.py',
    'raw_support.py', 'display_transform.py', 'float_appearance.py')


def _absolute(path):
    return os.path.abspath(os.fspath(path))


def _manifest(root, run):
    if not isinstance(run, dict): raise ValueError('保存版本记录格式无效')
    name = run.get('run_id')
    if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('保存版本身份无效')
    return Path(root) / 'outputs/runs' / name / 'manifest.json'


def run_metadata(run):
    """Only fields that bind a saved shaping inspection, not the whole run."""
    if not isinstance(run, dict): return {'invalid_record_type':type(run).__name__}
    options = run.get('options') or {}; plan = run.get('task_plan') or {}
    if not isinstance(options, dict): options = {}
    if not isinstance(plan, dict): plan = {}
    def intent(value):
        return {key:copy.deepcopy(value.get(key)) for key in ('mode', 'services')} if isinstance(value, dict) else value
    design = run.get('generation_design') or {}; artifacts = run.get('artifacts') or {}
    if not isinstance(design, dict): design = {}
    observed = design.get('observed_design') or {}
    if not isinstance(observed, dict): observed = {}
    if not isinstance(artifacts, dict): artifacts = {}
    return copy.deepcopy({'run_id':run.get('run_id'), 'revision':run.get('revision'),
        'parent_run_id':run.get('parent_run_id'), 'publication':run.get('result_publication'),
        'source':run.get('source'), 'scope':plan.get('editing_scope'),
        'design_id':observed.get('design_id'),
        'current':options.get('photo_appearance'), 'shaping':options.get('shaping_state'),
        'source_appearance':run.get('initial_source_appearance'),
        'draft':artifacts.get('auto_draft'),
        'shaping_request':plan.get('shaping_request'),
        'shaping_authorization':plan.get('shaping_authorization'),
        'intents':[intent(run.get('task_intent')), intent(options.get('task_intent')), intent(plan.get('intent'))]})


def _stat(path):
    try:
        value = os.stat(path)
        return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    except OSError:
        return None


def observe_run(root, run, *, watch_paths=()):
    """Cheap GUI observation. No file content, numerical work or locks."""
    from runtime_resources import code_path
    paths = {_absolute(path) for path in watch_paths}
    try: paths.add(_absolute(_manifest(root, run)))
    except (TypeError, ValueError): pass
    record = run if isinstance(run, dict) else {}
    options = record.get('options') or {}
    if not isinstance(options, dict): options = {}
    saved = options.get('shaping_state') or {}
    if not isinstance(saved, dict): saved = {}
    for asset in (options.get('photo_appearance'), saved.get('base_appearance')):
        if isinstance(asset, dict) and isinstance(asset.get('path'), str):
            paths.add(_absolute(Path(asset['path']) / 'receipt.json'))
    artifacts = record.get('artifacts') or {}
    draft = artifacts.get('auto_draft') if isinstance(artifacts, dict) else None
    if isinstance(draft, str) and draft: paths.add(_absolute(draft))
    return {'metadata':run_metadata(run), 'files':{path:_stat(path) for path in sorted(paths)},
        'readers':{name:_stat(code_path(name)) for name in _READERS}}


def _cancelled(cancel):
    if cancel is not None and cancel(): raise ValueError('只读检查已取消')


def _read_receipt(asset):
    path = Path(asset['path']) / 'receipt.json'
    if path.stat().st_size > 1024 * 1024: raise ValueError('冻结资产记录过大')
    data = path.read_bytes()
    if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != asset['receipt_sha256']:
        raise ValueError('冻结资产记录已变化')
    return json.loads(data)


def delivery_fact(receipt, *, receipt_sha256=''):
    """Requested strength beside the fraction one saved receipt delivered.

    Only a verified geometric receipt records the mapping statistics
    (strain_limited_fraction / delivered_fraction_of_request) that say how much
    of the requested amount was actually delivered. Every other receipt returns
    None, so a caller shows unknown instead of presenting the request as met.
    Nothing here is recomputed from pixels or from the request.
    """
    if not isinstance(receipt, dict) or receipt.get('kind') != 'geometric_transform': return None
    operation = receipt.get('operation')
    if not isinstance(operation, dict): return None
    metadata = operation.get('mapping_metadata')
    stats = metadata.get('stats') if isinstance(metadata, dict) else None
    if not isinstance(stats, dict): return None
    values = (operation.get('amount'), stats.get('strain_limited_fraction'),
        stats.get('delivered_fraction_of_request'))
    if any(type(value) not in (int, float) for value in values): return None
    if any(not 0 <= value <= 1 for value in values): return None
    requested, limited, delivered = (float(value) for value in values)
    return {'requested_amount':requested, 'strain_limited_fraction':limited,
        'delivered_fraction_of_request':delivered,
        'natural_strain_bound_met':stats.get('natural_strain_bound_met') is True,
        'receipt_sha256':str(receipt_sha256 or '')}


def _owned_paths(asset):
    from appearance_assets import declared_files_from_receipt
    receipt = _read_receipt(asset)
    names = declared_files_from_receipt(receipt, asset['receipt_sha256'])
    return [str(Path(asset['path']) / name) for name in names]


def _binding(run):
    return {'source_sha256':run['source']['sha256'], 'scope':run['task_plan']['editing_scope'],
        'design_id':run['generation_design']['observed_design']['design_id']}


def _read_saved(root, run):
    path = _manifest(root, run)
    if path.stat().st_size > 16 * 1024 * 1024: raise ValueError('保存版本记录过大')
    data = path.read_bytes()
    if len(data) > 16 * 1024 * 1024: raise ValueError('保存版本记录过大')
    actual = json.loads(data)
    if run_metadata(actual) != run_metadata(run):
        raise ValueError('界面版本与实际保存记录不同')
    if (actual.get('result_publication') or {}).get('state') != 'committed':
        raise ValueError('当前版本尚未完整保存')
    return hashlib.sha256(data).hexdigest()


def _runtime_paths():
    """Discover runtime files only in the worker; the GUI later stats them."""
    import importlib
    import importlib.metadata
    paths = []
    for package in ('numpy', 'scipy'):
        distribution = importlib.metadata.distribution(package)
        members = [item for item in distribution.files or () if str(item).endswith('.dist-info/METADATA')]
        if len(members) != 1: raise ValueError('数值运行环境元数据不明确')
        paths.append(str(distribution.locate_file(members[0])))
    for name in ('numpy._core._multiarray_umath', 'scipy.ndimage._nd_image', 'scipy.spatial._qhull'):
        paths.append(str(importlib.import_module(name).__file__))
    return paths


def _inspect(root, run, *, candidate=False, cancel=None):
    """One authority check for current and U, including the saved preview."""
    from appearance_assets import force_audit
    from asset_identity import sha256_file
    from photo_editing import identity
    from photo_shaping import validate_state
    _cancelled(cancel)
    options = run['options']; current = options['photo_appearance']
    saved = options.get('shaping_state') or {}; base = saved.get('base_appearance') or current
    binding = _binding(run); paths = _owned_paths(current)+_runtime_paths()
    if base != current: paths += _owned_paths(base)
    before = observe_run(root, run, watch_paths=paths)
    manifest_sha = _read_saved(root, run)
    if candidate:
        request = saved['request']; apply = request['action'] == 'apply'
        policy = {'text_vision':apply, 'visual_review':apply}
        plan = run.get('task_plan') or {}
        intents = (run.get('task_intent'), options.get('task_intent'), plan.get('intent'))
        if (plan.get('shaping_request') != request or plan.get('shaping_authorization') != saved['authorization']
                or any(not isinstance(value, dict) or value.get('services') != policy for value in intents)):
            raise ValueError('候选的受检操作或模型服务权限不符')
    if saved:
        validate_state(saved, current=current, **binding)
        # validate_state has just verified both assets through the sole typed
        # readers. Read their small receipts without another numeric audit.
        selected = _read_receipt(current); baseline = selected if base == current else _read_receipt(base)
    else:
        selected = force_audit(current, cancel=cancel, **binding)
        baseline = selected
    # The delivered strength is read from the same verified receipt; it is
    # never recomputed here and never inferred from the request.
    delivery = delivery_fact(selected, receipt_sha256=current.get('receipt_sha256') if isinstance(current, dict) else '')
    _cancelled(cancel)
    if sha256_file(run['artifacts']['auto_draft'], force=True) != selected['files']['preview']['sha256']:
        raise ValueError('保存画面与当前母版不一致')
    after = observe_run(root, run, watch_paths=paths)
    if before != after or _read_saved(root, run) != manifest_sha:
        raise ValueError('保存版本或冻结文件在检查期间变化')
    parent = {'run_id':run['run_id'], 'manifest_sha256':manifest_sha,
        'state_sha256':identity({'source':run['source'], 'scope':binding['scope'], 'current':current, 'shaping':saved}),
        'base_id':base['asset_id'], 'base_receipt':base['receipt_sha256'],
        'base_preview_sha256':baseline['files']['preview']['sha256']}
    return {'ok':True, 'snapshot':after, 'watch_paths':paths,
        'state':{'enabled':True, 'reason':'', 'parent':parent, 'base':copy.deepcopy(base),
            'preview':str(Path(base['path']) / baseline['files']['preview']['file']),
            'restore_available':bool(saved.get('enabled')), 'delivery':delivery},
        'manifest_sha256':manifest_sha}


def _failed_fact(root, run, error):
    paths = []
    options = run.get('options') if isinstance(run, dict) else None
    if isinstance(options, dict):
        saved = options.get('shaping_state') or {}
        assets = [options.get('photo_appearance')]
        if isinstance(saved, dict): assets.append(saved.get('base_appearance'))
        for asset in assets:
            try: paths.extend(_owned_paths(asset))
            except (OSError, KeyError, TypeError, ValueError, AttributeError): pass
    try: paths.extend(_runtime_paths())
    except (OSError, KeyError, TypeError, ValueError, AttributeError): pass
    return {'ok':False, 'reason':'保存形体记录或冻结资产无效：'+str(error),
        'snapshot':observe_run(root, run, watch_paths=paths), 'watch_paths':paths}


def inspect_parent(root, run, *, cancel=None):
    """Fresh read-only audit; a returned UI fact grants no operation rights."""
    try: return _inspect(root, run, cancel=cancel)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return _failed_fact(root, run, error)


def inspect_candidate(root, run, *, parent_run, parent_identity, request, cancel=None):
    """Fresh parent + candidate audit used for preview and again on adopt."""
    parent = inspect_parent(root, parent_run, cancel=cancel)
    if not parent['ok']: return {'ok':False, 'reason':parent['reason'], 'parent_fact':parent}
    try:
        if parent['state']['parent'] != parent_identity:
            raise ValueError('当前父版本已变化')
        fact = _inspect(root, run, candidate=True, cancel=cancel)
        saved = run['options']['shaping_state']; auth = saved['authorization']; base = saved['base_appearance']
        current = parent_run['options']['photo_appearance']
        if (run.get('parent_run_id') != parent_identity['run_id'] or saved['request'] != request
                or base['asset_id'] != parent_identity['base_id']
                or base['receipt_sha256'] != parent_identity['base_receipt']
                or auth['parent_run_id'] != parent_identity['run_id']
                or auth['parent_manifest_sha256'] != parent_identity['manifest_sha256']
                or any(auth['parent_appearance'][key] != current[key] for key in ('asset_id', 'receipt_sha256'))):
            raise ValueError('候选的父版本、调整前基线或已检查请求不符')
        _cancelled(cancel)
        return {**fact, 'parent_fact':parent}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {**_failed_fact(root, run, error), 'parent_fact':parent}
