"""Immutable native-canvas appearance masters and their viewing previews.

This is asset IO, not a runner. A preview never reconstructs a source master.
Generated coverage records pixel provenance, not segmentation or approval.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from photo_editing import identity, validate_scope
import workflow_timing

SCHEMA = 'cosluma_photo_appearance.v1'
DOMAIN = 'display_linear_sRGB'
KINDS = {'source_baseline', 'mixed_regional', 'generated_full', 'photographic_adjustment'}
_VALIDATED = {}
_PRODUCER_FILES = ('photo_appearance.py', 'asset_identity.py', 'raw_support.py',
    'display_transform.py', 'float_appearance.py', 'photo_editing.py')


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Appearance requires a SHA-256 content identity')
    return value


def _producer():
    return {name:sha256_file(Path(__file__).with_name(name)) for name in _PRODUCER_FILES}


def _cancel(cancel):
    if cancel and cancel():
        raise InterruptedError('Appearance preparation cancelled')


def _array(values, shape=None):
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim not in (2, 3):
        raise ValueError('Appearance arrays must retain actual float32 precision')
    if shape is not None and values.shape != tuple(shape):
        raise ValueError('Appearance array differs from its oriented native canvas')
    for top in range(0, len(values), 128):
        block = values[top:top+128]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError('Appearance must be finite normalized display-linear SDR')
    return values


def _size(master):
    if master.ndim != 3 or master.shape[2] != 3 or min(master.shape[:2]) < 1:
        raise ValueError('Appearance master must be a complete RGB canvas')
    return [master.shape[1], master.shape[0]]


def _file(directory, record):
    if (not isinstance(record, dict) or set(record) != {'file', 'sha256'}
            or not isinstance(record['file'], str) or not record['file']
            or Path(record['file']).name != record['file']
            or '/' in record['file'] or '\\' in record['file'] or ':' in record['file']
            or record['file'] in {'.', '..'}):
        raise ValueError('Appearance files require owned relative basenames')
    path = directory/record['file']
    if not path.resolve().is_relative_to(directory.resolve()) or sha256_file(path) != _sha(record['sha256']):
        raise ValueError('Appearance file content changed')
    return path


def _record(path):
    return {'file':path.name, 'sha256':sha256_file(path)}


def _photo_semantics(parent):
    kind=parent['kind']
    only=(kind=='source_baseline' or kind=='photographic_adjustment' and
        parent['photographic_operation']['source_only_photographic_lineage'])
    return {'kind':kind,'source_only_photographic_lineage':bool(only)}


def _validate_photo_operation(directory,receipt):
    from photo_development import operator_for,producer_identity
    meta=receipt['photographic_operation']
    if (not isinstance(meta,dict) or set(meta)!={'operation_receipt','source_only_photographic_lineage','parent_semantics'}
            or type(meta['source_only_photographic_lineage']) is not bool):
        raise ValueError('Photographic appearance needs an actual CPU operation binding')
    semantics=meta['parent_semantics']
    if (not isinstance(semantics,dict) or set(semantics)!={'kind','source_only_photographic_lineage'}
            or semantics['kind'] not in KINDS or type(semantics['source_only_photographic_lineage']) is not bool
            or meta['source_only_photographic_lineage']!=semantics['source_only_photographic_lineage']
            or semantics['kind']=='source_baseline' and not semantics['source_only_photographic_lineage']
            or semantics['kind'] in {'generated_full','mixed_regional'} and semantics['source_only_photographic_lineage']):
        raise ValueError('Photographic-only lineage differs from the actual parent semantics')
    operation=json.loads(_file(directory,meta['operation_receipt']).read_text(encoding='utf8'))
    expected_meta={key:value for key,value in meta.items() if key!='operation_receipt'}
    if (operation.get('schema')!='cosluma_photographic_development.v1'
            or operation.get('source_sha256')!=receipt['source_sha256']
            or operation.get('scope_id')!=receipt['scope_id'] or operation.get('design_id')!=receipt['design_id']
            or operation.get('input_parent')!=receipt['parent']
            or operation.get('coverage')!=receipt['edit_coverage']
            or operation.get('coverage_semantics')!='photographic_edit_permission_not_generated_pixel_ownership'
            or operation.get('photographic_operation')!=expected_meta
            or operation.get('producer')!=producer_identity()
            or operation.get('image_generation_submissions')!=0 or operation.get('human_approved') is not False
            or operation.get('physical_relighting_verified') is not False
            or operation.get('output')!={'master_sha256':receipt['files']['master']['sha256'],
                'preview':receipt['files']['preview']}):
        raise ValueError('Photographic CPU receipt does not bind this exact native float/preview')
    operator=operation.get('operator') or {}
    if operator!=operator_for(operator.get('parameters')):
        raise ValueError('Photographic operator or checked numeric parameters changed')
    stats=operation.get('stats') or {}
    if (not isinstance(stats,dict) or stats.get('native_pixels')!=receipt['canvas_size'][0]*receipt['canvas_size'][1]
            or type(stats.get('changed_pixels')) is not int or not 0<=stats['changed_pixels']<=stats['native_pixels']):
        raise ValueError('Photographic receipt needs actual native pixel statistics')
    return operation


def _coverage(directory, record, shape):
    if not isinstance(record, dict):
        raise ValueError('Explicit appearance coverage is required')
    if set(record) == {'uniform'} and type(record['uniform']) is int and record['uniform'] in (0, 1):
        return record['uniform']
    if set(record) == {'array'}:
        return _array(np.load(_file(directory, record['array']), mmap_mode='r', allow_pickle=False), shape)
    raise ValueError('Coverage must be uniform zero/one or an owned float32 array')


def _save_coverage(directory, name, value):
    if type(value) is int and value in (0, 1):
        return {'uniform':value}
    if np.all(value == 0):
        return {'uniform':0}
    if np.all(value == 1):
        return {'uniform':1}
    path = directory/(name+'.npy')
    np.save(path, value, allow_pickle=False)
    return {'array':_record(path)}


def _box(value, size):
    if (not isinstance(value, list) or len(value) != 4 or any(type(x) is not int for x in value)
            or not 0 <= value[0] < value[2] <= size[0]
            or not 0 <= value[1] < value[3] <= size[1]):
        raise ValueError('Generation box must be in the oriented native canvas')
    return value


def _generation(directory, value, source_sha256, size, kind, coverage=None, *, scope,copy_files=False):
    """Bind original registered bytes without rewriting historical locators."""
    keys = {'image', 'provider_receipt', 'context_box', 'core_box'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Generated appearance requires actual registered image and receipt evidence')
    records = {}
    for role in ('image', 'provider_receipt'):
        item = value[role]
        if copy_files:
            if (not isinstance(item, dict) or set(item) != {'path', 'sha256'}
                    or sha256_file(item['path']) != _sha(item['sha256'])):
                raise ValueError('Generation input evidence changed')
            path = directory/('generated.png' if role == 'image' else 'generation-receipt.json')
            shutil.copyfile(item['path'], path)
            if sha256_file(path) != item['sha256']:
                raise ValueError('Generation evidence changed during copy')
            records[role] = _record(path)
        else:
            records[role] = item
    image = _file(directory, records['image'])
    receipt = json.loads(_file(directory, records['provider_receipt']).read_text(encoding='utf-8'))
    with Image.open(image) as raster:
        generated_size = list(raster.size)
        if raster.format != 'PNG' or raster.mode not in {'RGB', 'RGBA'}:
            raise ValueError('Generation evidence must be the actual registered PNG')
        raster.verify()
    if (receipt.get('status') != 'reference_candidate'
            or (receipt.get('source') or {}).get('sha256') != source_sha256
            or (receipt.get('asset') or {}).get('sha256') != records['image']['sha256']
            or not isinstance(receipt.get('provider'), str) or not receipt['provider']
            or not isinstance(receipt.get('model'), str) or not receipt['model']):
        raise ValueError('Registered provider provenance does not bind the actual image/source')
    declared = (receipt.get('asset') or {}).get('oriented_size')
    if declared is not None and declared != generated_size:
        raise ValueError('Generated image resolution differs from its provider receipt')
    core, context = value['core_box'], value['context_box']
    if kind == 'generated_full':
        if core is not None or context is not None:
            raise ValueError('A complete generated image cannot claim regional source ownership')
    else:
        core, context = _box(core, size), _box(context, size)
        if (core == [0, 0, *size] and scope['work_type']=='retouch'):
            raise ValueError('Retouching cannot own a whole-canvas generated edit')
        if not (context[0] <= core[0] < core[2] <= context[2]
                 and context[1] <= core[1] < core[3] <= context[3]):
            raise ValueError('Regional generation requires a permitted core and context')
        if coverage is not None:
            for top in range(0, size[1], 128):
                active = np.asarray(coverage[top:top+128]) > 0
                if (np.any(active[:, :core[0]]) or np.any(active[:, core[2]:])
                        or top < core[1] and np.any(active[:min(len(active), core[1]-top)])
                        or top+len(active) > core[3] and np.any(active[max(0, core[3]-top):])):
                    raise ValueError('Edit coverage extends outside its permitted generation core')
    return {**records, 'context_box':context, 'core_box':core}, generated_size


def _preview_matches(master, preview):
    from raw_support import deterministic_srgb_profile
    from float_appearance import linear_to_srgb
    with Image.open(preview) as image:
        if (image.mode != 'RGB' or list(image.size) != _size(master)
                or image.info.get('icc_profile') != deterministic_srgb_profile()):
            raise ValueError('Appearance preview must have matching canvas and deterministic sRGB ICC')
        for top in range(0, len(master), 128):
            expected = np.uint8(linear_to_srgb(master[top:top+128])*255.0+.5)
            actual = np.asarray(image.crop((0, top, image.width, min(top+128, image.height))))
            if not np.array_equal(actual, expected):
                raise ValueError('Appearance preview does not encode the actual float master')


def validate_asset(asset, *, source_sha256, scope, design_id):
    """Validate current-version frozen bytes with stat-bound identity memo."""
    scope = validate_scope(scope)
    _sha(source_sha256); _sha(design_id)
    if not isinstance(asset, dict) or set(asset) != {'path', 'asset_id', 'receipt_sha256'}:
        raise ValueError('Expected a typed appearance asset locator')
    directory = Path(asset['path']).resolve()
    receipt_path = directory/'receipt.json'
    if sha256_file(receipt_path) != _sha(asset['receipt_sha256']):
        raise ValueError('Appearance receipt changed')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    expected_keys = {'schema', 'kind', 'source_sha256', 'scope_id', 'design_id', 'working_domain',
        'canvas_size', 'producer', 'parent', 'source_preparation', 'source_origin', 'generation',
        'generated_resolution', 'source_coverage_semantics', 'native_generated_detail',
        'human_approved', 'geometry_verified', 'physical_relighting_verified', 'files',
        'edit_coverage', 'generated_coverage', 'photographic_operation', 'asset_id'}
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError('Appearance requires the current explicit typed receipt')
    producer=receipt.get('producer')
    if not isinstance(producer,dict) or set(producer)!=set(_PRODUCER_FILES):
        raise ValueError('Appearance lacks the recorded producer identities')
    for value in producer.values():_sha(value)
    if producer!=_producer():
        raise ValueError('Current preparation producer differs from the frozen appearance')
    content = {key:value for key,value in receipt.items() if key != 'asset_id'}
    if (receipt.get('schema') != SCHEMA or receipt.get('kind') not in KINDS
            or receipt.get('asset_id') != identity(content) or receipt['asset_id'] != asset['asset_id']
            or receipt.get('source_sha256') != source_sha256 or receipt.get('scope_id') != scope['scope_id']
            or receipt.get('design_id') != design_id or receipt.get('working_domain') != DOMAIN
            or receipt.get('human_approved') is not False
            or receipt.get('geometry_verified') is not False or receipt.get('physical_relighting_verified') is not False):
        raise ValueError('Appearance content, producer or checked source/design/scope binding changed')
    files = receipt.get('files') or {}
    if set(files) != {'master', 'preview'}:
        raise ValueError('Appearance needs one authoritative master and matching preview')
    paths = [_file(directory, files[role]) for role in ('master', 'preview')]
    for key in ('edit_coverage', 'generated_coverage'):
        record = receipt.get(key)
        if isinstance(record, dict) and 'array' in record:
            paths.append(_file(directory, record['array']))
    generation = receipt.get('generation')
    if generation:
        paths += [_file(directory, generation[key]) for key in ('image', 'provider_receipt')]
    if receipt.get('photographic_operation') is not None:
        paths.append(_file(directory,receipt['photographic_operation']['operation_receipt']))
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths)
    key = (str(directory), asset['receipt_sha256'], signature)
    if key in _VALIDATED:
        return copy.deepcopy(receipt)
    master = _array(np.load(paths[0], mmap_mode='r', allow_pickle=False))
    size = _size(master)
    if receipt.get('canvas_size') != size:
        raise ValueError('Appearance native canvas changed')
    shape = master.shape[:2]
    edit = _coverage(directory, receipt.get('edit_coverage'), shape)
    generated = _coverage(directory, receipt.get('generated_coverage'), shape)
    kind = receipt['kind']
    if kind!='photographic_adjustment' and receipt.get('photographic_operation') is not None:
        raise ValueError('A source/generation asset cannot impersonate photographic processing')
    from display_transform import validate_source_transform
    origin = receipt.get('source_origin') or {}
    if (set(origin) != {'source_precision', 'decode_version', 'transform',
            'source_display_applied_once', 'baseline_master_sha256'}
            or origin['source_display_applied_once'] is not True or not origin['decode_version']):
        raise ValueError('Appearance lost the original source preparation lineage')
    validate_source_transform(origin['transform'], origin['source_precision'])
    _sha(origin['baseline_master_sha256'])
    if kind == 'source_baseline':
        preparation = receipt.get('source_preparation') or {}
        validate_source_transform(preparation.get('transform'), preparation.get('source_precision'))
        if (preparation.get('source_display_applied_once') is not True
                or not preparation.get('decode_version') or type(edit) is not int or edit != 0
                or type(generated) is not int or generated != 0
                or generation is not None or receipt.get('parent') is not None
                or origin != {**preparation, 'baseline_master_sha256':files['master']['sha256']}):
            raise ValueError('Source baseline cannot claim generated or caller-supplied appearance')
    else:
        parent = receipt.get('parent') or {}
        if (set(parent) != {'asset_id', 'master_sha256', 'preview_sha256'}
                or receipt.get('source_preparation') is not None):
            raise ValueError('Edited appearance requires explicit frozen parent content')
        for value in parent.values():_sha(value)
        if kind == 'generated_full' and (type(edit) is not int or edit != 1
                or type(generated) is not int or generated != 1 or scope['work_type'] == 'retouch'):
            raise ValueError('Generated full coverage cannot claim native source detail or retouch scope')
        if kind == 'mixed_regional' and (not isinstance(edit, np.ndarray) or not np.any(edit > 0)
                or not np.any(edit < 1)):
            raise ValueError('Mixed appearance requires actual partial edit coverage')
        if kind=='photographic_adjustment':
            _validate_photo_operation(directory,receipt)
            if receipt['photographic_operation']['source_only_photographic_lineage'] and (
                    type(generated) is not int or generated!=0 or generation is not None):
                raise ValueError('Source-only photographic processing cannot introduce generated pixels')
        if generation is not None:
            owner_kind=kind if kind!='photographic_adjustment' else (
                'generated_full' if generation['core_box'] is None else 'mixed_regional')
            checked, actual_size = _generation(directory, generation, source_sha256, size, owner_kind,
                edit if kind == 'mixed_regional' else None,scope=scope)
            if checked != generation or receipt.get('generated_resolution') != actual_size:
                raise ValueError('Generation resolution or ownership provenance changed')
        elif kind!='photographic_adjustment' or receipt.get('generated_resolution') is not None:
            raise ValueError('Generated appearance lacks actual original generation evidence')
    if (receipt.get('native_generated_detail') is not False
            or receipt.get('source_coverage_semantics') != 'one_minus_cumulative_generated_coverage_not_unprocessed_light'):
        raise ValueError('Appearance coverage must not falsely certify native camera detail')
    _preview_matches(master, paths[1])
    if len(_VALIDATED) >= 64:_VALIDATED.pop(next(iter(_VALIDATED)))
    _VALIDATED[key] = True
    return copy.deepcopy(receipt)


def load_asset(asset, *, source_sha256, scope, design_id):
    """Load the frozen float master, never decode or reconstruct from preview."""
    receipt = validate_asset(asset, source_sha256=source_sha256, scope=scope, design_id=design_id)
    directory = Path(asset['path']).resolve()
    master = np.load(directory/receipt['files']['master']['file'], mmap_mode='r', allow_pickle=False)
    generated = _coverage(directory, receipt['generated_coverage'], master.shape[:2])
    source = 1-generated
    return {'receipt':receipt, 'master':master,
        'preview_path':directory/receipt['files']['preview']['file'],
        'generated_coverage':generated, 'source_coverage':source,
        'edit_coverage':_coverage(directory, receipt['edit_coverage'], master.shape[:2])}


@workflow_timing.timed('photo_appearance_materialize',model_request=False,http_request=False)
def materialize_asset(asset, directory, *, source_sha256, scope, design_id):
    """Give another revision owned bytes without rewriting any content identity."""
    receipt=validate_asset(asset,source_sha256=source_sha256,scope=scope,design_id=design_id)
    source=Path(asset['path']).resolve();destination=Path(directory).resolve()
    if destination.exists():raise FileExistsError('Appearance destination already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    claim=destination.with_name('.'+destination.name+'.claim');claim.mkdir()
    staging=destination.with_name('.'+destination.name+'.partial-'+uuid4().hex)
    try:
        staging.mkdir()
        records=list(receipt['files'].values())
        records += [receipt[key]['array'] for key in ('edit_coverage','generated_coverage')
            if 'array' in receipt[key]]
        if receipt['generation']:
            records += [receipt['generation'][key] for key in ('image','provider_receipt')]
        if receipt.get('photographic_operation'):
            records.append(receipt['photographic_operation']['operation_receipt'])
        shutil.copyfile(source/'receipt.json',staging/'receipt.json')
        for row in records:shutil.copyfile(_file(source,row),staging/row['file'])
        located={**asset,'path':str(staging)}
        validate_asset(located,source_sha256=source_sha256,scope=scope,design_id=design_id)
        validate_asset(asset,source_sha256=source_sha256,scope=scope,design_id=design_id)
        if destination.exists():raise FileExistsError('Appearance destination already exists')
        os.rename(staging,destination)
        return {**asset,'path':str(destination)}
    finally:
        shutil.rmtree(staging,ignore_errors=True);claim.rmdir()


def _publish(directory, master, receipt, *, scope, edit, generated, generation=None,
             cancel=None, precommit=None, photographic_operation=None):
    from raw_support import linear_preview
    directory = Path(directory).resolve()
    if directory.exists():raise FileExistsError('Appearance destination already exists')
    directory.parent.mkdir(parents=True, exist_ok=True)
    # The claim prevents concurrent creators from publishing this directory.
    claim = directory.with_name('.'+directory.name+'.claim')
    claim.mkdir()
    staging = directory.with_name('.'+directory.name+'.partial-'+uuid4().hex)
    try:
        if directory.exists():raise FileExistsError('Appearance destination already exists')
        staging.mkdir();_cancel(cancel)
        np.save(staging/'master.npy', master, allow_pickle=False)
        image = linear_preview(master)
        image.save(staging/'preview.png', format='PNG', icc_profile=image.info['icc_profile'])
        receipt.update(files={'master':_record(staging/'master.npy'), 'preview':_record(staging/'preview.png')},
            edit_coverage=_save_coverage(staging, 'edit-coverage', edit),
            generated_coverage=_save_coverage(staging, 'generated-coverage', generated))
        if receipt['kind'] == 'source_baseline':
            receipt['source_origin'] = {**copy.deepcopy(receipt['source_preparation']),
                'baseline_master_sha256':receipt['files']['master']['sha256']}
        if generation is not None:
            owner_kind=receipt['kind'] if receipt['kind']!='photographic_adjustment' else (
                'generated_full' if generation['core_box'] is None else 'mixed_regional')
            evidence, size = _generation(staging, generation, receipt['source_sha256'],
                receipt['canvas_size'], owner_kind,
                edit if receipt['kind']=='mixed_regional' and isinstance(edit,np.ndarray) else None,
                scope=scope,copy_files=True)
            receipt.update(generation=evidence, generated_resolution=size)
        if photographic_operation is not None:
            semantics=receipt['photographic_operation']
            operation={'schema':'cosluma_photographic_development.v1',
                'source_sha256':receipt['source_sha256'],'scope_id':receipt['scope_id'],'design_id':receipt['design_id'],
                'input_parent':receipt['parent'],'photographic_operation':copy.deepcopy(semantics),
                **copy.deepcopy(photographic_operation),
                'coverage':receipt['edit_coverage'],
                'coverage_semantics':'photographic_edit_permission_not_generated_pixel_ownership',
                'output':{'master_sha256':receipt['files']['master']['sha256'],'preview':receipt['files']['preview']},
                'image_generation_submissions':0,'human_approved':False,'physical_relighting_verified':False}
            operation_path=staging/'photographic-operation.json'
            with operation_path.open('x',encoding='utf8') as stream:
                json.dump(operation,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
                stream.flush();os.fsync(stream.fileno())
            receipt['photographic_operation']['operation_receipt']=_record(operation_path)
        receipt['asset_id'] = identity(receipt)
        path = staging/'receipt.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        descriptor = {'path':str(staging), 'asset_id':receipt['asset_id'], 'receipt_sha256':sha256_file(path)}
        validate_asset(descriptor, source_sha256=receipt['source_sha256'], scope=scope,
            design_id=receipt['design_id'])
        if precommit is not None:precommit(staging)
        _cancel(cancel)
        os.rename(staging, directory)
        descriptor['path'] = str(directory)
        return descriptor
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        claim.rmdir()


def _receipt(kind, source_sha256, scope, design_id, master):
    scope = validate_scope(scope);_sha(source_sha256);_sha(design_id)
    return {'schema':SCHEMA, 'kind':kind, 'source_sha256':source_sha256,
        'scope_id':scope['scope_id'], 'design_id':design_id, 'working_domain':DOMAIN,
        'canvas_size':_size(master), 'producer':_producer(), 'parent':None,
        'source_preparation':None, 'source_origin':None, 'generation':None, 'generated_resolution':None,
        'photographic_operation':None,
        'source_coverage_semantics':'one_minus_cumulative_generated_coverage_not_unprocessed_light',
        'native_generated_detail':False, 'human_approved':False,
        'geometry_verified':False, 'physical_relighting_verified':False}


@workflow_timing.timed('source_native_baseline',model_request=False,http_request=False)
def create_source_baseline(directory, *, source_path, source_sha256, scope, design_id, cancel=None):
    """Decode original bytes and apply native source display preparation once."""
    from raw_support import open_linear
    from display_transform import source_transform_for_precision, prepare_source_display_inplace
    scope = validate_scope(scope);_sha(source_sha256);_sha(design_id)
    if Path(directory).exists():raise FileExistsError('Appearance destination already exists')
    source_path = Path(source_path).resolve();_cancel(cancel)
    if sha256_file(source_path) != source_sha256:raise ValueError('Original source changed before decode')
    master, precision = open_linear(source_path)
    transform = source_transform_for_precision(precision['source_precision'])
    prepare_source_display_inplace(master, transform)
    _array(master);receipt = _receipt('source_baseline', source_sha256, scope, design_id, master)
    receipt['source_preparation'] = {'source_precision':precision['source_precision'],
        'decode_version':precision['decode_version'], 'transform':transform, 'source_display_applied_once':True}
    if sha256_file(source_path) != source_sha256:raise ValueError('Original source changed during decode')
    def precommit(staging):
        if sha256_file(source_path) != source_sha256:raise ValueError('Original source changed before publication')
    return _publish(directory, master, receipt, scope=scope, edit=0, generated=0,
        cancel=cancel, precommit=precommit)


@workflow_timing.timed('photo_appearance_freeze',model_request=False,http_request=False)
def freeze_candidate(directory, *, kind, master, source_sha256, scope, design_id,
                     parent, coverage=None, generation=None, photographic_operation=None, cancel=None):
    """Freeze the actual current float result with explicit pixel ownership."""
    if kind not in {'mixed_regional', 'generated_full', 'photographic_adjustment'}:
        raise ValueError('Only original decode may create a source baseline')
    scope = validate_scope(scope);_array(master);size = _size(master)
    prior = load_asset(parent, source_sha256=source_sha256, scope=scope, design_id=design_id)
    if prior['receipt']['canvas_size'] != size:raise ValueError('Edited appearance changed native geometry')
    receipt = _receipt(kind, source_sha256, scope, design_id, master)
    receipt['source_origin'] = copy.deepcopy(prior['receipt']['source_origin'])
    receipt['parent'] = {'asset_id':prior['receipt']['asset_id'],
        'master_sha256':prior['receipt']['files']['master']['sha256'],
        'preview_sha256':prior['receipt']['files']['preview']['sha256']}
    if kind in {'mixed_regional','photographic_adjustment'}:
        edit=(coverage if type(coverage) is int and coverage in (0,1) and kind=='photographic_adjustment'
            else _array(coverage,master.shape[:2]).copy())
        if kind=='mixed_regional' and (not np.any(edit>0) or not np.any(edit<1)):
            raise ValueError('Mixed candidate needs partial edit coverage')
        for top in range(0, len(master), 128):
            outside = (np.full(master[top:top+128].shape[:2],edit==0) if type(edit) is int else edit[top:top+128]==0)
            if not np.array_equal(master[top:top+128][outside], prior['master'][top:top+128][outside]):
                raise ValueError('Regional candidate changed float pixels outside permitted coverage')
        if kind=='photographic_adjustment':
            from photo_development import operator_for,producer_identity,iter_development_bands
            if (not isinstance(photographic_operation,dict) or set(photographic_operation)!={'operator','stats','producer'}
                    or photographic_operation['producer']!=producer_identity()
                    or photographic_operation['operator']!=operator_for(photographic_operation['operator'].get('parameters'))
                    or generation is not None):
                raise ValueError('Photographic appearance requires the actual checked CPU operation, not generation claims')
            counts={'changed_pixels':0,'covered_pixels':0,'unrecoverable_black_pixels':0,
                'linear_gain_would_exceed_sdr_pixels':0,'gamut_compressed_pixels':0}
            for top,expected,expected_coverage,stats in iter_development_bands(prior['master'],
                    photographic_operation['operator']['parameters'],cancel=cancel):
                actual_coverage=edit if type(edit) is int else edit[top:top+len(expected)]
                if (not np.array_equal(expected,master[top:top+len(expected)])
                        or not np.all(expected_coverage==actual_coverage)):
                    raise ValueError('Photographic pixels/coverage do not result from the actual native parent and checked operator')
                for key,value in stats.items():counts[key]+=value
            actual_stats=photographic_operation['stats']
            expected_stats={**counts,'native_pixels':size[0]*size[1],
                'band_rows':actual_stats.get('band_rows'),'physical_relighting_verified':False,
                'no_change':counts['changed_pixels']==0}
            if (actual_stats!=expected_stats or type(actual_stats.get('band_rows')) is not int
                    or not 1<=actual_stats['band_rows']<=512):
                raise ValueError('Photographic pixel statistics differ from actual computation')
            semantics=_photo_semantics(prior['receipt'])
            receipt['photographic_operation']={'source_only_photographic_lineage':semantics['source_only_photographic_lineage'],
                'parent_semantics':semantics}
            generated=(prior['generated_coverage'] if type(prior['generated_coverage']) is int else prior['generated_coverage'].copy())
            inherited=prior['receipt'].get('generation')
            if inherited:
                generation={**{role:{'path':str(Path(parent['path'])/inherited[role]['file']),
                    'sha256':inherited[role]['sha256']} for role in ('image','provider_receipt')},
                    'context_box':inherited['context_box'],'core_box':inherited['core_box']}
        else:
            if photographic_operation is not None:raise ValueError('A generated candidate cannot claim photographic-only processing')
            generated = np.asarray(prior['generated_coverage']+edit*(1-prior['generated_coverage']), np.float32)
    else:
        if coverage is not None or photographic_operation is not None or scope['work_type'] == 'retouch':
            raise ValueError('Full generated appearance requires composite permission and full ownership')
        edit = generated = 1
    def precommit(staging):
        validate_asset(parent, source_sha256=source_sha256, scope=scope, design_id=design_id)
        if kind in {'mixed_regional','photographic_adjustment'}:
            frozen = np.load(staging/'master.npy', mmap_mode='r', allow_pickle=False)
            for top in range(0, len(master), 128):
                outside=(np.full(frozen[top:top+128].shape[:2],edit==0) if type(edit) is int else edit[top:top+128]==0)
                if not np.array_equal(frozen[top:top+128][outside], prior['master'][top:top+128][outside]):
                    raise ValueError('Regional pixels changed during candidate freezing')
    return _publish(directory, master, receipt, scope=scope, edit=edit, generated=generated,
        generation=generation, cancel=cancel, precommit=precommit,photographic_operation=photographic_operation)
