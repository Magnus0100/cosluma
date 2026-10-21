"""Immutable deterministic geometry applied to one owned native parent U.

No original decode, detector, image generator or historical validator runs here.
Native U is validated by its sole native owner; geometry is independently rebuilt.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from asset_identity import sha256_file
from photo_editing import identity, validate_scope
import photo_appearance as native

SCHEMA='cosluma_geometric_appearance.v1'
KIND='geometric_transform'
DOMAIN=native.DOMAIN
OPERATION_SCHEMA='cosluma_geometric_transform.v1'
EDIT_SEMANTICS='geometric_displacement_support_not_generated_pixel_ownership'
_KEYS={'schema','kind','source_sha256','scope_id','design_id','working_domain','canvas_size','producer','parent',
    'source_preparation','source_origin','generation','generated_resolution','source_coverage_semantics',
    'native_generated_detail','human_approved','geometry_verified','physical_relighting_verified','files',
    'edit_coverage','generated_coverage','photographic_operation','owned_parent','operation','asset_id'}
_OP_KEYS={'schema','amount','box','support','detections','support_masks','mapping','mapping_metadata',
    'edit_coverage_semantics','image_generation_submissions','semantic_truth_claimed','naturalness_verified'}


def _cancel(cancel):
    if cancel is not None and cancel():raise InterruptedError('Geometric appearance preparation cancelled')


def _amount(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Geometric amount requires a finite normalized non-boolean number')
    return float(value)


def _close(loaded):
    if loaded is None:return
    closed=set()
    for value in loaded.values():
        mmap=getattr(value,'_mmap',None)
        if mmap is not None and id(mmap) not in closed:mmap.close();closed.add(id(mmap))


def _coverage(directory,record,shape):
    if isinstance(record,dict) and set(record)=={'array'}:
        value=np.load(_read_file(directory,record['array']),mmap_mode='r',allow_pickle=False)
        try:return native._array(value,shape)
        except BaseException:
            _close({'coverage':value});raise
    return native._coverage(directory,record,shape)


def _producer():
    from face_contour_deform import producer_identity
    return {'geometric_appearance.py':sha256_file(__file__),'mapping':producer_identity(),
        'native':native._producer()}


def _json(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _record(path):
    return {'file':Path(path).name,'sha256':sha256_file(path)}


def _read_file(directory,record):
    from appearance_assets import _relative,_sha
    if not isinstance(record,dict) or set(record)!={'file','sha256'}:raise ValueError('Geometric owned file record required')
    relative=_relative(record['file']);_sha(record['sha256'])
    if '/' in relative:raise ValueError('Geometric local evidence requires a basename')
    path=Path(directory)/relative
    if not path.resolve().is_relative_to(Path(directory).resolve()) or sha256_file(path)!=record['sha256']:
        raise ValueError('Geometric owned file bytes changed')
    return path


def _header(receipt,receipt_sha):
    from appearance_assets import _relative,_sha
    _sha(receipt_sha)
    if not isinstance(receipt,dict) or set(receipt)!=_KEYS:raise ValueError('Current complete geometric receipt required')
    for key in ('source_sha256','scope_id','design_id','asset_id'):_sha(receipt[key])
    if (receipt['schema']!=SCHEMA or receipt['kind']!=KIND or receipt['working_domain']!=DOMAIN
            or receipt['asset_id']!=identity({k:v for k,v in receipt.items() if k!='asset_id'})
            or receipt['generation'] is not None or receipt['generated_resolution'] is not None
            or receipt['source_preparation'] is not None or receipt['photographic_operation'] is not None
            or any(receipt[key] is not False for key in ('native_generated_detail','human_approved','geometry_verified','physical_relighting_verified'))
            or receipt['source_coverage_semantics']!='one_minus_cumulative_generated_coverage_not_unprocessed_light'):
        raise ValueError('Geometric receipt cannot claim generation, relighting or human acceptance')
    size=receipt['canvas_size']
    if not isinstance(size,list) or len(size)!=2 or any(type(v) is not int or v<1 for v in size):
        raise ValueError('Geometric native canvas required')
    parent=receipt['parent'];owned=receipt['owned_parent'];operation=receipt['operation']
    if not isinstance(parent,dict) or set(parent)!={'asset_id','master_sha256','preview_sha256'}:
        raise ValueError('Geometric parent must bind actual native U pixels')
    for value in parent.values():_sha(value)
    if (not isinstance(owned,dict) or set(owned)!={'directory','asset_id','receipt_sha256','files'}
            or owned['directory']!='parent' or owned['asset_id']!=parent['asset_id']):
        raise ValueError('Geometric appearance must own one native parent group')
    _sha(owned['asset_id']);_sha(owned['receipt_sha256'])
    if not isinstance(owned['files'],dict) or owned['files'].get('receipt.json')!=owned['receipt_sha256']:
        raise ValueError('Complete native parent file declaration required')
    for name,digest in owned['files'].items():
        if '/' in _relative(name):raise ValueError('Geometric parent must be the current flat native group')
        _sha(digest)
    if not isinstance(operation,dict) or set(operation)!=_OP_KEYS or operation['schema']!=OPERATION_SCHEMA:
        raise ValueError('Current deterministic geometric operation required')
    if (type(operation['amount']) is not float or _amount(operation['amount'])!=operation['amount']
            or operation['edit_coverage_semantics']!=EDIT_SEMANTICS
            or type(operation['image_generation_submissions']) is not int or operation['image_generation_submissions']!=0
            or operation['semantic_truth_claimed'] is not False or operation['naturalness_verified'] is not False):
        raise ValueError('Geometric operation must retain its explicit amount and zero generation cost')
    box=operation['box']
    if (not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box)
            or not 0<=box[0]<box[2]<=size[0] or not 0<=box[1]<box[3]<=size[1]):
        raise ValueError('Geometric operation box must be actual native coordinates')
    return receipt


def declared_files_from_receipt(receipt,receipt_sha):
    """Finite declared closure only; never walk historical arbitrary metadata."""
    from appearance_assets import _relative,_sha
    _header(receipt,receipt_sha)
    result={'receipt.json':receipt_sha}
    def add(record):
        if not isinstance(record,dict) or set(record)!={'file','sha256'}:raise ValueError('Explicit geometric file declaration required')
        name=_relative(record['file'])
        if '/' in name or name in result:raise ValueError('Geometric files must be unique owned basenames')
        result[name]=_sha(record['sha256'])
    files=receipt['files']
    if not isinstance(files,dict) or set(files)!={'master','preview'}:raise ValueError('Geometric master/preview required')
    for key in ('master','preview'):add(files[key])
    for key in ('edit_coverage','generated_coverage'):
        value=receipt[key]
        if not isinstance(value,dict):raise ValueError('Geometric coverage declaration required')
        if set(value)=={'array'}:add(value['array'])
        elif set(value)!={'uniform'} or type(value['uniform']) is not int or value['uniform'] not in (0,1):
            raise ValueError('Invalid geometric coverage declaration')
    for key in ('support','detections','support_masks','mapping'):add(receipt['operation'][key])
    for name,digest in receipt['owned_parent']['files'].items():result['parent/'+name]=digest
    return result


def _parent(directory,receipt,binding):
    from appearance_assets import declared_files
    owned=receipt['owned_parent'];parent_directory=Path(directory)/owned['directory']
    if not parent_directory.resolve().is_relative_to(Path(directory).resolve()):raise ValueError('Owned parent escaped asset')
    asset={'path':str(parent_directory),'asset_id':owned['asset_id'],'receipt_sha256':owned['receipt_sha256']}
    # Deliberately dispatch directly to the one active native implementation.
    # Geometric parents are never recursively admitted as a fresh U.
    prior=native.validate_asset(asset,**binding)
    if declared_files(asset,**binding)!=owned['files']:raise ValueError('Owned native parent declaration changed')
    actual={'asset_id':prior['asset_id'],'master_sha256':prior['files']['master']['sha256'],
        'preview_sha256':prior['files']['preview']['sha256']}
    if actual!=receipt['parent'] or prior['canvas_size']!=receipt['canvas_size'] or prior['source_origin']!=receipt['source_origin']:
        raise ValueError('Geometric appearance no longer binds its actual U/source origin')
    return asset,prior


def _support(directory,operation,base,prior,binding):
    import photo_shaping_geometry as geometry
    support=json.loads(_read_file(directory,operation['support']).read_text('utf8'))
    detections=json.loads(_read_file(directory,operation['detections']).read_text('utf8'))
    mask_path=_read_file(directory,operation['support_masks'])
    expected_base={'asset_id':base['asset_id'],'receipt_sha256':base['receipt_sha256'],
        'source_sha256':binding['source_sha256'],'scope_id':binding['scope']['scope_id'],'design_id':binding['design_id'],
        'master_sha256':prior['files']['master']['sha256'],'preview_sha256':prior['files']['preview']['sha256'],
        'native_size':prior['canvas_size']}
    observed=support.get('binding') or {}
    if (set(support)!={'schema','binding','geometry','detections','masks','support_id'}
            or support['schema']!=geometry.SCHEMA
            or support['support_id']!=identity({k:v for k,v in support.items() if k!='support_id'})
            or set(observed)!={'base','runtime','producer','parameters'}
            or observed.get('base')!=expected_base or observed.get('producer')!=geometry._producer()
            or observed.get('parameters')!=geometry.PARAMETERS
            or (support.get('detections') or {}).get('sha256')!=operation['detections']['sha256']
            or (support.get('masks') or {}).get('sha256')!=operation['support_masks']['sha256']):
        raise ValueError('Geometric support differs from its actual U/current canonical producer')
    runtime=observed['runtime']
    if not isinstance(runtime,dict) or set(runtime)!={'python_path','python_sha256','model_path','model_sha256',
            'mediapipe_metadata_path','mediapipe_metadata_sha256','mediapipe_version'}:
        raise ValueError('Original localization runtime provenance required')
    for key in ('python_sha256','model_sha256','mediapipe_metadata_sha256'):native._sha(runtime[key])
    for key in ('python_path','model_path','mediapipe_metadata_path','mediapipe_version'):
        if not isinstance(runtime[key],str) or not runtime[key]:raise ValueError('Incomplete original localization provenance')
    geometry._validate_detection(detections,observed)
    expected,metadata=geometry._geometry(detections,prior['canvas_size'])
    if metadata!=support['geometry']:raise ValueError('Owned canonical geometry differs from actual detections')
    with np.load(mask_path,allow_pickle=False) as planes:
        if set(planes.files)!=set(expected) or any(planes[key].dtype!=value.dtype
                or not np.array_equal(planes[key],value) for key,value in expected.items()):
            raise ValueError('Owned canonical support planes differ from deterministic geometry')
    return detections,support


def _mapping_parts(mapping):
    return ({k:copy.deepcopy(v) for k,v in mapping.items() if not isinstance(v,np.ndarray)},
        {k:v for k,v in mapping.items() if isinstance(v,np.ndarray)})


def _edit_support(mapping,size):
    edit=np.zeros((size[1],size[0]),np.float32);x0,y0,x1,y1=mapping['box']
    support=mapping['edit_support']
    if support.dtype!=bool or support.shape!=(y1-y0,x1-x0):raise ValueError('Actual geometric displacement support required')
    edit[y0:y1,x0:x1]=support
    return edit


def _same(actual,expected):
    return (type(actual) is int and type(expected) is int and actual==expected
        or isinstance(actual,np.ndarray) and np.array_equal(actual,expected)
        or isinstance(expected,np.ndarray) and type(actual) is int and np.all(expected==actual)
        or isinstance(actual,np.ndarray) and type(expected) is int and np.all(actual==expected))


def validate_asset(asset, *, source_sha256, scope, design_id):
    from appearance_assets import _schema
    from face_contour_deform import build_mapping,sample_mapping
    scope=validate_scope(scope);binding={'source_sha256':source_sha256,'scope':scope,'design_id':design_id}
    if _schema(asset)!=SCHEMA:raise ValueError('Current geometric asset required')
    directory=Path(asset['path']).resolve();receipt=json.loads((directory/'receipt.json').read_text('utf8'))
    _header(receipt,asset['receipt_sha256'])
    if (receipt['asset_id']!=asset['asset_id'] or receipt['source_sha256']!=source_sha256
            or receipt['scope_id']!=scope['scope_id'] or receipt['design_id']!=design_id or receipt['producer']!=_producer()):
        raise ValueError('Geometric appearance source/scope/design/producer changed')
    closure=declared_files_from_receipt(receipt,asset['receipt_sha256'])
    for relative,digest in closure.items():
        path=directory/relative
        if not path.resolve().is_relative_to(directory) or sha256_file(path)!=digest:
            raise ValueError('Geometric asset closure changed')
    parent,prior=_parent(directory,receipt,binding);operation=receipt['operation']
    detections,_=_support(directory,operation,parent,prior,binding)
    mapping=build_mapping(detections,receipt['canvas_size'],operation['amount'])
    metadata,arrays=_mapping_parts(mapping)
    if metadata!=operation['mapping_metadata'] or mapping['box']!=operation['box']:
        raise ValueError('Geometric mapping metadata differs from actual bound detections')
    with np.load(_read_file(directory,operation['mapping']),allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays) or any(saved[key].dtype!=value.dtype
                or not np.array_equal(saved[key],value) for key,value in arrays.items()):
            raise ValueError('Geometric mapping differs from deterministic actual-U reconstruction')
    loaded=native.load_asset(parent,**binding)
    try:expected_rgb,expected_g=sample_mapping(loaded['master'],loaded['generated_coverage'],mapping)
    finally:_close(loaded)
    buffers={}
    try:
        master=np.load(_read_file(directory,receipt['files']['master']),mmap_mode='r',allow_pickle=False);buffers['master']=master
        native._array(master)
        if native._size(master)!=receipt['canvas_size'] or not np.array_equal(master,expected_rgb):
            raise ValueError('Geometric RGB is not the actual resampled U master')
        edit=_coverage(directory,receipt['edit_coverage'],master.shape[:2]);buffers['edit']=edit
        generated=_coverage(directory,receipt['generated_coverage'],master.shape[:2]);buffers['generated']=generated
        if not _same(edit,_edit_support(mapping,receipt['canvas_size'])) or not _same(generated,expected_g):
            raise ValueError('Geometric coverage must be actual support and identically sampled parent ownership')
        native._preview_matches(master,_read_file(directory,receipt['files']['preview']))
    finally:_close(buffers)
    return copy.deepcopy(receipt)


def load_asset(asset, *, source_sha256, scope, design_id):
    receipt=validate_asset(asset,source_sha256=source_sha256,scope=scope,design_id=design_id)
    directory=Path(asset['path']).resolve()
    master=np.load(directory/receipt['files']['master']['file'],mmap_mode='r',allow_pickle=False)
    generated=_coverage(directory,receipt['generated_coverage'],master.shape[:2])
    return {'receipt':receipt,'master':master,'preview_path':directory/receipt['files']['preview']['file'],
        'edit_coverage':_coverage(directory,receipt['edit_coverage'],master.shape[:2]),
        'generated_coverage':generated,'source_coverage':1-generated}


def _apply_runtime(support):
    runtime=support['binding']['runtime']
    for path_key,sha_key in (('python_path','python_sha256'),('model_path','model_sha256'),
                            ('mediapipe_metadata_path','mediapipe_metadata_sha256')):
        path=Path(runtime[path_key])
        if not path.is_file() or sha256_file(path)!=runtime[sha_key]:
            raise ValueError('Current geometric apply localization runtime changed or is unavailable')


def create_asset(directory, *, base_appearance, support_descriptor, amount, source_sha256, scope, design_id, cancel=None):
    from face_contour_deform import build_mapping,sample_mapping
    from appearance_assets import _schema,declared_files
    from raw_support import linear_preview
    scope=validate_scope(scope);amount=_amount(amount);binding={'source_sha256':source_sha256,'scope':scope,'design_id':design_id}
    _cancel(cancel)
    if _schema(base_appearance)!=native.SCHEMA:raise ValueError('A new geometric result must start directly from native U')
    prior=native.validate_asset(base_appearance,**binding)
    if (not isinstance(support_descriptor,dict) or set(support_descriptor)!={'path','sha256'}
            or sha256_file(support_descriptor['path'])!=support_descriptor['sha256']):
        raise ValueError('Actual bound canonical support required')
    original_support=json.loads(Path(support_descriptor['path']).read_text('utf8'));_apply_runtime(original_support)
    directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError('Geometric destination already exists')
    directory.parent.mkdir(parents=True,exist_ok=True);claim=directory.with_name('.'+directory.name+'.claim');claim.mkdir()
    staging=directory.with_name('.'+directory.name+'.partial-'+uuid4().hex)
    try:
        staging.mkdir();owned=native.materialize_asset(base_appearance,staging/'parent',**binding)
        operation={'schema':OPERATION_SCHEMA,'amount':amount,'box':None,'support':None,'detections':None,'support_masks':None,
            'mapping':None,'mapping_metadata':None,'edit_coverage_semantics':EDIT_SEMANTICS,
            'image_generation_submissions':0,'semantic_truth_claimed':False,'naturalness_verified':False}
        for role,name,record in (('support','geometry-support.json',support_descriptor),
            ('detections','detections.json',original_support['detections']),('support_masks','support-masks.npz',original_support['masks'])):
            source=Path(record['path'])
            if sha256_file(source)!=record['sha256']:raise ValueError('Canonical support changed during owning')
            shutil.copyfile(source,staging/name);operation[role]=_record(staging/name)
            if operation[role]['sha256']!=record['sha256']:raise ValueError('Canonical support copy changed')
        detections,_=_support(staging,operation,owned,prior,binding)
        mapping=build_mapping(detections,prior['canvas_size'],amount,cancel=cancel)
        metadata,arrays=_mapping_parts(mapping);operation.update(box=copy.deepcopy(mapping['box']),mapping_metadata=metadata)
        np.savez_compressed(staging/'mapping.npz',**arrays);operation['mapping']=_record(staging/'mapping.npz')
        loaded=native.load_asset(owned,**binding)
        try:master,generated=sample_mapping(loaded['master'],loaded['generated_coverage'],mapping,cancel=cancel)
        finally:_close(loaded)
        native._array(master);np.save(staging/'master.npy',master,allow_pickle=False)
        preview=linear_preview(master);preview.save(staging/'preview.png',format='PNG',icc_profile=preview.info['icc_profile']);preview.close()
        receipt={'schema':SCHEMA,'kind':KIND,'source_sha256':source_sha256,'scope_id':scope['scope_id'],'design_id':design_id,
            'working_domain':DOMAIN,'canvas_size':copy.deepcopy(prior['canvas_size']),'producer':_producer(),
            'parent':{'asset_id':prior['asset_id'],'master_sha256':prior['files']['master']['sha256'],
                'preview_sha256':prior['files']['preview']['sha256']},
            'owned_parent':{'directory':'parent','asset_id':prior['asset_id'],'receipt_sha256':owned['receipt_sha256'],
                'files':declared_files(owned,**binding)},'source_preparation':None,'source_origin':copy.deepcopy(prior['source_origin']),
            'generation':None,'generated_resolution':None,'photographic_operation':None,
            'source_coverage_semantics':'one_minus_cumulative_generated_coverage_not_unprocessed_light',
            'native_generated_detail':False,'human_approved':False,'geometry_verified':False,'physical_relighting_verified':False,
            'files':{'master':_record(staging/'master.npy'),'preview':_record(staging/'preview.png')},
            'edit_coverage':native._save_coverage(staging,'edit-coverage',_edit_support(mapping,prior['canvas_size'])),
            'generated_coverage':native._save_coverage(staging,'generated-coverage',generated),'operation':operation}
        receipt['asset_id']=identity(receipt);_json(staging/'receipt.json',receipt)
        asset={'path':str(staging),'asset_id':receipt['asset_id'],'receipt_sha256':sha256_file(staging/'receipt.json')}
        validate_asset(asset,**binding);_cancel(cancel)
        native.validate_asset(base_appearance,**binding);_apply_runtime(original_support)
        if sha256_file(support_descriptor['path'])!=support_descriptor['sha256']:raise ValueError('Original support changed before publication')
        os.rename(staging,directory)
        return {**asset,'path':str(directory)}
    finally:
        shutil.rmtree(staging,ignore_errors=True);claim.rmdir()


def materialize_asset(asset, directory, *, source_sha256, scope, design_id):
    binding={'source_sha256':source_sha256,'scope':scope,'design_id':design_id}
    receipt=validate_asset(asset,**binding);source=Path(asset['path']).resolve();directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError('Geometric destination already exists')
    directory.parent.mkdir(parents=True,exist_ok=True);claim=directory.with_name('.'+directory.name+'.claim');claim.mkdir()
    staging=directory.with_name('.'+directory.name+'.partial-'+uuid4().hex)
    try:
        staging.mkdir()
        for relative,digest in declared_files_from_receipt(receipt,asset['receipt_sha256']).items():
            target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,target)
            if sha256_file(target)!=digest:raise ValueError('Geometric bytes changed during materialization')
        validate_asset({**asset,'path':str(staging)},**binding);validate_asset(asset,**binding)
        os.rename(staging,directory)
        return {**asset,'path':str(directory)}
    finally:
        shutil.rmtree(staging,ignore_errors=True);claim.rmdir()
