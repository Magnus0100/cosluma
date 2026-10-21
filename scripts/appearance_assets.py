"""Current typed appearance IO dispatch; native validation has one owner."""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import threading
from time import perf_counter
from uuid import uuid4

import numpy as np

from asset_identity import sha256_file
import photo_appearance as native
import appearance_read_scope
import workflow_timing

DOMAIN=native.DOMAIN
_LIMIT=32
_LOCK=threading.RLock()
_PROOFS=OrderedDict()
_FLIGHTS={}
_EPOCH=0
_COUNTS={'calls':0,'hits':0,'cold_verifications':0,'waits':0,'failures':0,
    'bytes_hashed':0,'hash_seconds':0.,'math_seconds':0.,'loads':0,'copies':0}


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Appearance file requires its exact SHA256')
    return value


def _relative(value):
    if (not isinstance(value,str) or not value or '\\' in value or ':' in value
            or any(c in value for c in '<>"|?*') or any(ord(c)<32 for c in value)
            or PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix()!=value
            or any(part in {'.','..',''} or part.endswith((' ','.')) for part in value.split('/'))):
        raise ValueError('Appearance closure requires safe owned relative paths')
    return value


def _schema(asset):
    if (not isinstance(asset,dict) or set(asset)!={'path','asset_id','receipt_sha256'}
            or not isinstance(asset['path'],str) or not asset['path']):
        raise ValueError('Current typed appearance descriptor required')
    _sha(asset['asset_id']);_sha(asset['receipt_sha256'])
    path=Path(asset['path']).resolve()/'receipt.json'
    if not path.is_file() or path.stat().st_size>1024*1024 or sha256_file(path)!=asset['receipt_sha256']:
        raise ValueError('Appearance receipt changed or is unavailable')
    receipt=json.loads(path.read_text(encoding='utf8'))
    if not isinstance(receipt,dict):raise ValueError('Appearance receipt must be an object')
    return receipt.get('schema')


def _module(asset):
    schema=_schema(asset)
    if schema==native.SCHEMA:return native
    import geometric_appearance as geometric
    if schema==geometric.SCHEMA:return geometric
    raise ValueError('Unsupported current appearance schema')


def validate_asset(asset, *, source_sha256, scope, design_id):
    """Authoritative byte audit; reuse mathematics only for identical contents."""
    receipt,_=_validated(asset,source_sha256=source_sha256,scope=scope,design_id=design_id)
    return receipt


def load_asset(asset, *, source_sha256, scope, design_id):
    """Open verified, read-only typed buffers; never resample an existing asset."""
    binding={'source_sha256':source_sha256,'scope':scope,'design_id':design_id}
    receipt,snapshot=_validated(asset,**binding)
    directory=Path(asset['path']).absolute();loaded={}
    try:
        master=np.load(directory/receipt['files']['master']['file'],mmap_mode='r',allow_pickle=False)
        loaded['master']=master
        if master.dtype!=np.float32 or master.shape!=(receipt['canvas_size'][1],receipt['canvas_size'][0],3):
            raise ValueError('Verified appearance buffer header changed')
        def coverage(record):
            if set(record)=={'uniform'}:return record['uniform']
            value=np.load(directory/record['array']['file'],mmap_mode='r',allow_pickle=False)
            loaded[record['array']['file']]=value
            if value.dtype!=np.float32 or value.shape!=master.shape[:2]:
                raise ValueError('Verified coverage buffer header changed')
            return value
        generated=coverage(receipt['generated_coverage']);edit=coverage(receipt['edit_coverage'])
        _unchanged(snapshot,_capture(asset,**binding))
        _count(loads=1)
        return {'receipt':copy.deepcopy(receipt),'master':master,
            'preview_path':directory/receipt['files']['preview']['file'],
            'generated_coverage':generated,'edit_coverage':edit,'source_coverage':1-generated}
    except BaseException:
        _close(loaded);_discard(snapshot['key']);raise


def materialize_asset(asset, directory, *, source_sha256, scope, design_id):
    """Copy one complete verified closure atomically, without changing its IDs."""
    binding={'source_sha256':source_sha256,'scope':scope,'design_id':design_id}
    receipt,snapshot=_validated(asset,**binding)
    source=Path(asset['path']).absolute();directory=Path(directory).absolute()
    if directory.resolve().is_relative_to(source.resolve()):
        raise ValueError('Appearance copy destination must not mutate its source closure')
    _no_links(directory,missing=True)
    if directory.exists():raise FileExistsError('Appearance destination already exists')
    _no_links(directory.parent,missing=True)
    directory.parent.mkdir(parents=True,exist_ok=True);_no_links(directory.parent)
    claim=directory.with_name('.'+directory.name+'.claim');claim.mkdir()
    staging=directory.with_name('.'+directory.name+'.partial-'+uuid4().hex)
    try:
        staging.mkdir()
        with workflow_timing.span('appearance_owned_copy',model_request=False,http_request=False):
            for relative in snapshot['files']:
                target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source/relative,target)
            staged={**asset,'path':str(staging)}
            _,copied=_validated(staged,**binding)
            if copied['key']!=snapshot['key']:raise ValueError('Appearance copied closure changed')
            _unchanged(snapshot,_capture(asset,**binding))
            os.rename(staging,directory)
        _count(copies=1)
        return {**asset,'path':str(directory)}
    except BaseException:
        _discard(snapshot['key']);raise
    finally:
        shutil.rmtree(staging,ignore_errors=True);claim.rmdir()


def declared_files_from_receipt(receipt, receipt_sha):
    """Safe declared closure for unpacking; full validation follows extraction.

    This does not duplicate the native semantic validator. Only its explicit
    file-bearing fields are interpreted; arbitrary provenance is never walked.
    """
    _sha(receipt_sha)
    if not isinstance(receipt,dict):raise ValueError('Current appearance receipt required')
    if receipt.get('schema')!=native.SCHEMA:
        import geometric_appearance as geometric
        if receipt.get('schema')==geometric.SCHEMA:return geometric.declared_files_from_receipt(receipt,receipt_sha)
        raise ValueError('Unsupported current appearance schema')
    if receipt.get('kind') not in native.KINDS:raise ValueError('Unsupported native appearance kind')
    from photo_editing import identity
    if receipt.get('asset_id')!=identity({k:v for k,v in receipt.items() if k!='asset_id'}):
        raise ValueError('Native appearance content identity changed')
    result={'receipt.json':receipt_sha}
    def add(record):
        if not isinstance(record,dict) or set(record)!={'file','sha256'}:raise ValueError('Owned native file record required')
        name=_relative(record['file'])
        if '/' in name or name in result:raise ValueError('Native appearance needs unique owned basenames')
        result[name]=_sha(record['sha256'])
    files=receipt.get('files')
    if not isinstance(files,dict) or set(files)!={'master','preview'}:raise ValueError('Native appearance master/preview required')
    for key in ('master','preview'):add(files[key])
    for key in ('edit_coverage','generated_coverage'):
        coverage=receipt.get(key)
        if not isinstance(coverage,dict):raise ValueError('Native coverage declaration required')
        if set(coverage)=={'array'}:add(coverage['array'])
        elif set(coverage)!={'uniform'} or type(coverage['uniform']) is not int or coverage['uniform'] not in (0,1):
            raise ValueError('Invalid native coverage declaration')
    generation=receipt.get('generation')
    if generation is not None:
        if not isinstance(generation,dict) or set(generation)!={'image','provider_receipt','core_box','context_box'}:
            raise ValueError('Invalid native generation declaration')
        add(generation['image']);add(generation['provider_receipt'])
    operation=receipt.get('photographic_operation')
    if operation is not None:
        if not isinstance(operation,dict) or set(operation)!={'operation_receipt','source_only_photographic_lineage','parent_semantics'}:
            raise ValueError('Invalid native photographic declaration')
        add(operation['operation_receipt'])
    return result


def declared_files(asset, *, source_sha256, scope, design_id):
    receipt=validate_asset(asset,source_sha256=source_sha256,scope=scope,design_id=design_id)
    return declared_files_from_receipt(receipt,asset['receipt_sha256'])


def _count(**values):
    with _LOCK:
        for key,value in values.items():_COUNTS[key]+=value


def proof_cache_info():
    """Process-local measured counters, never evidence of artistic acceptance."""
    with _LOCK:
        return {**_COUNTS,'entries':len(_PROOFS),'inflight':len(_FLIGHTS),'capacity':_LIMIT,'pid':os.getpid()}


def clear_proof_cache():
    """Invalidate proofs, including work begun before this explicit reset."""
    global _EPOCH
    with _LOCK:
        _EPOCH+=1;_PROOFS.clear()
        for key in _COUNTS:_COUNTS[key]=0. if key.endswith('_seconds') else 0


def _discard(key):
    with _LOCK:_PROOFS.pop(key,None)


def _cancel(cancel):
    if cancel is not None and cancel():raise InterruptedError('Appearance audit cancelled')


def _close(buffers):
    seen=set()
    for value in buffers.values():
        mapped=getattr(value,'_mmap',None)
        if mapped is not None and id(mapped) not in seen:mapped.close();seen.add(id(mapped))


def _fingerprint(path):
    row=path.lstat()
    if (stat.S_ISLNK(row.st_mode) or getattr(row,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024)):
        raise ValueError('Appearance closure cannot contain a symlink or junction')
    return (row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns,row.st_mode)


def _no_links(path,*,missing=False):
    path=Path(path).absolute()
    for part in reversed((path,*path.parents)):
        try:_fingerprint(part)
        except FileNotFoundError:
            if not missing:raise
    return path


def _audit_file(path,cancel=None):
    _cancel(cancel);before=_fingerprint(path)
    if not stat.S_ISREG(before[-1]):raise ValueError('Appearance closure requires regular files')
    def read_bytes():
        started=perf_counter()
        # Outside a live OS-protected saved-input lease, including every
        # copied target and current producer/runtime, read actual bytes again.
        digest=sha256_file(path,force=True)
        _count(bytes_hashed=before[2],hash_seconds=perf_counter()-started)
        return digest
    digest=appearance_read_scope.audit(path,read_bytes,cancel=cancel)
    if _fingerprint(path)!=before:raise ValueError('Appearance file changed during its byte audit')
    _cancel(cancel)
    return digest,before


def _inventory(directory):
    """No-follow finite tree; extra files and even extra empty directories fail."""
    directory=_no_links(directory);files=set();directories=set();pending=[directory]
    while pending:
        current=pending.pop()
        if not stat.S_ISDIR(_fingerprint(current)[-1]):raise ValueError('Appearance root must be a real directory')
        with os.scandir(current) as rows:
            for row in rows:
                if len(files)+len(directories)>=256:raise ValueError('Appearance closure exceeds current finite schema')
                path=Path(row.path);mode=_fingerprint(path)[-1]
                relative=path.relative_to(directory).as_posix();_relative(relative)
                if stat.S_ISDIR(mode):directories.add(relative);pending.append(path)
                elif stat.S_ISREG(mode):files.add(relative)
                else:raise ValueError('Appearance closure contains a special file')
    return files,directories


def _current_verifier(schema,cancel=None):
    """Observe current code/runtime bytes without changing any pixel producer."""
    from runtime_resources import code_path
    import geometric_appearance as geometric
    paths={'appearance_assets.py':Path(__file__),
        'appearance_read_scope.py':Path(appearance_read_scope.__file__)}
    paths.update({name:Path(native.__file__).with_name(name) for name in native._PRODUCER_FILES})
    if schema==geometric.SCHEMA:
        import face_contour_deform as mapping
        import photo_shaping_geometry as geometry
        paths['geometric_appearance.py']=Path(geometric.__file__)
        recorded=mapping.producer_identity()
        for name in {*recorded['code'],*geometry._producer()}:paths[name]=Path(code_path(name))
        distributions={}
        for package in ('numpy','scipy'):
            distribution=importlib.metadata.distribution(package)
            distributions[package]=distribution
            members=[item for item in distribution.files or () if str(item).endswith('.dist-info/METADATA')]
            if len(members)!=1:raise ValueError('Appearance numerical runtime metadata is ambiguous')
            paths[package+'/METADATA']=Path(distribution.locate_file(members[0]))
        for name in ('numpy._core._multiarray_umath','scipy.ndimage._nd_image','scipy.spatial._qhull'):
            paths[name]=Path(importlib.import_module(name).__file__)
    audited={name:_audit_file(path,cancel) for name,path in sorted(paths.items())}
    observed={name:value[0] for name,value in audited.items()}
    # Build the expected producer directly from forced byte observations;
    # never let a helper's stat-bound memo describe the current code/runtime.
    native_producer={name:observed[name] for name in native._PRODUCER_FILES}
    if schema==geometric.SCHEMA:
        current_mapping=copy.deepcopy(recorded)
        current_mapping['code']={name:observed[name] for name in recorded['code']}
        current_mapping['runtime']={name:{'version':distributions[name].version,
            'metadata_sha256':observed[name+'/METADATA']} for name in distributions}
        current_mapping['native_sampling_binaries']={name:observed[name] for name in recorded['native_sampling_binaries']}
        current_mapping['numpy_version']=np.__version__
        current_mapping['scipy_version']=importlib.import_module('scipy').__version__
        producer={'geometric_appearance.py':observed['geometric_appearance.py'],
            'mapping':current_mapping,'native':native_producer}
        canonical={name:observed[name] for name in geometry._producer()}
    else:producer=native_producer;canonical=None
    if any(_fingerprint(paths[name])!=value[1] for name,value in audited.items()):
        raise ValueError('Appearance verifier/runtime changed during audit')
    parameters=({'mapping':copy.deepcopy(mapping.PARAMETERS),'canonical':copy.deepcopy(geometry.PARAMETERS)}
                if schema==geometric.SCHEMA else None)
    return {'files':observed,'producer':producer,'canonical':canonical,'parameters':parameters}


def _native_operation_facts(directory,receipt,cancel=None):
    """Current operation dependencies of owned photographic U, before any hit.

    Native remains the sole semantic validator. Its private stat memo does
    not cover external photographic code/schema identities, so neither a warm
    geometric proof nor its cold native-parent check can establish these.
    """
    if receipt['schema']==native.SCHEMA:return None
    owned=receipt['owned_parent'];parent_directory=directory/owned['directory']
    parent_path=parent_directory/'receipt.json'
    if parent_path.stat().st_size>1024*1024:raise ValueError('Owned parent receipt exceeds metadata limit')
    parent_bytes=parent_path.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest()!=owned['receipt_sha256']:
        raise ValueError('Owned native parent receipt changed during audit')
    parent=json.loads(parent_bytes)
    if not isinstance(parent,dict):raise ValueError('Owned native parent receipt must be an object')
    if parent.get('kind')!='photographic_adjustment':return None
    import photo_development as development
    import photo_edit_session as session
    names=('photo_development.py','photo_appearance.py','photo_edit_session.py','photo_session_evidence.py')
    observations={name:_audit_file(Path(development.__file__).with_name(name),cancel) for name in names}
    current={name:value[0] for name,value in observations.items()}
    metadata=parent.get('photographic_operation')
    if not isinstance(metadata,dict):raise ValueError('Owned photographic U requires its operation record')
    record=metadata.get('operation_receipt')
    if not isinstance(record,dict) or set(record)!={'file','sha256'}:
        raise ValueError('Owned photographic U requires an actual operation file')
    relative=_relative(record['file']);_sha(record['sha256'])
    if '/' in relative:raise ValueError('Owned photographic operation must be a basename')
    path=parent_directory/relative
    if path.stat().st_size>1024*1024:raise ValueError('Photographic operation exceeds metadata limit')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:
        raise ValueError('Owned photographic operation bytes changed during audit')
    operation=json.loads(data)
    if not isinstance(operation,dict) or operation.get('producer')!=current:
        raise ValueError('Owned photographic U operation producer is not current')
    parameters=copy.deepcopy(development.PARAMETERS)
    session_parameters=copy.deepcopy(session.PHOTOGRAPHIC_PARAMETERS)
    if parameters!=session_parameters:
        raise ValueError('Owned photographic U current parameter schemas differ')
    # This owner checks operator_for, source/scope/design, direct parent,
    # output RGB/preview, coverage semantics, flags and parameter bounds even
    # when native.validate_asset itself would take its old stat-memo branch.
    checked=native._validate_photo_operation(parent_directory,parent)
    if checked!=operation:raise ValueError('Owned photographic U operation changed during validation')
    if any(_fingerprint(Path(development.__file__).with_name(name))!=value[1]
           for name,value in observations.items()):
        raise ValueError('Owned photographic U current producer changed during audit')
    _cancel(cancel)
    return {'producer':current,'parameters':parameters,'session_parameters':session_parameters,
        'operation_receipt_sha256':record['sha256'],'operator':copy.deepcopy(checked['operator'])}


def _capture(asset,*,source_sha256,scope,design_id,cancel=None):
    """Observe the complete byte closure afresh, independently of proof memo."""
    from photo_editing import identity,validate_scope
    _cancel(cancel);scope=validate_scope(scope);_sha(source_sha256);_sha(design_id)
    if (not isinstance(asset,dict) or set(asset)!={'path','asset_id','receipt_sha256'}
            or not isinstance(asset['path'],str) or not asset['path']):
        raise ValueError('Current typed appearance descriptor required')
    _sha(asset['asset_id']);_sha(asset['receipt_sha256'])
    directory=_no_links(Path(asset['path']).absolute());receipt_path=directory/'receipt.json'
    if receipt_path.stat().st_size>1024*1024:raise ValueError('Appearance receipt exceeds metadata limit')
    digest,receipt_stat=_audit_file(receipt_path,cancel)
    if digest!=asset['receipt_sha256']:raise ValueError('Appearance receipt changed or is unavailable')
    data=receipt_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest or _fingerprint(receipt_path)!=receipt_stat:
        raise ValueError('Appearance receipt changed while being parsed')
    receipt=json.loads(data)
    if not isinstance(receipt,dict):raise ValueError('Appearance receipt must be an object')
    if (receipt.get('asset_id')!=asset['asset_id'] or receipt.get('source_sha256')!=source_sha256
            or receipt.get('scope_id')!=scope['scope_id'] or receipt.get('design_id')!=design_id):
        raise ValueError('Appearance source/scope/design/asset binding changed')
    declared=declared_files_from_receipt(receipt,digest)
    # Current schemas have finite shallow owned groups; derive all parents
    # rather than accepting unrelated children or empty directories.
    expected_dirs={parent.as_posix() for name in declared for parent in PurePosixPath(name).parents
                   if parent.as_posix()!='.'}
    inventory=_inventory(directory)
    if inventory!=(set(declared),expected_dirs):raise ValueError('Appearance closure has missing or extra members')
    fingerprints={};observed={}
    for relative,expected in sorted(declared.items()):
        actual,signature=_audit_file(directory/relative,cancel)
        if actual!=expected:raise ValueError('Appearance closure actual bytes differ from declared SHA')
        observed[relative]=actual;fingerprints[relative]=signature
    verifier=_current_verifier(receipt.get('schema'),cancel)
    if receipt.get('producer')!=verifier['producer']:
        raise ValueError('Appearance current producer differs from frozen receipt')
    if verifier['canonical'] is not None:
        operation=receipt['operation'];support_record=operation['support']
        support_bytes=(directory/support_record['file']).read_bytes()
        if hashlib.sha256(support_bytes).hexdigest()!=support_record['sha256']:
            raise ValueError('Appearance canonical support changed during audit')
        support=json.loads(support_bytes)
        if not isinstance(support,dict):raise ValueError('Appearance canonical support must be an object')
        support_binding=support.get('binding');metadata=operation.get('mapping_metadata')
        if not isinstance(support_binding,dict) or not isinstance(metadata,dict):
            raise ValueError('Appearance canonical support and mapping metadata required')
        if (support_binding.get('producer')!=verifier['canonical']
                or support_binding.get('parameters')!=verifier['parameters']['canonical']):
            raise ValueError('Appearance canonical support producer/parameters are not current')
        if (metadata.get('producer')!=verifier['producer']['mapping']
                or metadata.get('parameters')!=verifier['parameters']['mapping']):
            raise ValueError('Appearance mapping producer/parameters are not current')
    native_operation=_native_operation_facts(directory,receipt,cancel)
    if _inventory(directory)!=inventory:raise ValueError('Appearance closure changed during audit')
    if any(_fingerprint(directory/name)!=signature for name,signature in fingerprints.items()):
        raise ValueError('Appearance input changed during closure audit')
    facts={'receipt_sha256':digest,'asset_id':asset['asset_id'],'files':observed,
        'source_sha256':source_sha256,'scope':scope,'design_id':design_id,
        'canvas_size':receipt.get('canvas_size'),'operation':receipt.get('operation'),
        'verifier':verifier,'native_operation':native_operation}
    return {'key':identity(facts),'facts':facts,'files':observed,'receipt':receipt,
        'fingerprints':fingerprints,'directory':str(directory)}


def _unchanged(before,after):
    if (before['key']!=after['key'] or before['directory']!=after['directory']
            or before['fingerprints']!=after['fingerprints']):
        _discard(before['key']);raise ValueError('Appearance input changed during verified operation')


def _validated(asset,*,source_sha256,scope,design_id,cancel=None):
    from photo_editing import validate_scope
    binding={'source_sha256':source_sha256,'scope':copy.deepcopy(validate_scope(scope)),'design_id':design_id}
    asset=copy.deepcopy(asset);_count(calls=1)
    try:
        with workflow_timing.span('appearance_closure_byte_audit',model_request=False,http_request=False):
            snapshot=_capture(asset,cancel=cancel,**binding)
    except BaseException:
        _count(failures=1);raise
    key=snapshot['key'];geometric=snapshot['receipt']['schema']!=native.SCHEMA
    flight=None;owner=False
    try:
        if geometric:
            while True:
                _cancel(cancel)
                with _LOCK:
                    if key in _PROOFS:
                        receipt=copy.deepcopy(_PROOFS[key]);_PROOFS.move_to_end(key);_COUNTS['hits']+=1
                        return receipt,snapshot
                    flight=_FLIGHTS.get(key)
                    if flight is None:
                        flight={'event':threading.Event(),'epoch':_EPOCH};_FLIGHTS[key]=flight;owner=True;break
                    _COUNTS['waits']+=1
                # A worker can wait for the one mathematical verifier. GUI
                # callers must schedule this authoritative API off-thread.
                while not flight['event'].wait(.05):_cancel(cancel)
                _cancel(cancel)
                _unchanged(snapshot,_capture(asset,cancel=cancel,**binding))
        module=_module(asset)
        started=perf_counter();_count(cold_verifications=int(geometric))
        with workflow_timing.span('appearance_mathematical_verification',model_request=False,http_request=False,
                proof_cache_hit=False):
            try:receipt=module.validate_asset(asset,**binding)
            finally:_count(math_seconds=perf_counter()-started)
        _cancel(cancel)
        with workflow_timing.span('appearance_closure_post_audit',model_request=False,http_request=False):
            _unchanged(snapshot,_capture(asset,cancel=cancel,**binding))
        if receipt!=snapshot['receipt']:raise ValueError('Appearance verified receipt changed')
        if geometric:
            with _LOCK:
                if flight['epoch']==_EPOCH:
                    _PROOFS[key]=copy.deepcopy(receipt);_PROOFS.move_to_end(key)
                    while len(_PROOFS)>_LIMIT:_PROOFS.popitem(last=False)
        return copy.deepcopy(receipt),snapshot
    except BaseException:
        _discard(key);_count(failures=1);raise
    finally:
        if owner:
            with _LOCK:
                _FLIGHTS.pop(key,None);flight['event'].set()


def force_audit(asset,*,source_sha256,scope,design_id,cancel=None):
    """Explicit authoritative byte audit, with optional cancellation of waits."""
    receipt,_=_validated(asset,source_sha256=source_sha256,scope=scope,design_id=design_id,cancel=cancel)
    return receipt


def bound_snapshot(asset,*,source_sha256,scope,design_id,cancel=None):
    """Return copied verification facts, never an unchecked-load capability."""
    _,snapshot=_validated(asset,source_sha256=source_sha256,scope=scope,design_id=design_id,cancel=cancel)
    return copy.deepcopy({'snapshot_id':snapshot['key'],**snapshot['facts']})
