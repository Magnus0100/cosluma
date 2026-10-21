"""Actual-U-bound canonical contour support for explicitly selected shaping.

Landmarks supply geometric protection, not semantic truth or visual acceptance.
Only the existing CPU detect-only worker is used; no parser or beauty model runs.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError

from asset_identity import sha256_file
from photo_editing import identity
from runtime_resources import code_path, external_python_popen, isolated_python_environment

SCHEMA='cosluma_shaping_geometry.v1'
REGION_SCHEMA='cosluma_shaping_geometry_region.v1'
PARAMETERS={'inner_width_face_fraction':.055,'outer_width_face_fraction':.025,
    'feature_margin_face_fraction':.018,'arc_endpoint_taper_fraction':.15,
    'edge_feather_fraction_of_band':.35,'face_envelope_padding_fraction':.08,
    'outside_envelope_ramp_face_fraction':.04,'minimum_face_span':32,
    'minimum_feature_span_face_fraction':.015,
    'feature_protection':'canonical_point_group_convex_hulls_and_conservative_raster_union_plus_margin',
    'maximum_faces':2,'maximum_worker_seconds':180,
    'geometry_strength':'protection support only; not a calibrated slimming percentage'}
NOSE=(168,6,197,195,5,4,1,19,94,2,98,97,326,327)


def _cancel(cancel):
    if cancel is not None and cancel():raise InterruptedError('Cancelled shaping geometry preparation')


def _json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def _record(path):
    return {'path':str(Path(path).resolve()),'sha256':sha256_file(path)}


def _read(record):
    if not isinstance(record,dict) or set(record)!={'path','sha256'}:
        raise ValueError('Content-bound shaping geometry evidence required')
    path=Path(record['path'])
    if not path.is_file() or sha256_file(path)!=record['sha256']:
        raise ValueError('Shaping geometry evidence changed or is missing')
    return path


def _base(base,source_sha256,scope,design_id):
    from photo_appearance import validate_asset
    receipt=validate_asset(base,source_sha256=source_sha256,scope=scope,design_id=design_id)
    return {'asset_id':base['asset_id'],'receipt_sha256':base['receipt_sha256'],
        'source_sha256':source_sha256,'scope_id':scope['scope_id'],'design_id':design_id,
        'master_sha256':receipt['files']['master']['sha256'],
        'preview_sha256':receipt['files']['preview']['sha256'],'native_size':receipt['canvas_size']},receipt


def _runtime(root,registry_path):
    from current_model_registry import load_registry
    registry=load_registry(root,registry_path=registry_path,
        required_keys={'face_python','face_model_path'},required_profiles=())
    python=Path(registry['face_python']).resolve();model=Path(registry['face_model_path']).resolve()
    for path in (python,model):
        if not path.is_file() or not 0<path.stat().st_size<=16*1024*1024:
            raise ValueError('Shaping localization requires the registered bounded CPU Python and model')
    sites=python.parent.parent/'Lib/site-packages'
    matches=sorted(sites.glob('mediapipe-*.dist-info/METADATA'))
    if len(matches)!=1 or matches[0].stat().st_size>16*1024*1024:
        raise ValueError('Missing or ambiguous registered MediaPipe metadata')
    metadata=matches[0]
    lines=metadata.read_text(encoding='utf8').splitlines()
    versions=[line[9:].strip() for line in lines if line.startswith('Version: ')]
    names=[line[6:].strip().lower() for line in lines if line.startswith('Name: ')]
    if names!=['mediapipe'] or len(versions)!=1 or not versions[0]:raise ValueError('Invalid MediaPipe package metadata')
    # Runtime/model paths are provenance, deliberately not session file records.
    return {'python_path':str(python),'python_sha256':sha256_file(python),
        'model_path':str(model),'model_sha256':sha256_file(model),
        'mediapipe_metadata_path':str(metadata),'mediapipe_metadata_sha256':sha256_file(metadata),
        'mediapipe_version':versions[0]}


def _preflight(runtime,preflight):
    if preflight is None:return
    if not isinstance(preflight,dict):raise ValueError('Checked face localization preflight required')
    checks={row.get('key'):row for row in preflight.get('checks',[]) if isinstance(row,dict)}
    expected={'face_model_path.observed_identity':('model_path','model_sha256','actual_sha256'),
        'face_python.observed_identity':('python_path','python_sha256','actual_sha256'),
        'face_runtime.mediapipe':('mediapipe_metadata_path','mediapipe_metadata_sha256','metadata_sha256')}
    for key,(path,digest,field) in expected.items():
        row=checks.get(key) or {}
        if (row.get('available') is not True or str(Path(row.get('path') or '').resolve())!=runtime[path]
                or row.get(field)!=runtime[digest]):
            raise ValueError('Face localization runtime differs from its checked preflight: '+key)
    if checks['face_runtime.mediapipe'].get('version')!=runtime['mediapipe_version']:
        raise ValueError('Checked MediaPipe version changed')


def _producer():
    return {name:sha256_file(code_path(name)) for name in
        ('photo_shaping_geometry.py','portrait_regions_worker.py','runtime_resources.py','asset_identity.py')}


def _run_detector(runtime,preview,output,cancel):
    _cancel(cancel)
    command=[runtime['python_path'],'-B','-X','utf8',str(code_path('portrait_regions_worker.py')),
        '--detect-only','--input',str(preview),'--landmarker-model',runtime['model_path'],
        '--max-faces',str(PARAMETERS['maximum_faces']),'--output-dir',str(output)]
    with (output/'detector.log').open('wb') as log:
        process=external_python_popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=isolated_python_environment(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic()
        try:
            while process.poll() is None:
                _cancel(cancel)
                if time.monotonic()-started>PARAMETERS['maximum_worker_seconds']:
                    raise TimeoutError('CPU shaping localization exceeded 180 seconds')
                try:process.wait(timeout=.1)
                except subprocess.TimeoutExpired:pass
            if process.returncode:raise RuntimeError('CPU shaping localization failed; inspect detector.log')
            _cancel(cancel)
        except BaseException:
            # This exact child handle is ours. No unrelated service is stopped.
            if process.poll() is None:process.kill()
            process.wait(timeout=10)
            raise


def _smooth(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)


def _polygon(points,size):
    image=Image.new('L',tuple(size));ImageDraw.Draw(image).polygon([tuple(p) for p in points],fill=1)
    return np.asarray(image,dtype=bool)


def _simple_polygon(points):
    """Reject crossing, touching non-neighbours and overlapping native edges."""
    points=np.asarray(points,np.float64);count=len(points)
    scale=max(1.,float(np.ptp(points,axis=0).max()));epsilon=scale*1e-7;area_epsilon=epsilon*scale
    def cross(a,b,c):
        x=b-a;y=c-a
        return float(x[0]*y[1]-x[1]*y[0])
    def on_segment(a,b,p):
        return (abs(cross(a,b,p))<=area_epsilon
            and np.all(p>=np.minimum(a,b)-epsilon) and np.all(p<=np.maximum(a,b)+epsilon))
    for index,a in enumerate(points):
        if any(np.linalg.norm(a-b)<=epsilon for b in points[index+1:]):
            raise ValueError('Canonical face polygon has duplicate or collapsed vertices')
        previous=points[index-1];following=points[(index+1)%count]
        if abs(cross(previous,a,following))<=area_epsilon and np.dot(previous-a,following-a)>0:
            raise ValueError('Canonical face polygon has overlapping adjacent edges')
        for other in range(index+1,count):
            if other==index+1 or (index==0 and other==count-1):continue
            b=points[(index+1)%count];c=points[other];d=points[(other+1)%count]
            first,second,third,fourth=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
            proper=((first>area_epsilon and second<-area_epsilon or first<-area_epsilon and second>area_epsilon)
                and (third>area_epsilon and fourth<-area_epsilon or third<-area_epsilon and fourth>area_epsilon))
            if proper or on_segment(a,b,c) or on_segment(a,b,d) or on_segment(c,d,a) or on_segment(c,d,b):
                raise ValueError('Canonical face polygon has crossing or touching non-adjacent edges')


def _chain(points,xx,yy):
    distances=np.full(xx.shape,np.inf,np.float32);positions=np.zeros(xx.shape,np.float32)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);total=float(lengths.sum());offset=0.
    if not np.isfinite(total) or total<=1:raise ValueError('Degenerate canonical contour chain')
    for a,b,length in zip(points,points[1:],lengths):
        if length<=1e-6:continue
        delta=b-a;fraction=np.clip(((xx-a[0])*delta[0]+(yy-a[1])*delta[1])/(length*length),0,1)
        distance=np.hypot(xx-a[0]-fraction*delta[0],yy-a[1]-fraction*delta[1])
        nearer=distance<distances;distances[nearer]=distance[nearer]
        positions[nearer]=((offset+fraction*length)/total)[nearer];offset+=float(length)
    return distances,positions


def _clip_chain(points,box):
    """Clip actual line segments to the core without clamping outside vertices."""
    points=np.asarray(points,np.float64);origin=np.asarray(box[:2],np.float64)
    extent=np.asarray(box[2:],np.float64)-origin;segments=[];current=None
    for a,b in zip(points,points[1:]):
        delta=b-a;low=0.;high=1.
        for axis in (0,1):
            if abs(delta[axis])<1e-12:
                if not box[axis]<=a[axis]<=box[axis+2]:high=-1.;break
            else:
                first=(box[axis]-a[axis])/delta[axis];last=(box[axis+2]-a[axis])/delta[axis]
                low=max(low,min(first,last));high=min(high,max(first,last))
        if high<=low or np.linalg.norm(delta*(high-low))<1e-8:
            current=None;continue
        start=(a+delta*low-origin)/extent;end=(a+delta*high-origin)/extent
        # Analytic intersections can differ from their exact border by one
        # floating point epsilon; only round that numerical residue.
        for point in (start,end):
            point[np.abs(point)<1e-12]=0.;point[np.abs(point-1.)<1e-12]=1.
        if current is None or not np.allclose(current[-1],start,rtol=0,atol=1e-8):
            current=[start.tolist()];segments.append(current)
        current.append(end.tolist())
    return segments


def _geometry(detections,size,*,cancel=None):
    from portrait_regions_worker import FACE_OVAL, FEATURE_POLYGONS
    if not isinstance(detections.get('faces'),list) or len(detections['faces'])!=1:
        raise ValueError('Shaping requires exactly one unambiguous localized face')
    points=np.asarray(detections['faces'][0].get('landmarks'),np.float32);width,height=map(int,size)
    if (points.ndim!=2 or points.shape[1]!=2 or not 468<=len(points)<=512 or not np.isfinite(points).all()
            or np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height)):
        raise ValueError('Canonical native face landmarks are incomplete, nonfinite or out of canvas')
    oval=points[list(FACE_OVAL)];_simple_polygon(oval)
    eye_a=(points[33]+points[133])*.5;eye_b=(points[362]+points[263])*.5
    horizontal=eye_b-eye_a;eye_span=float(np.linalg.norm(horizontal))
    if eye_span<=1:raise ValueError('Degenerate canonical eye axis')
    horizontal/=eye_span;vertical=np.array([-horizontal[1],horizontal[0]],np.float32)
    if (points[152]-(eye_a+eye_b)*.5)@vertical<0:vertical*=-1
    face_width=float(np.ptp(oval@horizontal));face_height=float(np.ptp(oval@vertical))
    area=abs(float(np.sum(oval[:,0]*np.roll(oval[:,1],-1)-oval[:,1]*np.roll(oval[:,0],-1))))*.5
    if (min(face_width,face_height)<PARAMETERS['minimum_face_span'] or area<face_width*face_height*.2
            or not .4<=face_width/face_height<=2.2 or not .15<=eye_span/face_width<=.85):
        raise ValueError('Face is too small, distorted or geometrically unreliable')
    inner=face_width*PARAMETERS['inner_width_face_fraction'];outer=face_width*PARAMETERS['outer_width_face_fraction']
    margin=face_width*PARAMETERS['feature_margin_face_fraction']
    envelope_padding=face_width*PARAMETERS['face_envelope_padding_fraction']
    ramp=face_width*PARAMETERS['outside_envelope_ramp_face_fraction']
    padding=int(np.ceil(max(inner,outer,margin,envelope_padding+ramp)))+2
    left,top=np.floor(oval.min(axis=0)-padding).astype(int);right,bottom=np.ceil(oval.max(axis=0)+padding+1).astype(int)
    if not (0<=left<right<=width and 0<=top<bottom<=height):
        raise ValueError('The padded face envelope is clipped by the native canvas')
    local=points-[left,top];roi_size=(right-left,bottom-top);inside=_polygon(local[list(FACE_OVAL)],roi_size)
    start,stop=FACE_OVAL.index(454),FACE_OVAL.index(234)
    lower_indices=FACE_OVAL[start:stop+1];upper_indices=FACE_OVAL[stop:]+FACE_OVAL[:start+1]
    envelope=inside|(distance_transform_edt(~inside)<=envelope_padding)
    features=np.zeros(inside.shape,bool);feature_hulls=[]
    for indices in (*FEATURE_POLYGONS,NOSE):
        group=local[list(indices)]
        if float(np.ptp(group,axis=0).max())<max(1.,face_width*PARAMETERS['minimum_feature_span_face_fraction']):
            raise ValueError('Canonical facial feature point group is collapsed or too small')
        try:
            hull=ConvexHull(group)
        except QhullError as error:raise ValueError('Degenerate canonical facial feature point group') from error
        if hull.volume<=np.finfo(np.float32).eps*face_width*face_width:
            raise ValueError('Canonical facial feature point group is effectively collinear')
        boundary=group[hull.vertices];feature_hulls.append(points[list(indices)][hull.vertices])
        # These are canonical point groups, not ordered feature outlines.
        # The hull protects their whole envelope; the old raster union also
        # preserves discrete edge pixels regardless of vertex ordering.
        features|=_polygon(boundary,roi_size)|_polygon(group,roi_size)
    features=distance_transform_edt(~features)<=margin
    if np.any(features&~envelope):
        raise ValueError('Expanded canonical facial feature protection leaves the safe face envelope')
    central=inside&(distance_transform_edt(inside)>=inner);protected=central|features
    weight=np.zeros(inside.shape,np.float32)
    for row in range(0,roi_size[1],96):
        _cancel(cancel)
        end=min(row+96,roi_size[1]);yy,xx=np.indices((end-row,roi_size[0]),dtype=np.float32);yy+=row
        distance,position=_chain(local[list(lower_indices)],xx,yy)
        upper,_=_chain(local[list(upper_indices)],xx,yy)
        band_width=np.where(inside[row:end],inner,outer)
        spatial=_smooth((band_width-distance)/(band_width*PARAMETERS['edge_feather_fraction_of_band']))
        taper=_smooth(np.minimum(position,1-position)/PARAMETERS['arc_endpoint_taper_fraction'])
        spatial[(distance>=upper)|protected[row:end]|~envelope[row:end]]=0
        weight[row:end]=spatial*taper
    if not np.any(weight>0) or np.any(weight[protected]!=0) or np.any((weight>0)&~envelope):
        raise ValueError('No safe nonempty canonical cheek/jaw support remains')
    outside=_smooth(distance_transform_edt(~envelope)/ramp).astype(np.float32);outside[envelope]=0
    geometry={'box':[int(left),int(top),int(right),int(bottom)],'face_width':face_width,'face_height':face_height,
        'eye_span':eye_span,'inner_width':inner,'outer_width':outer,'feature_margin':margin,
        'envelope_padding':envelope_padding,'outside_ramp':ramp,
        'normalized_cheek_jaw_contour':(points[list(lower_indices)]/np.array([width,height])).tolist(),
        'normalized_protected_feature_polygons':[(boundary/np.array([width,height])).tolist() for boundary in feature_hulls],
        'feature_protection':PARAMETERS['feature_protection'],
        'canonical_lower_indices':list(lower_indices),'semantic_truth_claimed':False,'naturalness_verified':False}
    return {'J':weight,'P':protected,'E':envelope,'O':outside},geometry


def _validate_detection(detections,binding):
    if (detections.get('source_sha256')!=binding['base']['preview_sha256']
            or detections.get('source_native_size')!=binding['base']['native_size']
            or detections.get('model_sha256')!=binding['runtime']['model_sha256']
            or detections.get('mediapipe_version')!=binding['runtime']['mediapipe_version']
            or detections.get('method')!='mediapipe_offline_multi_face_landmarks'):
        raise ValueError('Face localization does not describe the actual checked U/runtime')


def _support(directory,binding=None):
    path=Path(directory)/'support.json';document=json.loads(path.read_text(encoding='utf8'))
    if (document.get('schema')!=SCHEMA or (binding is not None and document.get('binding')!=binding)
            or document.get('support_id')!=identity({key:value for key,value in document.items() if key!='support_id'})):
        raise ValueError('Shaping geometry support identity changed')
    if (document['binding'].get('producer')!=_producer() or document['binding'].get('parameters')!=PARAMETERS):
        raise ValueError('Shaping geometry support producer or algorithm changed')
    detections=_read(document['detections']);mask=_read(document['masks'])
    observed=json.loads(detections.read_text(encoding='utf8'))
    _validate_detection(observed,document['binding'])
    expected,geometry=_geometry(observed,document['binding']['base']['native_size'])
    if geometry!=document.get('geometry'):
        raise ValueError('Saved shaping geometry differs from its actual canonical detections')
    with np.load(mask,allow_pickle=False) as arrays:
        box=document['geometry']['box'];shape=(box[3]-box[1],box[2]-box[0])
        if set(arrays.files)!={'J','P','E','O'}:raise ValueError('Incomplete shaping geometry planes')
        j,p,e,o=(arrays[key] for key in ('J','P','E','O'))
        if (any(value.shape!=shape for value in (j,p,e,o)) or j.dtype!=np.float32 or o.dtype!=np.float32
                or p.dtype!=bool or e.dtype!=bool or not np.isfinite(j).all() or not np.isfinite(o).all()
                or np.any((j<0)|(j>1)) or np.any((o<0)|(o>1)) or not np.any(j>0)
                or np.any(j[p]!=0) or np.any((j>0)&~e) or np.any(o[e]!=0) or np.any(p&~e)):
            raise ValueError('Invalid canonical shaping protection planes')
        if any(not np.array_equal(arrays[key],value) for key,value in expected.items()):
            raise ValueError('Saved shaping protection differs from deterministic canonical geometry')
    return document


def _copy_support(source,output,binding,cancel,verify):
    document=_support(source,binding);output=Path(output).resolve()
    if output.exists():
        _support(output,binding);_cancel(cancel);return _record(output/'support.json')
    output.parent.mkdir(parents=True,exist_ok=True);staging=output.with_name('.'+output.name+'.partial-'+uuid4().hex)
    staging.mkdir()
    try:
        for key in ('detections','masks'):
            original=_read(document[key]);target=staging/original.name;shutil.copyfile(original,target)
            if sha256_file(target)!=document[key]['sha256']:raise ValueError('Geometry evidence changed during owning')
            document[key]['path']=str(output/original.name)
        document['support_id']=identity({key:value for key,value in document.items() if key!='support_id'})
        _json(staging/'support.json',document);_cancel(cancel);verify();os.rename(staging,output)
        _support(output,binding);return _record(output/'support.json')
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def prepare_geometry(root,base_appearance,*,source_sha256,scope,design_id,output,registry_path=None,cancel=None,preflight=None):
    """Own verified canonical support; stale or partial support is never reused."""
    from cache_store import exclusive_file_lock
    root=Path(root).resolve();_cancel(cancel)
    base,receipt=_base(base_appearance,source_sha256,scope,design_id)
    runtime=_runtime(root,registry_path);_preflight(runtime,preflight)
    binding={'base':base,'runtime':runtime,'producer':_producer(),'parameters':copy.deepcopy(PARAMETERS)}
    cache=root/'outputs/cache/shaping-geometry'/identity(binding)
    preview=Path(base_appearance['path'])/receipt['files']['preview']['file']
    cache.parent.mkdir(parents=True,exist_ok=True)
    def current():
        _cancel(cancel);now,_=_base(base_appearance,source_sha256,scope,design_id)
        actual=_runtime(root,registry_path);_preflight(actual,preflight)
        if now!=base or actual!=runtime or _producer()!=binding['producer']:
            raise ValueError('Shaping geometry input, producer or runtime changed during execution')
    with exclusive_file_lock(cache.with_suffix('.lock'),timeout=190,cancel=cancel):
        current()
        if not cache.exists():
            stage=cache.with_name('.'+cache.name+'.partial-'+uuid4().hex);stage.mkdir()
            try:
                _run_detector(runtime,preview,stage,cancel);current()
                detections=json.loads((stage/'detections.json').read_text(encoding='utf8'))
                _validate_detection(detections,binding);masks,geometry=_geometry(detections,base['native_size'],cancel=cancel)
                np.savez_compressed(stage/'geometry.npz',**masks)
                document={'schema':SCHEMA,'binding':binding,'geometry':geometry,
                    'detections':{'path':str(cache/'detections.json'),'sha256':sha256_file(stage/'detections.json')},
                    'masks':{'path':str(cache/'geometry.npz'),'sha256':sha256_file(stage/'geometry.npz')}}
                document['support_id']=identity(document);_json(stage/'support.json',document);current()
                os.rename(stage,cache)
            except BaseException:
                shutil.rmtree(stage,ignore_errors=True);raise
        result=_copy_support(cache,Path(output),binding,cancel,current);current();return result


def region_weight(support_descriptor,*,base_appearance,source_sha256,scope,design_id,core_box,selected_parts):
    """Return actual-core coverage; body union cannot reopen protected face."""
    from photo_shaping import normalize_request
    parts=normalize_request({'action':'apply','parts':selected_parts})['parts']
    path=_read(support_descriptor);document=_support(path.parent)
    base,_=_base(base_appearance,source_sha256,scope,design_id)
    if document['binding']['base']!=base or document['binding']['producer']!=_producer() or document['binding']['parameters']!=PARAMETERS:
        raise ValueError('Canonical shaping support is stale or belongs to different U')
    runtime=document['binding']['runtime']
    for path_key,sha_key in (('python_path','python_sha256'),('model_path','model_sha256'),
                            ('mediapipe_metadata_path','mediapipe_metadata_sha256')):
        if not Path(runtime[path_key]).is_file() or sha256_file(runtime[path_key])!=runtime[sha_key]:
            raise ValueError('Canonical shaping localization runtime changed after preparation')
    width,height=base['native_size']
    if (not isinstance(core_box,(list,tuple)) or len(core_box)!=4 or any(type(v) is not int for v in core_box)
            or not 0<=core_box[0]<core_box[2]<=width or not 0<=core_box[1]<core_box[3]<=height):
        raise ValueError('Shaping core must be an integer native canvas box')
    left,top,right,bottom=core_box;shape=(bottom-top,right-left)
    j=np.zeros(shape,np.float32);p=np.zeros(shape,bool);e=np.zeros(shape,bool);outside=np.ones(shape,np.float32)
    box=document['geometry']['box'];x0=max(left,box[0]);y0=max(top,box[1]);x1=min(right,box[2]);y1=min(bottom,box[3])
    if x0<x1 and y0<y1:
        target=(slice(y0-top,y1-top),slice(x0-left,x1-left));source=(slice(y0-box[1],y1-box[1]),slice(x0-box[0],x1-box[0]))
        with np.load(_read(document['masks']),allow_pickle=False) as planes:
            for output,key in ((j,'J'),(p,'P'),(e,'E'),(outside,'O')):output[target]=planes[key][source]
    face='face' in parts;body=bool(set(parts)-{'face'})
    if face and not np.any(j>0):raise ValueError('Selected native core contains no safe canonical cheek/jaw band')
    weight=np.maximum(j if face else 0,outside if body else 0).astype(np.float32)
    weight[p]=0
    if (not np.any(weight>0) or np.any(weight[p]!=0) or np.any(weight[e&(j==0)]!=0)
            or (not body and np.any(weight[j==0]!=0))):
        raise ValueError('Shaping coverage violates the protected canonical face envelope')
    key=identity({'support_sha256':support_descriptor['sha256'],'core_box':list(core_box),'parts':parts})
    folder=path.parent/'regions';folder.mkdir(exist_ok=True);target=folder/(key+'.npz')
    if not target.exists():
        stage=folder/('.'+key+'.partial-'+uuid4().hex+'.npz')
        try:
            np.savez_compressed(stage,J=j,P=p,E=e,weight=weight);os.replace(stage,target)
        finally:stage.unlink(missing_ok=True)
    else:
        with np.load(target,allow_pickle=False) as saved:
            if set(saved.files)!={'J','P','E','weight'} or any(not np.array_equal(saved[k],v) for k,v in (('J',j),('P',p),('E',e),('weight',weight))):
                raise ValueError('Previously frozen shaping core coverage changed')
    evidence={'schema':REGION_SCHEMA,'support':copy.deepcopy(support_descriptor),'base':base,'core_box':list(core_box),
        'selected_parts':parts,'core_masks':_record(target),'geometry':copy.deepcopy(document['geometry']),
        'coverage_policy':'(1-P)*max(selected_face_J,selected_body_O); O=0 throughout E',
        'non_band_face_coverage_exact_zero':True,'protected_pixels':int(p.sum()),
        'positive_weight_pixels':int(np.count_nonzero(weight)),'semantic_truth_claimed':False,'naturalness_verified':False,
        'lower_contour_in_core':((np.asarray(document['geometry']['normalized_cheek_jaw_contour'])*[width,height]-[left,top])/[right-left,bottom-top]).tolist(),
        'lower_contour_segments_in_core':_clip_chain(
            np.asarray(document['geometry']['normalized_cheek_jaw_contour'])*[width,height],core_box),
        'band_width_in_core':{key:[document['geometry'][field]/(right-left),document['geometry'][field]/(bottom-top)]
            for key,field in (('inner','inner_width'),('outer','outer_width'))},
        'hard_protection':'central_face_and_features_exact_zero','geometry_only':True}
    return weight,evidence
