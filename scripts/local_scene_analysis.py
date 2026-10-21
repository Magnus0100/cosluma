"""Scene assets in explicit source/candidate spaces; GPU packages stay in the worker."""
from __future__ import annotations

from runtime_resources import external_python_popen, external_python_run

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

import numpy as np
from PIL import Image

from raw_support import open_rgb, RGB_DECODE_VERSION
from local_photo_analysis import file_identity
from runtime_resources import code_path
from asset_identity import sha256_file
from cache_roots import cache_dir
from cache_store import exclusive_file_lock
from runtime_resources import isolated_python_environment


RECIPE = 'learned-scene-local-mirror-registration-contact-v3'


# SAM3.1 is intentionally kept in its isolated runtime.  A short-lived
# worker pays the 3.5 GB checkpoint load and Python import cost for every
# photo; the service below keeps that process warm while moving its detector
# back to CPU between requests.  Thus subsequent photos reuse imports and
# mapped weights without reserving GPU memory while BeautyGRPO is active.
_SCENE_SERVICE = None
_SCENE_SERVICE_LOCK = threading.Lock()


class _SceneServiceUnavailable(RuntimeError):
    """The warm worker could not be started; caller may use the one-shot path."""


def _scene_service_key(paths):
    return tuple(str(Path(paths[key]).resolve()) for key in ('python', 'worker', 'models', 'profile'))


def _stop_scene_service():
    global _SCENE_SERVICE
    service = _SCENE_SERVICE
    _SCENE_SERVICE = None
    if not service:
        return
    process = service.get('process')
    try:
        if process and process.poll() is None:
            process.stdin.close()
            process.wait(timeout=8)
    except Exception:
        try:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            try:
                if process and process.poll() is None:
                    process.kill()
            except Exception:
                pass
    try:
        service.get('log_stream').close()
    except Exception:
        pass


atexit.register(_stop_scene_service)


def _scene_worker_environment():
    return isolated_python_environment({
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})


def _read_scene_ready(process, timeout_seconds):
    """Bound a blocking Windows pipe read without retaining the caller thread."""
    complete = threading.Event()
    response = []
    def read():
        try:
            response.append(process.stdout.readline())
        except Exception as error:
            response.append(error)
        finally:
            complete.set()
    threading.Thread(target=read, name='sam31-startup-receipt', daemon=True).start()
    if not complete.wait(timeout_seconds):
        raise TimeoutError(f'SAM3.1 ready receipt timed out after {timeout_seconds:g} seconds')
    if isinstance(response[0], Exception):
        raise response[0]
    return response[0]


def _start_scene_service(root, paths, *, timeout_seconds=180):
    global _SCENE_SERVICE
    key = _scene_service_key(paths)
    if _SCENE_SERVICE and _SCENE_SERVICE.get('key') == key:
        process = _SCENE_SERVICE.get('process')
        if process and process.poll() is None:
            return _SCENE_SERVICE
        _stop_scene_service()
    if _SCENE_SERVICE:
        _stop_scene_service()
    service_dir = cache_dir(root, 'local-scene') / '.service'
    service_dir.mkdir(parents=True, exist_ok=True)
    log_path = service_dir / 'sam3-scene-service.log'
    stream = log_path.open('a', encoding='utf8')
    command = [str(paths['python']), '-X', 'utf8', str(paths['worker']), '--service',
               '--model-root', str(paths['models']), '--device', 'cuda']
    # Scene inference runs in its pinned SAM3.1 environment.  Do not inherit
    # the desktop/agent PYTHONPATH: it can expose a different numpy dist-info
    # before the isolated runtime and make the strict package identity check
    # fail even when the pinned interpreter and wheels are correct.
    worker_env = _scene_worker_environment()
    process = None
    try:
        process = external_python_popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stream, text=True, bufsize=1,
            start_new_session=os.name != 'nt',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env=worker_env)
        ready_line = _read_scene_ready(process, timeout_seconds)
        ready = json.loads(ready_line) if ready_line else None
        if not ready or ready.get('ready') is not True:
            raise RuntimeError('SAM3.1 service did not return its ready receipt')
    except Exception as exc:
        try:
            if process is not None and process.poll() is None:
                # A Windows venv executable can launch the actual model
                # process. Stop its owned tree before falling back to a new
                # worker, so a startup timeout cannot leave two GPU owners.
                from beautygrpo_prepare import _stop_process
                _stop_process(process)
        except Exception:
            if process is not None:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
        stream.close()
        raise _SceneServiceUnavailable(f'Could not start SAM3.1 warm worker: {exc}') from exc
    _SCENE_SERVICE = {'key': key, 'process': process, 'log_stream': stream,
                      'log_path': str(log_path.resolve()), 'ready': ready,
                      'model_root': str(Path(paths['models']).resolve())}
    return _SCENE_SERVICE


def _run_scene_worker_once(argv, log):
    with log.open('w', encoding='utf-8') as stream:
        process = external_python_run(argv, stdout=stream, stderr=stream, timeout=900,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env=_scene_worker_environment())
    if process.returncode:
        raise RuntimeError(f'Local scene worker failed; see {log}')


def _scene_service_request(root, paths, request):
    """Submit one scene pair to the warm worker and wait for its JSON receipt."""
    with _SCENE_SERVICE_LOCK:
        service = _start_scene_service(root, paths)
        process = service['process']
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('SAM3.1 warm worker exited without a response')
            receipt = json.loads(line)
        except Exception:
            _stop_scene_service()
            raise
        if receipt.get('ok') is not True:
            raise RuntimeError('SAM3.1 warm worker failed: '+str(receipt.get('error') or 'unknown error'))
        return receipt


def digest(path):
    return sha256_file(path)


def scene_runtime_paths(root, *, registry_path=None):
    from current_model_registry import load_registry
    config = load_registry(root, registry_path=registry_path)
    from sam31_model_profile import specification
    specification(config['scene_profile'])
    return {'python': Path(config['scene_python']), 'worker': code_path('sam3_scene_worker.py'),
            'models': Path(config['scene_model_root']), 'profile': config['scene_profile'],
            'registry': config['registry_path'], 'ready':config['ready']}


def local_scene_ready(root, *, registry_path=None):
    paths = scene_runtime_paths(root, registry_path=registry_path)
    from sam31_model_profile import model_identity
    if not paths['ready'] or not paths['python'].is_file() or not paths['worker'].is_file():
        return False
    try:
        model_identity(paths['models'])
    except (ValueError, OSError, KeyError):
        return False
    return True


def scene_model_identity(model_root, *, profile='sam31_static_detector_v1'):
    """Bind the current static instance masks to actual weights/code/tokenizer."""
    from sam31_model_profile import specification, model_identity
    specification(profile)
    return model_identity(Path(model_root))


def prepare_local_scene(root, source, candidate, *, source_hash=None, candidate_hash=None,
                        output_dir=None, max_edge=1536, source_subject_mask=None, candidate_subject_mask=None,
                        registry_path=None, decoded_images=None):
    """Return learned mirror/contact assets, with verified geometry where available.

    Callers can consume source contact directly. Mirror replacement is supported
    only for a verified pair, and its whole-region mask must never be presented
    as an exact human contamination segmentation.
    """
    started=time.perf_counter(); root=Path(root).resolve()
    source,candidate=Path(source).resolve(),Path(candidate).resolve()
    paths=scene_runtime_paths(root, registry_path=registry_path)
    if not local_scene_ready(root, registry_path=registry_path):
        raise RuntimeError('Selected local scene model/runtime is unavailable: '+paths['profile'])
    source_hash,candidate_hash=source_hash or digest(source),candidate_hash or digest(candidate)
    subject_inputs={role:(Path(supplied).resolve() if supplied else None)
        for role,supplied,content_hash in [('source',source_subject_mask,source_hash),('candidate',candidate_subject_mask,candidate_hash)]}
    identity={'recipe':RECIPE,'decode_version':RGB_DECODE_VERSION,'max_edge':max_edge,
        'worker_sha256':digest(paths['worker']),'helper_sha256':digest(__file__),
        'model_files':scene_model_identity(paths['models'],profile=paths['profile']),
        'subject_exclusion_inputs':{role:{'path':str(path),'sha256':digest(path)} if path else None for role,path in subject_inputs.items()}}
    identity.update(recipe=RECIPE+'-sam31-static-v1',scene_profile=paths['profile'],
        registry_sha256=digest(paths['registry']),
        geometry_consumer_sha256=digest(code_path('scene_geometry.py')))
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    cache_base = cache_dir(root, 'local-scene')
    directory=Path(output_dir) if output_dir else cache_base/f'{source_hash[:16]}-{candidate_hash[:16]}'/identity_hash
    directory.mkdir(parents=True,exist_ok=True)
    with exclusive_file_lock(directory.with_name(directory.name+'.single-flight.lock')):
        result_path=directory/'analysis.json'
        if result_path.exists():
            saved=json.loads(result_path.read_text(encoding='utf-8'))
            if saved['source']['sha256']==source_hash and saved['candidate']['sha256']==candidate_hash and saved['identity']==identity:
                return {**saved,'cache_hit':True,'cache_root':str(cache_base.resolve()),
                        'elapsed_seconds':round(time.perf_counter()-started,3)}
        originals={}; proxies={}
        for role,path in [('source',source),('candidate',candidate)]:
            shared = decoded_images.pop(role, None) if decoded_images is not None else None
            if shared is not None:
                image = shared['image']
                expected_hash = source_hash if role == 'source' else candidate_hash
                if (Path(shared['path']).resolve() != path or shared['sha256'] != expected_hash
                        or not isinstance(image, Image.Image) or image.mode != 'RGB'):
                    if isinstance(image, Image.Image):
                        image.close()
                    raise ValueError('Shared scene image does not bind to the current '+role)
            else:
                image = open_rgb(path)
            try:
                originals[role]=list(image.size)
                image.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
                target=directory/f'{role}-scene-proxy.png'; image.save(target)
                proxy_size = list(image.size)
            finally:
                image.close()
            proxies[role]={'path':str(target.resolve()),'size':proxy_size,'sha256':digest(target)}
            subject=subject_inputs[role]
            if subject:
                if subject.suffix=='.npz':
                    with np.load(subject) as resource:
                        values=resource['labels']>0 if 'labels' in resource else resource['values']>.02
                    if values.ndim!=2:
                        raise ValueError('Subject exclusion mask must be a single-channel spatial mask')
                    mask=Image.fromarray(np.uint8(values)*255)
                else:
                    mask=Image.open(subject).convert('L')
                mask=mask.resize(proxy_size,Image.Resampling.NEAREST)
                mask_path=directory/f'{role}-subject-exclusion.png';mask.save(mask_path)
                proxies[role]['subject_exclusion_path']=str(mask_path.resolve())
        log=directory/'scene-worker.log'
        request = {'request_id': identity_hash,
            'source': proxies['source']['path'], 'candidate': proxies['candidate']['path'],
            'output_dir': str(directory.resolve()), 'model_root': str(paths['models'].resolve()),
            'source_subject_mask': proxies['source'].get('subject_exclusion_path'),
            'candidate_subject_mask': proxies['candidate'].get('subject_exclusion_path')}
        try:
            # Keep the isolated SAM3 process warm and return its model to CPU
            # between requests.  If it cannot be started, retain the proven
            # one-shot worker path below.
            _scene_service_request(root, paths, request)
            log.write_text('SAM3.1 persistent worker completed request '+identity_hash+'\n', encoding='utf8')
        except _SceneServiceUnavailable:
            argv=[str(paths['python']),'-X','utf8',str(paths['worker']),
                  '--source',proxies['source']['path'],'--candidate',proxies['candidate']['path'],
                  '--output-dir',str(directory.resolve()),'--model-root',str(paths['models'])]
            for role in ('source','candidate'):
                if proxies[role].get('subject_exclusion_path'):
                    argv += [f'--{role}-subject-mask',proxies[role]['subject_exclusion_path']]
            _run_scene_worker_once(argv, log)
        scene=json.loads((directory/'scene.json').read_text(encoding='utf-8'))
        for match in scene['reflection_matches']:
            if not match['supported']:
                continue
            sp,cp=proxies['source']['size'],proxies['candidate']['size']
            so,co=originals['source'],originals['candidate']
            original_mapping=np.diag([co[0]/cp[0],co[1]/cp[1],1]) @ np.asarray(match['source_to_candidate_proxy']) @ np.diag([sp[0]/so[0],sp[1]/so[1],1])
            match['source_original_to_candidate_original']=original_mapping.tolist()
            match['candidate_original_to_source_original']=np.linalg.inv(original_mapping).tolist()
            match['source_original_size']=so; match['candidate_original_size']=co
            for key in ('source_reflection_region','trusted_source_reflection'):
                match[key]['coordinate_space']='source_scene_proxy'
                match[key]['original_canvas_size']=so
            for key in ('candidate_replacement_region','reflection_contamination_partial'):
                match[key]['coordinate_space']='candidate_scene_proxy'
                match[key]['original_canvas_size']=co
        for role,contact in scene['contact'].items():
            for key in ('contact_support','support_surface','footwear'):
                contact[key]['coordinate_space']=f'{role}_scene_proxy'
                contact[key]['original_canvas_size']=originals[role]
            px,py=proxies[role]['size'];ox,oy=originals[role]
            contact['original_points']=[{**point,'point':[point['point'][0]*ox/px,point['point'][1]*oy/py],
                                        'coordinate_space':f'{role}_original_oriented'} for point in contact['points']]
        supported=[match for match in scene['reflection_matches'] if match['supported']]
        scene['recommended_reflection_match']=max(supported,key=lambda match:match['inliers']*match['mirror_mask_iou']) if supported else None
        background=scene['background_mapping']
        sp,cp=proxies['source']['size'],proxies['candidate']['size'];so,co=originals['source'],originals['candidate']
        candidate_to_source_original=np.diag([so[0]/sp[0],so[1]/sp[1],1]) @ np.asarray(background['candidate_to_source_proxy']) @ np.diag([cp[0]/co[0],cp[1]/co[1],1])
        background['candidate_original_to_source_original']=candidate_to_source_original.tolist()
        background['source_original_to_candidate_original']=np.linalg.inv(candidate_to_source_original).tolist()
        background['source_original_size']=so;background['candidate_original_size']=co
        scene['mapping_recommendation']={'method':background['method'],
            'scope':'whole_background' if background['supported'] else 'center_cover_framing',
            'global_background_mapping_established':background['supported'],
            'unsupported_mirrors_action':'retain_source_reflection_or_leave_unchanged_without_forced_copy'}
        scene.update({'ok':True,'source':{'path':str(source),'sha256':source_hash,'oriented_size':originals['source']},
            'candidate':{'path':str(candidate),'sha256':candidate_hash,'oriented_size':originals['candidate']},
            'proxies':proxies,'identity':identity,'cache_root':str(cache_base.resolve()),
            'analysis_directory':str(directory.resolve()),
            'cache_hit':False,'elapsed_seconds':round(time.perf_counter()-started,3),
            'gpt_used':False,'image_generation_used':False})
        scene['provenance'].update({'source_sha256':source_hash,'candidate_sha256':candidate_hash,
            'source_decode':RGB_DECODE_VERSION,'model_content_sha256':identity['model_files']['content_sha256']})
        temporary = result_path.with_name(result_path.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        os.replace(temporary, result_path)
        return scene


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True); parser.add_argument('--candidate',required=True)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output-dir'); parser.add_argument('--max-edge',type=int,default=1536)
    parser.add_argument('--model-registry'); parser.add_argument('--source-subject-mask')
    parser.add_argument('--candidate-subject-mask')
    args=parser.parse_args()
    result=prepare_local_scene(args.root,args.source,args.candidate,output_dir=args.output_dir,max_edge=args.max_edge,
        registry_path=args.model_registry, source_subject_mask=args.source_subject_mask,
        candidate_subject_mask=args.candidate_subject_mask)
    print(json.dumps({'ok':result['ok'],'reflection_supported':result['reflection_supported'],
        'supported_matches':sum(match['supported'] for match in result['reflection_matches']),
        'source_contact_supported':result['contact']['source']['supported'],
        'analysis_directory':result['analysis_directory'],'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':
    main()
