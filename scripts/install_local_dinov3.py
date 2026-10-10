"""Install user-provided ViT-L DINOv3 files with pinned ModelScope/HF provenance.

Only public metadata and ModelScope README/license text are fetched. The provided
weight is copied, never moved; its SHA must match both published registries.
This installer performs no model inference and does not change production defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from reference_local_correspondence import (DINO3_L_PROFILE, DINO3_L_REVISION,
    algorithm_configuration, file_sha)

MODELSCOPE_REVISION = '99b6205b0321c4d9726dd072d8c6a80aa6be72df'
FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE.md')


def git_blob_sha(content):
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def verify_file(path, scope_metadata, official_metadata):
    """Verify raw SHA/size plus an official LFS SHA or Git blob identity."""
    path = Path(path)
    digest = file_sha(path)
    size = path.stat().st_size
    if size != scope_metadata['Size'] or digest != scope_metadata['Sha256']:
        raise ValueError('Local file differs from pinned ModelScope metadata: '+path.name)
    if size != official_metadata['size']:
        raise ValueError('Local file length differs from pinned official metadata: '+path.name)
    lfs = official_metadata.get('lfs')
    if lfs:
        if digest != lfs['sha256']:
            raise ValueError('Local weight differs from official LFS SHA: '+path.name)
        method = 'sha256_official_lfs_and_modelscope'
    else:
        if git_blob_sha(path.read_bytes()) != official_metadata['blobId']:
            raise ValueError('Local text differs from official Git blob: '+path.name)
        method = 'sha256_modelscope_and_official_git_blob_sha1'
    return {'sha256': digest, 'bytes': size, 'verification': method}


def install(source, *, proxy=None, evidence_directory=None):
    import requests
    from safetensors import safe_open
    source = Path(source).resolve()
    adapter = algorithm_configuration(DINO3_L_PROFILE)
    repository = adapter['repository']
    directory = adapter['model_directory'].resolve()
    evidence_directory = Path(evidence_directory or
        Path(__file__).resolve().parents[1]/'outputs/model-check/dinov3-vitl16').resolve()
    evidence_directory.mkdir(parents=True, exist_ok=True)
    if not source.is_dir() or source == directory:
        raise ValueError('A distinct user download directory is required')
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies.update(http=proxy, https=proxy)
    receipt = {'schema': 'local_correspondence_installation.current.v1',
        'algorithm_profile': DINO3_L_PROFILE, 'repository': repository, 'revision': DINO3_L_REVISION,
        'source_directory': str(source), 'installed_directory': str(directory),
        'source_originals_retained': True, 'ready': False, 'status': 'verifying_local_modelscope_files',
        'production_default_changed': False, 'models_inferred': 0, 'gpu_inference': 0,
        'weight_downloads': 0, 'credentials_consumed': False}

    def get_json(url, name, **kwargs):
        response = session.get(url, timeout=(8, 30), **kwargs)
        response.raise_for_status()
        data = response.json()
        (evidence_directory/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        return data

    try:
        official = get_json('https://huggingface.co/api/models/'+repository+'/revision/'+DINO3_L_REVISION,
            'hf-public-pinned-model-metadata.json', params={'blobs': 'true'})
        if official.get('sha') != DINO3_L_REVISION or official.get('id') != repository:
            raise ValueError('Official public metadata does not describe the pinned model')
        scope = get_json('https://modelscope.cn/api/v1/models/'+repository, 'modelscope-model-metadata.json')
        listing = get_json('https://modelscope.cn/api/v1/models/'+repository+'/repo/files',
            'modelscope-pinned-files-metadata.json', params={'Revision': MODELSCOPE_REVISION, 'Recursive': 'true'})
        if scope.get('Code') != 200 or listing.get('Code') != 200:
            raise ValueError('ModelScope metadata request was not successful')
        details = scope['Data']
        if details['Path']+'/'+details['Name'] != repository:
            raise ValueError('ModelScope publisher/model identity changed')
        scope_files = {row['Path']: row for row in listing['Data']['Files']}
        official_files = {row['rfilename']: row for row in official['siblings']}
        verified, source_hashes = {}, {}
        for name in FILES:
            original = source/name
            destination = directory/name
            if name in ('README.md', 'LICENSE.md') and not original.is_file():
                response = session.get('https://modelscope.cn/api/v1/models/'+repository+'/repo',
                    params={'Revision': MODELSCOPE_REVISION, 'FilePath': name}, timeout=(8, 30))
                response.raise_for_status()
                temporary = destination.with_suffix(destination.suffix+'.download')
                temporary.write_bytes(response.content)
                verified[name] = verify_file(temporary, scope_files[name], official_files[name])
                temporary.replace(destination)
            else:
                verified[name] = verify_file(original, scope_files[name], official_files[name])
                source_hashes[name] = verified[name]['sha256']
                if not destination.is_file() or file_sha(destination) != verified[name]['sha256']:
                    temporary = destination.with_suffix(destination.suffix+'.copy')
                    shutil.copyfile(original, temporary)
                    if file_sha(temporary) != source_hashes[name]:
                        raise ValueError('Current model copy verification failed: '+name)
                    temporary.replace(destination)
        config = json.loads((directory/'config.json').read_text(encoding='utf8'))
        if any(config.get(key) != value for key, value in adapter['expected_config'].items()):
            raise ValueError('Current model config differs from explicit ViT-L architecture')
        # Read tensor headers only: no allocation of a full model or GPU context.
        with safe_open(directory/'model.safetensors', framework='pt', device='cpu') as weights:
            keys = list(weights.keys())
            shapes = {name: weights.get_slice(name).get_shape() for name in
                ('embeddings.cls_token', 'embeddings.register_tokens', 'embeddings.patch_embeddings.weight')}
            if shapes != {'embeddings.cls_token': [1, 1, 1024],
                'embeddings.register_tokens': [1, 4, 1024],
                'embeddings.patch_embeddings.weight': [1024, 3, 16, 16]}:
                raise ValueError('Candidate safetensor layout differs from ViT-L adapter')
            layer_ids = sorted({int(name.split('.')[1]) for name in keys if name.startswith('layer.')})
            if layer_ids != list(range(24)):
                raise ValueError('Candidate checkpoint does not contain all 24 transformer layers')
        if any(file_sha(source/name) != digest for name, digest in source_hashes.items()):
            raise ValueError('User download changed during copy; installation is not accepted')
        source_provenance = {'registry': 'ModelScope', 'repository': repository,
            'revision': MODELSCOPE_REVISION, 'url': 'https://modelscope.cn/models/'+repository+'/files',
            'publisher_namespace': details['Path'], 'organization': details.get('Organization', {}).get('FullName'),
            'created_by': details.get('CreatedBy'), 'model_source': details.get('ModelSource'),
            'official_distribution_claimed': False,
            'official_content_verified_at_revision': DINO3_L_REVISION,
            'metadata_sha256': {name: file_sha(evidence_directory/name) for name in
                ('hf-public-pinned-model-metadata.json', 'modelscope-model-metadata.json',
                 'modelscope-pinned-files-metadata.json')}}
        manifest = {'repository': repository, 'revision': DINO3_L_REVISION,
            'license': 'DINOv3 License', 'license_source': 'LICENSE.md from pinned ModelScope revision; official Git blob verified',
            'files_sha256': {name: value['sha256'] for name, value in verified.items()},
            'algorithm_profile': DINO3_L_PROFILE, 'installation_source': source_provenance,
            'quality_status': 'installed_current_model_not_visual_acceptance'}
        (directory/'model-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        receipt.update(status='installed_current_model', ready=True, verified_files=verified,
            source_files_sha256=source_hashes, installation_source=source_provenance,
            model_file_bytes=(directory/'model.safetensors').stat().st_size,
            safetensors_header={'tensor_count': len(keys), 'required_shapes': shapes, 'transformer_layers': len(layer_ids)})
    except Exception as exc:
        receipt.update(status='local_model_verification_failed', error_type=type(exc).__name__,
            error=str(exc), status_code=getattr(getattr(exc, 'response', None), 'status_code', None))
    (directory/'installation-current.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (evidence_directory/'installation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--proxy')
    parser.add_argument('--evidence-directory', type=Path)
    args = parser.parse_args()
    result = install(args.source, proxy=args.proxy, evidence_directory=args.evidence_directory)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
