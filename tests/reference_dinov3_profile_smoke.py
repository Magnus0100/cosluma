"""CPU current ViT-L token/grid, content binding and cache contracts."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import reference_local_correspondence as module
from reference_local_correspondence_worker import patch_tokens, extract, geometric_matches, build_maps
from reference_automatic_layers import pack_correspondence, load_maps, MAP_REPRESENTATION


def rejected(call):
    try: call()
    except (ValueError, RuntimeError, OSError): return
    raise AssertionError('Invalid or old DINO profile/token/layout was accepted')


def fake_install(path):
    path.mkdir(parents=True)
    config = module.algorithm_configuration()['expected_config']
    (path/'config.json').write_text(json.dumps(config), encoding='utf8')
    for name in ('preprocessor_config.json', 'README.md', 'model.safetensors', 'LICENSE.md'):
        (path/name).write_bytes(('CPU fixture, never executable model: '+name).encode())
    files = {name: module.file_sha(path/name) for name in
        ('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors', 'LICENSE.md')}
    (path/'model-manifest.json').write_text(json.dumps({
        'repository': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'revision': module.DINO3_L_REVISION, 'license': 'cpu-fixture-only',
        'files_sha256': files}), encoding='utf8')


def main():
    import torch
    layout = module.TOKEN_LAYOUT
    hidden = torch.zeros((1, 41, 1024)); hidden[:, :5] = 999
    hidden[:, 5:] = torch.eye(36, 1024)
    tokens = patch_tokens(hidden, (96,96), patch_size=16, token_layout=layout)
    assert tuple(tokens.shape)==(36,1024) and tokens.max()==1
    for wrong in ({**layout,'embedding_dimension':768}, {**layout,'register_tokens':0}):
        rejected(lambda: patch_tokens(hidden,(96,96),patch_size=16,token_layout=wrong))
    rejected(lambda: patch_tokens(hidden[:,:-1],(96,96),patch_size=16,token_layout=layout))
    rejected(lambda: patch_tokens(hidden,(97,96),patch_size=16,token_layout=layout))
    for old in ('legacy_dinov2_small','candidate_dinov3_vitb16','candidate_dinov3_vitl16','latest_unknown'):
        rejected(lambda: module.algorithm_configuration(old))
        rejected(lambda: module.algorithm_profile_for_run({'options': {'algorithm_profile':old}}))
    assert module.algorithm_profile_for_run({})==module.DINO3_L_PROFILE=='dinov3_vitl16'
    class FakeModel:
        def __call__(self, *, pixel_values):
            assert pixel_values.device.type=='cpu' and pixel_values.dtype==torch.float32
            return SimpleNamespace(last_hidden_state=hidden)
    labels=np.ones((96,96),np.uint8)
    observed=extract(FakeModel(),Image.new('RGB',(96,96),'#505050'),labels,'cpu',maximum_edge=96)
    np.testing.assert_array_equal(observed['features'],torch.eye(36,1024).numpy())
    np.testing.assert_array_equal(observed['points'][0],[7.5,7.5])
    reference={**observed,'points':observed['points']+8}
    maps,diag=build_maps(observed,reference,labels,labels)
    assert maps['valid'].sum()>3000
    yy,xx=np.nonzero(maps['valid'])
    assert np.max(abs(maps['ref_x'][yy,xx]-xx-8))<1e-4
    assert np.max(abs(maps['ref_y'][yy,xx]-yy-8))<1e-4
    assert diag['regions']['clothing']['methods']==['dinov3_mutual_semantic_neighbours_local_geometry']
    rejected(lambda: geometric_matches([],observed,{**reference,'patch_size':14}))
    real_root=Path(__file__).resolve().parents[1]
    runtime=real_root/'.cache/local-models/sapiens2/runtime/Scripts/python.exe'
    with tempfile.TemporaryDirectory(prefix='dinov3-current-cpu-') as folder:
        root=Path(folder);model=root/'model';fake_install(model)
        registry={'python':str(runtime),'dinov3_model_root':str(model)}
        with patch('current_model_registry.load_registry',return_value=registry):
            identity=module.local_correspondence_identity(root)
            assert identity['token_layout']['embedding_dimension']==1024
            manifest_path=model/'model-manifest.json';original_manifest=manifest_path.read_bytes()
            config_path=model/'config.json';original_config=config_path.read_bytes()
            for key,wrong in [('hidden_size',768),('num_hidden_layers',12),('num_attention_heads',12),('num_register_tokens',0)]:
                config=json.loads(original_config);config[key]=wrong;config_path.write_text(json.dumps(config))
                manifest=json.loads(original_manifest);manifest['files_sha256']['config.json']=module.file_sha(config_path)
                manifest_path.write_text(json.dumps(manifest))
                rejected(lambda: module.local_correspondence_identity(root))
            config_path.write_bytes(original_config);manifest_path.write_bytes(original_manifest)
            manifest=json.loads(original_manifest);manifest['revision']='wrong-revision';manifest_path.write_text(json.dumps(manifest))
            rejected(lambda: module.local_correspondence_identity(root))
            manifest_path.write_bytes(original_manifest)
            actual={role:root/(role+'.png') for role in ('source','reference')}
            for path in actual.values():Image.new('RGB',(96,96),'#505050').save(path)
            run={role:{'path':str(path),'sha256':module.file_sha(path),'oriented_size':[96,96]} for role,path in actual.items()}
            run['options']={'algorithm_profile':module.DINO3_L_PROFILE};requests=[]
            def worker(args,**kwargs):
                request=json.loads(Path(args[-1]).read_text());requests.append(request)
                assert request['identity']['algorithm_profile']==module.DINO3_L_PROFILE
                assert Path(request['model_directory'])==model
                directory=Path(request['output_directory']);assert module.DINO3_L_PROFILE in directory.parts
                asset=directory/'local-correspondence.npz';np.savez_compressed(asset,**maps)
                receipt={'schema':module.SCHEMA,'identity':request['identity'],'binding':request['binding'],
                    'source_size':[96,96],'reference_size':[96,96],'proxy_size':[96,96],
                    'coordinate_space':'source_proxy','reference_coordinate_space':'original_oriented_reference_pixels',
                    'confidence_is_spatial_evidence_not_effect_dose':True,'person_geometry_warped':False,
                    'reference_texture_imported':False,'background_H_used':False,
                    'asset':{'path':str(asset),'sha256':module.file_sha(asset),'representation':MAP_REPRESENTATION},'regions':diag['regions']}
                (directory/'receipt.json').write_text(json.dumps(receipt));return SimpleNamespace(returncode=0)
            with patch.object(module.subprocess,'run',side_effect=worker):
                first=module.prepare_local_correspondence(root,run,(96,96),labels,labels)
                second=module.prepare_local_correspondence(root,run,(96,96),labels,labels)
                assert len(requests)==1 and not first['cache_hit'] and second['cache_hit']
                record=pack_correspondence(first,run,(96,96),root=root,source_labels=labels,reference_labels=labels)
                params={'binding':{role:run[role]['sha256'] for role in actual}}
                restored=load_maps(record,labels.shape,labels.shape,params,root=root,source_labels=labels,reference_labels=labels)
                np.testing.assert_array_equal(restored['valid'],maps['valid'])
                changed=copy.deepcopy(record);changed['algorithm_profile']='candidate_dinov3_vitl16'
                rejected(lambda:load_maps(changed,labels.shape,labels.shape,params,root=root,source_labels=labels,reference_labels=labels))
                altered=copy.deepcopy(first['receipt']);altered['identity']['algorithm_profile']='legacy_dinov2_small'
                rejected(lambda:module.validate_local_correspondence(altered))
            from install_local_dinov3 import git_blob_sha,verify_file
            fixture=root/'registry-file';fixture.write_bytes(b'published metadata verification CPU fixture')
            scope={'Size':fixture.stat().st_size,'Sha256':module.file_sha(fixture)}
            official={'size':fixture.stat().st_size,'blobId':git_blob_sha(fixture.read_bytes())}
            assert verify_file(fixture,scope,official)['sha256']==scope['Sha256']
            rejected(lambda:verify_file(fixture,scope,{**official,'blobId':'f'*40}))
            rejected(lambda:verify_file(fixture,scope,{**official,'lfs':{'sha256':'f'*64}}))
            fixture.write_bytes(fixture.read_bytes()+b' altered');rejected(lambda:verify_file(fixture,scope,official))
            manifest_path.unlink();rejected(lambda:module.local_correspondence_identity(root))
    print(json.dumps({'ok':True,'current_profile_only':True,'CLS_and_four_registers_excluded':True,
        'ViTL_1024_dimensions_and_architecture_bound':True,'patch16_coordinates_and_maps_verified':True,
        'cache_and_BS_binding_verified':True,'old_profile_and_published_metadata_tampering_rejected':True,
        'real_models_loaded':0,'GPU_inference':0,'network_calls':0}))


if __name__=='__main__':main()
