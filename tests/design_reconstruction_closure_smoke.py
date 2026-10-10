"""CPU regression for nested frozen JSON records and parent hardlink isolation."""
from __future__ import annotations

import hashlib
import argparse
import contextlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image

from design_reconstruction import materialize_plan
from portable_project import export_project,import_project
from verify_saved_native_delivery import json_records
from display_transform import source_transform_for_precision
from raw_support import RGB_DECODE_VERSION


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):return {'path':str(Path(path).resolve()),'sha256':digest(path),'bytes':Path(path).stat().st_size}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    return path


def check_closure(value):
    pending=[value];seen=set();count=0
    while pending:
        for item in json_records(pending.pop()):
            target=Path(item['path'])
            if target.suffix.lower() in {'.pt','.pth','.safetensors','.onnx','.exe','.dll','.py'}:
                continue
            assert target.is_file()
            assert digest(target)==item['sha256'],str(target)
            if 'bytes' in item:assert target.stat().st_size==item['bytes']
            count+=1
            if target.suffix=='.json' and target not in seen:
                seen.add(target);pending.append(json.loads(target.read_text('utf8')))
    return count


def main(output_project=None):
    if output_project is not None and Path(output_project).exists():
        raise ValueError('Retained closure proof project must be a new directory')
    project_context=(contextlib.nullcontext(str(Path(output_project).resolve())) if output_project is not None
        else tempfile.TemporaryDirectory(prefix='cosplay-owned-closure-'))
    with project_context as temporary:
        root=Path(temporary);root.mkdir(parents=True,exist_ok=True);cache=root/'cache';cache.mkdir()
        source=cache/'source.png';Image.new('RGB',(16,12),(30,70,100)).save(source)
        reference=cache/'reference.png';reference.write_bytes(b'original-design-bytes')
        payload=cache/'evidence.npz';payload.write_bytes(b'frozen-evidence-bytes')
        bottom=save(cache/'analysis.json',{'source':record(source),'asset':record(payload),
            'result_path':str((cache/'analysis.json').resolve())})
        middle=save(cache/'observation.json',{'binding':{'analysis':record(bottom)},'list':[record(bottom)]})
        leaves=[]
        for branch,content in (('left',b'left unique evidence'),('right',b'right unique evidence')):
            evidence=cache/branch/'evidence.npz';evidence.parent.mkdir();evidence.write_bytes(content)
            leaves.append(save(cache/branch/'same-name.json',{'evidence':record(evidence),
                'result_path':str((cache/branch/'same-name.json').resolve())}))
        deep=save(cache/'deep-20.json',{'same-prefix/path':record(leaves[0]),'same-prefix?path':record(leaves[1])})
        for level in reversed(range(20)):
            path=cache/f'deep-{level}.json'
            deep=save(path,{'deep_logical_key_'+'x'*180:record(deep),'result_path':str(path.resolve())})
        common='same_long_logical_prefix_'+'x'*300
        worker=cache/'worker.py';worker.write_bytes(b'current installed producer bytes')
        historical_worker_sha=hashlib.sha256(b'archived producing worker bytes').hexdigest()
        worker_record={'path':str(worker.resolve()),'sha256':historical_worker_sha,'bytes':len(b'archived producing worker bytes')}
        model=cache/'model.pth';model.write_bytes(b'large model represented by a tiny fixture')
        top=save(cache/'receipt.json',{'source_observation':{'binding':{'analysis':record(middle)}},
            'qualification':record(middle),'source':record(source),'deep_chain':record(deep),
            common+'-left':record(leaves[0]),common+'-right':record(leaves[1]),
            'producer':worker_record,'model_identity':record(model)})
        original={str(path):digest(path) for path in cache.rglob('*') if path.is_file()}
        run={'kind':'single_input_run','input_mode':'single_source','run_id':'fixture','sample':'fixture','revision':1,'root':str(root),'source':record(source),
            'reference':record(reference),'artifacts':{},'options':{'render_mode':'native_source_design'}}
        assets={}
        for key,values in (('foreground_premultiplied',np.full((12,16,3),.1,np.float32)),('foreground_alpha',np.full((12,16),.8,np.float32)),('old_environment',np.full((12,16,3),.2,np.float32)),('validity',np.ones((12,16),np.float32))):
            path=cache/(key+'.npz');np.savez_compressed(path,values=values);assets[key]={**record(path),'size':[16,12]}
        plan={'schema_version':'design_reconstruction_plan.v1','enabled':False,'working_domain':'display_linear_sRGB',
            'source_sha256':run['source']['sha256'],'candidate_sha256':run['reference']['sha256'],
            'canvas':{'source_oriented_size':[16,12]},'execution':{'source_reference':'paired_native_baseline'},
            'source_baseline':{'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB',
                'source_sha256':run['source']['sha256'],'decode_version':RGB_DECODE_VERSION,
                'source_precision':'raster8_srgb','source_display':source_transform_for_precision('raster8_srgb'),
                'source_display_applied_once':True,
                'assets':{key:assets[key] for key in ('foreground_premultiplied','foreground_alpha','old_environment')}},
            'layers':[{'id':'background_design','kind':'background_design','enabled':False,
                'assets':{**assets,'background':{**assets['old_environment'],'representation':'sampling_validity_associated_RGB','validity_associated':True},'receipt':record(top)}}],
            'mask_roles':{'foreground_alpha':assets['foreground_alpha']}}
        from local_semantics_worker import CLASS_NAMES
        from sapiens_model_profiles import profile
        from source_semantic_binding import bind_source_semantics, load_source_semantics
        semantic_path=cache/'source-labels.npz';np.savez_compressed(semantic_path,labels=np.full((12,16),3,np.uint8))
        semantic_profile=profile('sapiens2_seg_0.4b','segmentation')
        plan['mask_roles']['source_semantics']=bind_source_semantics({'labels_path':str(semantic_path),
            'model':semantic_profile['repository'],'model_profile':'sapiens2_seg_0.4b',
            'model_revision':semantic_profile['revision'],'checkpoint_sha256':semantic_profile['checkpoint_sha256'],
            'taxonomy_sha256':hashlib.sha256(json.dumps(CLASS_NAMES,separators=(',',':')).encode()).hexdigest()},
            source_sha256=run['source']['sha256'],native_size=[16,12],proxy_size=[16,12])
        directory=root/'outputs/runs/fixture/inputs/design-reconstruction';directory.mkdir(parents=True)
        target=directory/'background_design-receipt.json';os.link(top,target)
        owned=materialize_plan(root,run,plan)
        checked=check_closure(owned)
        assert checked>=50
        assert all(digest(path)==expected for path,expected in original.items())
        assert not target.samefile(top),'Rewritten receipt must detach from the parent hardlink'
        owned_receipt=json.loads(Path(owned['layers'][0]['assets']['receipt']['path']).read_text('utf8'))
        assert owned_receipt['producer']==worker_record,'Current installed code relabeled the historical producer'
        assert not any(path.suffix.lower() in {'.py','.pth'} for path in directory.iterdir())
        left,right=(owned_receipt[common+suffix]['path'] for suffix in ('-left','-right'))
        assert left!=right, 'Distinct full logical/source paths collided after shortening'
        assert json.loads(Path(left).read_text('utf8'))['evidence']['sha256']!=json.loads(Path(right).read_text('utf8'))['evidence']['sha256']
        files=[path for path in directory.iterdir() if path.is_file()]
        maximum_name=max(len(path.name.encode('utf-16-le'))//2 for path in files)
        maximum_rewrite_path=max(len((str(path)+'.rewrite.tmp').encode('utf-16-le'))//2 for path in files)
        assert maximum_name<=96 and maximum_rewrite_path<=240
        first_owned_bytes={str(path):digest(path) for path in files}
        repeated=materialize_plan(root,run,plan)
        assert check_closure(repeated)>=checked
        assert all(digest(path)==expected for path,expected in first_owned_bytes.items())
        run['options']['appearance_revision']={'design_reconstruction_plan':owned}
        save(root/'outputs/runs/fixture/manifest.json',run)
        archive=root/'fixture.cosplay';export_project(root,'fixture',archive)
        with zipfile.ZipFile(archive) as bundle:
            assert not any(Path(name).suffix.lower() in {'.py','.pth'} for name in bundle.namelist())
        moved=import_project(root/'moved',archive,run_id='relocated')['run']
        moved_checked=check_closure(moved['options']['appearance_revision']['design_reconstruction_plan'])
        assert moved_checked>=checked
        moved_semantics=moved['options']['appearance_revision']['design_reconstruction_plan']['mask_roles']['source_semantics']
        values,binding=load_source_semantics(moved_semantics,source_sha256=run['source']['sha256'],native_size=[16,12],proxy_size=[16,12])
        assert (values==3).all() and Path(binding['path']).is_relative_to(root/'moved')
        assert binding['sha256']==plan['mask_roles']['source_semantics']['sha256']
        moved_receipt=json.loads(Path(moved['options']['appearance_revision']['design_reconstruction_plan']['layers'][0]['assets']['receipt']['path']).read_text('utf8'))
        assert moved_receipt['producer']['sha256']==historical_worker_sha
        assert moved_receipt['producer']['bytes']==worker_record['bytes']
        assert not Path(moved_receipt['producer']['path']).is_file(),'Unbundled historical code became a current runtime'
        report={'ok':True,'nested_hash_and_bytes_records':checked,'portable_records':moved_checked,
            'parent_source_JSON_and_payload_unchanged':True,'rewritten_hardlink_detached':True,
            'three_level_and_duplicate_reference_closure':True,'self_result_locator_final_hash':True,
            'twenty_level_full_logical_prefix_closure':True,'same_readable_prefix_distinct_sources_no_collision':True,
            'maximum_owned_filename_characters':maximum_name,'maximum_owned_atomic_rewrite_path_characters':maximum_rewrite_path,
            'stable_repeat_materialization_bytes':True,'tiny_actual_portable_relocation':True,
            'historical_producer_SHA_and_byte_count_preserved':True,'current_code_does_not_relabel_archived_pixels':True,
            'runtime_code_and_models_not_packaged':True,
            'independent_source_semantics_owned_and_portable':True,
            'API_calls':0,'GPU_calls':0}
        if output_project is not None:
            report['retained_project']=str(root)
            save(root/'proof.json',report)
        print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-project',type=Path)
    main(parser.parse_args().output_project)
