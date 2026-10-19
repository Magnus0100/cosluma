"""Actual current encoders, immutable masters and retired-format rejection."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import tifffile
from PIL import Image
import single_source_workflow as workflow
from export_artifact import export_source_run
from native_source_baseline import prepare_source_only
from raw_support import RGB_DECODE_VERSION,linear_preview,save_render_master,save_scene_master


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cosplay-export-current-') as folder:
        root=Path(folder);directory=root/'outputs/runs/fixture';directory.mkdir(parents=True)
        source=root/'source.png';Image.new('RGB',(61,47),(89,143,207)).save(source)
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        run={'kind':'single_input_run','input_mode':'single_source','run_id':'fixture','sample':'fixture','revision':2,
             'source':{'path':str(source),'sha256':source_sha},'options':{'render_mode':'native_source_design'}}
        values=np.linspace(.01,.95,47*61*3,dtype=np.float32).reshape(47,61,3)
        pixels,local=prepare_source_only(values,run,{'decode_version':RGB_DECODE_VERSION,'source_precision':'raster8_srgb'})
        run['options']['source_only_plan']=local
        draft=directory/'auto_draft.png';master=directory/'render_master.tif';scene=directory/'render_scene.tif'
        preview=linear_preview(pixels,display_transform={'method':'clip_sdr'});preview.save(draft,icc_profile=preview.info['icc_profile'])
        save_render_master(pixels,master,display_transform={'method':'clip_sdr'})
        save_scene_master(pixels,scene,working_domain='display_linear_sRGB')
        run['render']={'operator':{'working_domain':'display_linear_sRGB'},'render_precision':{
            'working_domain':'display_linear_sRGB','working_precision':'float32_linear','master_encoding':'uint16_srgb'}}
        run['artifacts']={'auto_draft':str(draft),'render_master':str(master),'render_scene_master':str(scene)}
        manifest=directory/'manifest.json';manifest.write_text(json.dumps(run),'utf8')
        delivery=root/'accepted';delivery.mkdir()
        for path in (draft,master,scene,manifest):(delivery/path.name).write_bytes(path.read_bytes())
        review=delivery/'visual_review.json';review.write_text('{}','utf8')
        protected={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in delivery.iterdir()}
        def export(format,artifact='draft'):
            return export_source_run(root,'fixture',directory=str(root/'exports'),format=format,artifact=artifact)
        for format in ('JPEG','PNG','TIFF','TIFF16','TIFF32_LINEAR'):
            result=export(format);metadata=result['metadata']
            assert metadata['working_domain']=='display_linear_sRGB' and not metadata['export_rerendered']
            assert metadata['actual_bit_depth']==(32 if format=='TIFF32_LINEAR' else 16 if format=='TIFF16' else 8)
            if format in ('TIFF16','TIFF32_LINEAR'):
                original=scene if format=='TIFF32_LINEAR' else master
                assert Path(result['path']).read_bytes()==original.read_bytes()
            else:
                with Image.open(result['path']) as actual:
                    assert actual.size==(61,47) and actual.mode=='RGB' and actual.info.get('icc_profile')
            checks.append(format)
        normal_copy = shutil.copy2
        def corrupt_same_size(source, target):
            normal_copy(source, target)
            with Path(target).open('r+b') as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
        for format in ('PNG', 'TIFF16', 'TIFF32_LINEAR'):
            with patch('export_artifact.shutil.copy2', side_effect=corrupt_same_size):
                try: export(format)
                except IOError: pass
                else: raise AssertionError('Same-size export corruption was not rejected')
        checks.append('copied_target_content_verified')
        with patch.object(workflow,'_delivery_record',return_value=(delivery,{'reviewer':'fixture'})):
            master.write_bytes(b'changed mutable master')
            result=export('TIFF16','delivery')
            assert result['metadata']['approved'] and Path(result['path']).read_bytes()==(delivery/master.name).read_bytes()
        master.write_bytes((delivery/master.name).read_bytes())
        assert protected=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in delivery.iterdir()}
        checks.append('immutable_approved_master')
        def reject(label,mutation,format='TIFF16'):
            changed=copy.deepcopy(run);mutation(changed);manifest.write_text(json.dumps(changed),'utf8')
            try:export(format)
            except (ValueError,FileNotFoundError,RuntimeError):pass
            else:raise AssertionError(label+' was accepted')
            manifest.write_text(json.dumps(run),'utf8');checks.append(label)
        reject('missing_true_master',lambda r:r['artifacts'].pop('render_master'))
        reject('retired_scene_render',lambda r:r['render']['operator'].update(working_domain='scene_linear_sRGB'))
        reject('missing_current_plan',lambda r:r['options'].pop('source_only_plan'))
        reject('pending_continuation',lambda r:r.update(continuation_edit={'render_state':'pending'}))
        reject('raw_decode_is_not_export',lambda r:None,format='TIFF16_SOURCE')
        tifffile.imwrite(master,np.zeros((47,61,3),np.uint8),photometric='rgb')
        reject('rgb8_container_is_not_uint16_master',lambda r:None)
        master.write_bytes((delivery/master.name).read_bytes())
        save_scene_master(pixels,scene,working_domain='scene_linear_sRGB')
        reject('scene_domain_header_is_not_display_master',lambda r:None,format='TIFF32_LINEAR')
    print(json.dumps({'ok':True,'checks':checks,'api_calls':0,'gpu_calls':0}))


if __name__=='__main__':main()
