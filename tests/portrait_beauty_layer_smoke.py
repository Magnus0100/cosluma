"""Generated face appearance survives native/proxy/ROI replay without models."""
from pathlib import Path
import json
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from portrait_beauty_layer import (freeze_prepared, load_result, apply_result, apply_for_run, sha256,
    VISIBLE_FACE_POLICY, hair_correction_guard_for_run, sample_visible_face_support)
from display_transform import source_transform_for_precision
from raw_support import RGB_DECODE_VERSION


def main():
    with tempfile.TemporaryDirectory() as value:
        directory = Path(value)
        image = directory/'generated.png'
        Image.new('RGB',(10,12),(220,140,110)).save(image)
        weight = np.zeros((12,10),np.float32)
        weight[2:10,2:8] = 1
        weights = directory/'weights.npy'; np.save(weights,weight)
        transform = source_transform_for_precision('raster8_srgb')
        prepared = {'schema':'beautygrpo_layer_prepare.v1','working_domain':'display_linear_sRGB',
            'source_path':str(directory/'source.png'),'source_sha256':'bound-source',
            'native_size':[40,30],'source_display':transform,
            'source_precision':{'source_precision':'raster8_srgb','decode_version':RGB_DECODE_VERSION},
            'model_identity':{'name':'BeautyGRPO','lora_active':True},
            'faces':[{'id':'face-01','native_box':[5,6,15,18],
                'generated_native_roi':str(image),'generated_native_roi_sha256':sha256(image),
                'composition_weight':str(weights),'composition_weight_sha256':sha256(weights),
                'composition_policy':VISIBLE_FACE_POLICY,'composition':{'policy':VISIBLE_FACE_POLICY,
                    'qualification_status':'full_visible_face','nonface_material_excluded':True,
                    'skin_only':False,'features_gate_applied':False,'raw_probability_applied':False,
                    'source_texture_restoration':False}}]}
        path, result = freeze_prepared(prepared,directory/'frozen')
        source = np.full((30,40,3),.2,np.float32)
        full = apply_result(source,result,parent=path.parent)
        assert np.any(full[8:16,7:13] != source[8:16,7:13])
        # Includes eyes/mouth positions: no skin-only or high-frequency gate.
        expected = np.asarray(Image.open(image),np.float32)/255
        from float_appearance import srgb_to_linear
        assert np.array_equal(full[8:16,7:13],srgb_to_linear(expected)[2:10,2:8])
        untouched = np.ones(source.shape[:2],bool); untouched[8:16,7:13] = False
        assert np.array_equal(full[untouched],source[untouched])
        assert apply_result(source,result,strength=0) is source
        assert np.array_equal(apply_result(source,result,disabled_faces=['face-01']),source)
        assert np.allclose(apply_result(source,result,strength=.5),source+(full-source)*.5)
        roi = apply_result(source[4:20,3:18],result,canvas_size=(40,30),box=(3,4,18,20))
        assert np.array_equal(roi,full[4:20,3:18])
        alpha = np.full((30,40),.3,np.float32); alpha_copy=alpha.copy()
        associated = apply_result(source*alpha[...,None],result,premultiplied_alpha=alpha)
        assert np.allclose(associated,full*alpha[...,None]) and np.array_equal(alpha,alpha_copy)
        proxy = apply_result(np.full((15,20,3),.2,np.float32),result,canvas_size=(20,15))
        assert np.isfinite(proxy).all() and np.any(proxy != .2)
        run = {'source':{'sha256':'bound-source'},'options':{
            'source_only_plan':{'schema':'source_only_display_plan.v1','working_domain':'display_linear_sRGB',
                'source_sha256':'bound-source','decode_version':RGB_DECODE_VERSION,
                'source_precision':'raster8_srgb','source_display':transform,
                'native_size':[40,30],'source_display_applied_once':True},
            'appearance_revision':{'portrait_beauty':{'result':str(path),'strength':1.}}}}
        replay,receipt = apply_for_run(source,run)
        assert np.array_equal(replay,full) and receipt['alpha_changed'] is False
        guard = hair_correction_guard_for_run(run)
        visibility = sample_visible_face_support(guard,(0,0,40,30),(40,30))
        assert np.array_equal(visibility[6:18,5:15],weight)
        assert np.array_equal(sample_visible_face_support(guard,(3,4,18,20),(40,30)),visibility[4:20,3:18])
        run['options']['appearance_revision']['portrait_beauty'].update(strength=0,disabled_faces=['face-01'])
        assert hair_correction_guard_for_run(run)['binding'] == guard['binding']
        assert np.array_equal(sample_visible_face_support(hair_correction_guard_for_run(run),(0,0,40,30),(40,30)),visibility)
        run['options']['appearance_revision']['portrait_beauty'].update(strength=1,disabled_faces=[])
        assert hair_correction_guard_for_run({'options':{}}) is None
        assert not sample_visible_face_support(None,(0,0,40,30),(40,30)).any()
        invalid = json.loads(path.read_text(encoding='utf8'))
        del invalid['faces'][0]['composition']['nonface_material_excluded']
        invalid_path = directory/'unqualified.json';invalid_path.write_text(json.dumps(invalid),encoding='utf8')
        run['options']['appearance_revision']['portrait_beauty']['result'] = str(invalid_path)
        try: hair_correction_guard_for_run(run)
        except ValueError: pass
        else: raise AssertionError('Arbitrary masked beauty layer supplied a visible-face hair veto')
        run['options']['appearance_revision']['portrait_beauty']['result'] = str(path)
        reloaded,_ = load_result(path)
        assert np.array_equal(apply_result(source,reloaded),full)
        run['source']['sha256'] = 'wrong'
        try: apply_for_run(source,run)
        except ValueError: pass
        else: raise AssertionError('An unrelated source consumed the beauty layer')
        try: hair_correction_guard_for_run(run)
        except ValueError: pass
        else: raise AssertionError('Wrong source consumed visible-face support')
        # A changed asset must fail even in a warm replay cache.
        asset = Path(result['faces'][0]['pixels']['path'])
        with asset.open('ab') as stream: stream.write(b'changed')
        try: apply_result(source,result)
        except ValueError: pass
        else: raise AssertionError('Changed generated pixels were replayed')
    print('portrait_beauty_layer_smoke: passed')


if __name__ == '__main__':
    main()
