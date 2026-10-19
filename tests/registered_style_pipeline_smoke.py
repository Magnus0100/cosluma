"""The real final renderer, native ROI and portable consume one style asset."""
from pathlib import Path
import copy
import json
import tempfile

import numpy as np

from current_stack_fixture import prepare_fixture
from registered_reference_style_smoke import frozen
import registered_reference_style as style
from design_reconstruction import plan_for_run, materialize_plan
from native_source_baseline import source_for_plan
from float_appearance import render_source_float, prepare_registered_style_input, apply_registered_style_layers
from native_roi_render import render_native_roi
from display_transform import source_transform_for_precision, display_linear
from native_current_only_smoke import save
from portable_project import export_project, import_project


def check_hdr_editing(root, run, source, layer, native):
    """Real appearance/optics before style share the final SDR boundary."""
    plan=plan_for_run(run)
    original=list(plan['layers'])
    revision=run['options']['appearance_revision']
    old_params=copy.deepcopy(revision.get('params'))
    for label,params,bloom in (
            ('exposure_saturation',{'exposure':2.,'saturation':1.},False),
            ('exposure_bloom',{'exposure':2.},True)):
        revision['params']={'global':params}
        plan['layers']=[row for row in original if row['kind']!=style.KIND]
        if bloom:
            plan['layers'].append({'id':'test_bloom','kind':'bloom','enabled':True,'strength':1.,
                'params':{'source':'visible_composite','threshold_linear':.1,'radius_fraction':.02},'assets':{}})
        ungraded,_=render_source_float(source,run)
        assert ungraded.max()>1 or ungraded.min()<0,label
        fit_baseline=prepare_registered_style_input(ungraded)
        assert np.array_equal(fit_baseline,display_linear(ungraded,{'method':'clip_sdr'}))
        expected=style.apply(fit_baseline,layer,(0,0,*native),native)
        plan['layers'].append(layer)
        final,_=render_source_float(source,run)
        assert np.array_equal(final,expected),label
        for box in ((0,0,1,native[1]),(5,111,48,258),(20,125,30,134)):
            roi,_=render_native_roi(run,box)
            assert np.array_equal(roi,final[box[1]:box[3],box[0]:box[2]]),(label,box)
        # Disabling style preserves the pre-existing HDR working path; only
        # its final encoder clips. The style entry must not change off mode.
        layer['strength']=0
        off,_=render_source_float(source,run)
        assert np.array_equal(off,ungraded),label
        layer['strength']=1.
    plan['layers']=original
    if old_params is None:revision.pop('params',None)
    else:revision['params']=old_params
    # No style leaves the working buffer unchanged; nonfinite values may not
    # enter the SDR conversion, even though np.clip would otherwise hide inf.
    hdr=np.full((2,2,3),1.6,np.float32)
    empty={'enabled':True,'layers':[]}
    unchanged,_=apply_registered_style_layers(hdr,empty,run)
    assert unchanged is hdr
    for value in (np.nan,np.inf,-np.inf):
        bad=hdr.copy();bad[0,0,0]=value
        try:prepare_registered_style_input(bad)
        except ValueError:pass
        else:raise AssertionError('Nonfinite style input entered SDR')


def check_currentness_receipt(root, run, layer):
    from preparation_currentness import preparation_currentness
    import reference_pipeline as pipeline
    from unittest.mock import patch
    stack=next(row for row in plan_for_run(run)['layers'] if row['kind']=='reference_appearance_stack')
    snapshot=root/'current-preparation-snapshot.json'
    style._write(snapshot,{'recipe':pipeline.RECIPE,'pipeline_inputs':{
        'recipe':pipeline.RECIPE,'code':pipeline._code_identity(root)}})
    stack['assets']['preparation_snapshot']=style._asset(snapshot)
    with patch('numpy.load',side_effect=AssertionError('Currentness decoded native assets')):
        current=preparation_currentness(root,run)
    assert current['is_current'] and not current['reprepare_required'],current
    # Recommendation data is an actual style producer dependency. A catalog
    # update invalidates new fitting without decoding or mutating frozen work.
    import preset_store
    alternate=root/'changed-preset-catalog.json'
    alternate.write_bytes(preset_store.DEFAULT_PATH.read_bytes()+b'\n ')
    with patch.object(preset_store,'DEFAULT_PATH',alternate), \
            patch('numpy.load',side_effect=AssertionError('Currentness decoded native assets')):
        changed_catalog=preparation_currentness(root,run)
    assert changed_catalog['reference_appearance_is_current'],changed_catalog
    assert not changed_catalog['is_current'],changed_catalog
    assert 'presets/presets.json' in changed_catalog['registered_reference_style']['changed_producers']
    path=Path(layer['assets']['contract']['path']);original=path.read_bytes()
    path.write_bytes(original+b'\n ')
    changed=preparation_currentness(root,run)
    assert changed['state']=='unavailable' and not changed['is_current'] and changed['reprepare_required'],changed
    path.write_bytes(original)
    unavailable=path.with_suffix('.missing');path.replace(unavailable)
    try:
        missing=preparation_currentness(root,run)
        assert missing['state']=='unavailable' and not missing['is_current'] and missing['reprepare_required'],missing
    finally:unavailable.replace(path)
    assert preparation_currentness(root,run)['is_current']


def check_material_permission_replay(root):
    from unittest.mock import patch
    from skin_illumination_smoke import fixture, v2_layer
    values,_,_,_,context,assets,evidence,_=fixture(root/'material')
    layer=v2_layer(root/'material',values,context,assets,evidence)
    contract=style.validate_layer(layer)['identity']['reference_contract']
    intent={'mode':'create_reference','brief':'清除手臂和手的杂光并统一肤色',
        'constraints':{'reference_role':'source_conditioned_design'}}
    run={'source':{'sha256':contract['source_sha256']},
        'reference':{'sha256':contract['reference_sha256']},'task_intent':intent,
        'task_plan':{'ready':True,'intent':intent,'creative_direction':{'material_refinement':context['goal']}}}
    plan={'enabled':True,'layers':[layer],'canvas':{'source_oriented_size':[224,288]}}
    with patch('creative_direction.skin_cast_contract_for',return_value=context['goal']):
        actual,_=apply_registered_style_layers(values,plan,run)
        assert np.array_equal(actual,style.apply(values,layer,(0,0,224,288),(224,288)))
        for change in ('missing_permission','changed_intent','unchecked'):
            other=copy.deepcopy(run)
            if change=='missing_permission':other['task_plan']['creative_direction'].clear()
            elif change=='changed_intent':other['task_intent']={**intent,'brief':'保留手臂原有红光'}
            else:other['task_plan']['ready']=False
            try:apply_registered_style_layers(values,plan,other)
            except ValueError:pass
            else:raise AssertionError('Skin permission crossed its checked scope: '+change)
    with patch('creative_direction.skin_cast_contract_for',return_value=None):
        try:apply_registered_style_layers(values,plan,run)
        except ValueError:pass
        else:raise AssertionError('Skin permission survived an explicit current veto')


def main():
    with tempfile.TemporaryDirectory(prefix='registered-style-pipeline-') as folder:
        root=Path(folder);native=(67,289)
        run=prepare_fixture(root/'fixture',native=native)
        plan=plan_for_run(run)
        plan['source_baseline']['source_display']=source_transform_for_precision('raster8_srgb')
        plan['source_baseline']['source_sha256']=run['source']['sha256']
        source=source_for_plan(plan);baseline,_=render_source_float(source,run)
        coefficients=np.zeros((11,7,12),np.float32)
        coefficients[...,0]=.05;coefficients[...,4]=-.1
        coefficients[...,9]=np.linspace(-.012,.025,11)[:,None]
        coefficients[...,10]=-.015;coefficients[...,11]=-.02
        layer=frozen(root/'style',coefficients,native);layer['id']=style.KIND
        receipt=style.validate_layer(layer)
        receipt['identity']['reference_contract']={
            'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256']}
        receipt['identity']['producer_code']=style.producer_code()
        style._write(layer['assets']['contract']['path'],receipt)
        layer['assets']['contract']=style._asset(layer['assets']['contract']['path'])
        layer['params']['identity_sha256']=style._digest(receipt['identity'])
        expected=style.apply(baseline,layer,(0,0,*native),native)
        plan['layers'].append(layer)
        whole,operator=render_source_float(source,run)
        assert np.array_equal(whole,expected)
        assert any(row['kind']==style.KIND and row['applied']
            for row in operator['design_reconstruction_plan']['layers'])
        for box in ((0,0,1,289),(5,111,48,258),(20,125,30,134)):
            crop,_=render_native_roi(run,box)
            assert np.array_equal(crop,whole[box[1]:box[3],box[0]:box[2]])
        layer['strength']=0
        off,_=render_source_float(source,run)
        assert np.array_equal(off,baseline)
        layer['strength']=1
        wrong=copy.deepcopy(run);wrong['reference']['sha256']='e'*64
        try:render_source_float(source,wrong)
        except ValueError:pass
        else:raise AssertionError('Style applied to another reference')
        check_hdr_editing(root,run,source,layer,native)
        check_currentness_receipt(root,run,layer)
        run['options']['appearance_revision']['design_reconstruction_plan']=materialize_plan(root,run,plan)
        save(root,run,whole,operator)
        archive=root/'current.cosplayproj';export_project(root,run['run_id'],archive)
        imported=import_project(root/'empty',archive)['run']
        crop,_=render_native_roi(imported,(5,111,48,258))
        assert np.array_equal(crop,whole[111:258,5:48])
        check_material_permission_replay(root)
    print(json.dumps({'ok':True,'final_operator_applied_once':True,
        'full_thin_and_row_boundary_ROI_bitexact':True,'zero_strength_exact':True,
        'wrong_reference_refused':True,'portable_frozen_replay':True,'model_submissions':0}))


if __name__=='__main__':main()
