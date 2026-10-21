"""Real float photographic development and owned CPU assets; no API/GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

import photo_appearance as appearance
import photo_development as development
import photo_appearance_smoke as asset_fixture
from photo_edit_tools import _close_loaded
from raw_support import open_linear


def settings(region=(0,0,1,1),ev=.6,warmth=.15,tint=.2,feather=.15,focus_sigma=None):
    item={'region':list(region),'exposure_ev':ev,'warmth_ev':warmth,'tint_ev':tint,'feather':feather}
    if focus_sigma is not None:item['focus_sigma']=focus_sigma
    return {'adjustments':[item]}


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.fx=asset_fixture.AppearanceTests(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root;self.binding=self.fx.binding
        self.enterContext(patch('reference_provider.main',side_effect=AssertionError('No generation')))

    def freeze(self,parent,name,parameters,*,band_rows=128):
        prior=appearance.load_asset(parent,**self.binding)
        try:values,coverage,stats=development.develop_float(prior['master'],parameters,band_rows=band_rows)
        finally:_close_loaded(prior)
        asset=appearance.freeze_candidate(self.root/name,kind='photographic_adjustment',master=values,
            parent=parent,coverage=coverage,photographic_operation={'operator':development.operator_for(parameters),
                'stats':stats,'producer':development.producer_identity()},**self.binding)
        return asset,values,coverage,stats

    def test_two_native_layouts_keep_sub_rgb8_float_and_outside_exact(self):
        for index,shape in enumerate(((17,23,3),(23,17,3))):
            values=np.full(shape,.125001,np.float32)
            values[...,1]+=np.arange(shape[1],dtype=np.float32)[None,:]*.000001
            base=self.fx.baseline('base'+str(index),values)
            asset,expected,coverage,stats=self.freeze(base,'developed'+str(index),settings((.2,.2,.8,.8)))
            loaded=appearance.load_asset(asset,**self.binding)
            try:
                self.assertTrue(np.array_equal(loaded['master'],expected))
                self.assertTrue(np.array_equal(loaded['master'][coverage==0],values[coverage==0]))
                self.assertFalse(np.array_equal(open_linear(loaded['preview_path'])[0],expected))
                self.assertGreater(stats['changed_pixels'],0)
                self.assertEqual(loaded['generated_coverage'],0)
                self.assertEqual(loaded['source_coverage'],1)
                self.assertTrue(loaded['receipt']['photographic_operation']['source_only_photographic_lineage'])
                self.assertEqual(loaded['receipt']['source_origin']['baseline_master_sha256'],
                    appearance.validate_asset(base,**self.binding)['files']['master']['sha256'])
            finally:_close_loaded(loaded)

    def test_ordered_soft_overrides_not_additive_ev_and_no_op_is_exact(self):
        values=np.full((19,29,3),.125001,np.float32)
        single=settings(ev=1,warmth=0,tint=0,feather=0)
        repeated={'adjustments':single['adjustments']*4}
        one,_,_=development.develop_float(values,single,band_rows=1)
        four,_,_=development.develop_float(values,repeated,band_rows=7)
        self.assertTrue(np.array_equal(one,four))
        final=settings(ev=-.5,warmth=0,tint=0,feather=0)
        override={'adjustments':single['adjustments']+final['adjustments']}
        actual,_,_=development.develop_float(values,override)
        expected,_,_=development.develop_float(values,final)
        self.assertTrue(np.array_equal(actual,expected))
        unchanged,_,stats=development.develop_float(values,settings(ev=0,warmth=0,tint=0))
        self.assertTrue(np.array_equal(unchanged,values));self.assertTrue(stats['no_change'])

    def test_focus_two_layouts_softens_background_preserves_skin_and_has_no_band_seams(self):
        for height,width in ((83,127),(127,83)):
            yy,xx=np.indices((height,width));texture=((xx+yy)%2).astype(np.float32)
            values=np.repeat((.1+.75*texture)[...,None],3,axis=2)
            # Skin-coloured synthetic portrait area is outside the chosen background.
            values[:,:width//3]=np.asarray([.280001,.160002,.110003],np.float32)
            parameters=settings((.5,0,1,1),ev=-.2,warmth=.1,tint=-.05,feather=.2,focus_sigma=.02)
            outputs=[]
            for rows in (1,7,128):
                actual,coverage,stats=development.develop_float(values,parameters,band_rows=rows)
                outputs.append(actual)
                self.assertTrue(np.array_equal(actual[coverage==0],values[coverage==0]))
                self.assertTrue(np.array_equal(actual[:,:width//3],values[:,:width//3]))
                self.assertGreater(stats['changed_pixels'],0)
                self.assertTrue(np.isfinite(actual).all())
                interior=(slice(2*height//5,3*height//5),slice(7*width//10,4*width//5))
                detail=lambda image:np.mean(np.std(image[interior].reshape(-1,3),axis=0))
                self.assertLess(detail(actual),detail(values)*.1)
            self.assertTrue(np.array_equal(outputs[0],outputs[1]))
            self.assertTrue(np.array_equal(outputs[0],outputs[2]))
            # Feather reduces the transition rather than switching to full blur at the edge.
            focused,coverage,_=development.develop_float(values,settings((.5,0,1,1),ev=0,warmth=0,
                tint=0,feather=.2,focus_sigma=.02))
            from scipy.ndimage import gaussian_filter
            sigma=.02*min(height,width);radius=int(np.ceil(4*sigma))
            full_reference=gaussian_filter(values,sigma=(sigma,sigma),axes=(0,1),
                radius=(radius,radius),mode='reflect')
            expected=values*(1-coverage[...,None])+full_reference*coverage[...,None]
            expected[coverage==0]=values[coverage==0]
            self.assertTrue(np.array_equal(focused,expected))
            change=np.mean(np.abs(focused-values),axis=2)
            self.assertLess(change[height//2,width//2+1],change[height//2,3*width//4])

    def test_focus_uses_immutable_parent_then_colour_and_zero_is_exact(self):
        from scipy.ndimage import gaussian_filter
        rng=np.random.default_rng(64);values=rng.uniform(.01,.7,(43,67,3)).astype(np.float32)
        plain=settings(feather=0)
        unchanged_zero,_,_=development.develop_float(values,settings(feather=0,focus_sigma=0))
        unchanged_old,_,_=development.develop_float(values,plain)
        self.assertTrue(np.array_equal(unchanged_zero,unchanged_old))
        exact,_,stats=development.develop_float(values,settings(ev=0,warmth=0,tint=0,focus_sigma=0))
        self.assertTrue(np.array_equal(exact,values));self.assertTrue(stats['no_change'])
        focus=settings(feather=0,focus_sigma=.02)
        sigma=.02*43;radius=int(np.ceil(4*sigma))
        blurred=gaussian_filter(values,sigma=(sigma,sigma),axes=(0,1),radius=(radius,radius),mode='reflect')
        expected,_,_=development.develop_float(blurred,plain)
        actual,_,_=development.develop_float(values,focus)
        self.assertTrue(np.array_equal(actual,expected))
        repeated={'adjustments':focus['adjustments']*3}
        actual_repeated,_,_=development.develop_float(values,repeated,band_rows=7)
        self.assertTrue(np.array_equal(actual_repeated,actual))
        overridden={'adjustments':focus['adjustments']+settings(feather=0,focus_sigma=0)['adjustments']}
        restored,_,_=development.develop_float(values,overridden)
        self.assertTrue(np.array_equal(restored,unchanged_old))

    def test_focus_optional_schema_rejects_nonfinite_bool_and_out_of_range(self):
        from photo_edit_session import validate_parameters
        for value in (float('nan'),float('inf'),-.00001,.020001,True,'0.01'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_parameters(settings(focus_sigma=value),development.PARAMETERS)
        for value in (0,.02):
            params=settings(focus_sigma=value)
            self.assertEqual(validate_parameters(params,development.PARAMETERS),params)

    def test_focus_banded_pixels_freeze_against_independent_default_band_recomputation(self):
        values=np.random.default_rng(24).uniform(.05,.75,(61,43,3)).astype(np.float32)
        base=self.fx.baseline('textured',values)
        parent=appearance.load_asset(base,**self.binding)
        try:prepared=parent['master'].copy()
        finally:_close_loaded(parent)
        parameters=settings((.3,.1,.9,.9),focus_sigma=.02,feather=.2)
        asset,expected,coverage,_=self.freeze(base,'focused-bands',parameters,band_rows=1)
        loaded=appearance.load_asset(asset,**self.binding)
        try:
            self.assertTrue(np.array_equal(loaded['master'],expected))
            self.assertTrue(np.array_equal(loaded['master'][coverage==0],prepared[coverage==0]))
        finally:_close_loaded(loaded)

    def test_relative_colour_and_shadow_exposure_are_bounded_without_recovering_black(self):
        values=np.full((11,13,3),[.12,.2,.12],np.float32);values[0,0]=0;values[-1,-1]=1
        actual,coverage,stats=development.develop_float(values,settings(ev=1,warmth=0,tint=.5,feather=0))
        self.assertEqual(coverage,1)
        self.assertGreater(actual[5,5].mean(),values[5,5].mean())
        self.assertLess(actual[5,5,1]/actual[5,5,0],values[5,5,1]/values[5,5,0])
        self.assertTrue(np.array_equal(actual[0,0],values[0,0]));self.assertEqual(stats['unrecoverable_black_pixels'],1)
        self.assertTrue(np.isfinite(actual).all());self.assertGreaterEqual(actual.min(),0);self.assertLessEqual(actual.max(),1)
        self.assertFalse(stats['physical_relighting_verified'])

    def test_mixed_and_full_parents_inherit_real_generation_coverage_and_resolution(self):
        base=self.fx.baseline()
        mixed=self.fx.mixed(base)
        full=appearance.freeze_candidate(self.root/'full',kind='generated_full',master=np.full((17,23,3),.2,np.float32),
            parent=base,generation=self.fx.generation('full-evidence',full=True),**self.binding)
        for index,parent in enumerate((mixed,full)):
            before=appearance.load_asset(parent,**self.binding)
            asset,_,_,_=self.freeze(parent,'photo'+str(index),settings(focus_sigma=.02))
            after=appearance.load_asset(asset,**self.binding)
            try:
                self.assertTrue(np.array_equal(after['generated_coverage'],before['generated_coverage']))
                self.assertEqual(after['receipt']['generated_resolution'],[9,11])
                self.assertEqual(after['receipt']['generation'],before['receipt']['generation'])
                self.assertFalse(after['receipt']['native_generated_detail'])
                self.assertFalse(after['receipt']['photographic_operation']['source_only_photographic_lineage'])
            finally:_close_loaded(before,after)

    def test_owned_receipt_reopens_after_original_parent_and_raw_are_deleted(self):
        base=self.fx.baseline();first,_,_,_=self.freeze(base,'first',settings())
        asset,values,_,_=self.freeze(first,'second',settings(ev=-.2,focus_sigma=.02))
        original=appearance.validate_asset(asset,**self.binding)
        moved=appearance.materialize_asset(asset,self.root/'owned',**self.binding)
        for row in (base,first,asset):shutil.rmtree(row['path'])
        self.fx.source.unlink()
        with patch('raw_support.open_linear',side_effect=AssertionError('No re-decode')):
            loaded=appearance.load_asset(moved,**self.binding)
        try:
            self.assertTrue(np.array_equal(loaded['master'],values))
            self.assertEqual(loaded['receipt'],original)
            operation=json.loads((Path(moved['path'])/'photographic-operation.json').read_text())
            self.assertEqual(operation['output']['preview'],original['files']['preview'])
            self.assertEqual(operation['image_generation_submissions'],0)
            self.assertFalse(operation['human_approved'])
            self.assertEqual(operation['operator']['processing_order'][0],'gaussian_current_parent_and_soft_region_blend')
            self.assertFalse(operation['operator']['physical_depth_of_field'])
        finally:_close_loaded(loaded)

    def test_fake_float_roundtrip_wrong_coverage_and_statistics_cannot_be_frozen(self):
        base=self.fx.baseline(values=np.full((17,23,3),.125001,np.float32))
        loaded=appearance.load_asset(base,**self.binding)
        params=settings((.2,.2,.8,.8),focus_sigma=.02)
        values,coverage,stats=development.develop_float(loaded['master'],params,band_rows=1)
        _close_loaded(loaded)
        operation={'operator':development.operator_for(params),
            'stats':stats,'producer':development.producer_identity()}
        for index,change in enumerate(('pixels','coverage','stats','producer')):
            candidate=values.copy();mask=coverage.copy();op=copy.deepcopy(operation)
            if change=='pixels':candidate[8,11,0]+=np.float32(.00001)
            elif change=='coverage':mask[8,11]=0
            elif change=='stats':op['stats']['covered_pixels']+=1
            else:op['producer']['photo_development.py']='a'*64
            with self.subTest(change=change),self.assertRaises(ValueError):
                appearance.freeze_candidate(self.root/('bad'+str(index)),kind='photographic_adjustment',master=candidate,
                    parent=base,coverage=mask,photographic_operation=op,**self.binding)
            self.assertFalse((self.root/('bad'+str(index))).exists())

    def test_operation_receipt_tampering_and_cancellation_preserve_parent(self):
        base=self.fx.baseline();asset,_,_,_=self.freeze(base,'photo',settings())
        path=Path(asset['path'])/'photographic-operation.json';original=path.read_bytes()
        path.write_bytes(original+b'changed')
        with self.assertRaises(ValueError):appearance.validate_asset(asset,**self.binding)
        path.write_bytes(original);appearance.validate_asset(asset,**self.binding)
        prior=appearance.load_asset(base,**self.binding)
        try:
            original=prior['master'].copy();calls=0
            def cancel_during_focus():
                nonlocal calls
                calls+=1
                return calls>=3
            with self.assertRaises(InterruptedError):development.develop_float(prior['master'],
                settings(focus_sigma=.02),cancel=cancel_during_focus)
            self.assertTrue(np.array_equal(prior['master'],original))
            self.assertTrue(np.isfinite(prior['master']).all())
        finally:_close_loaded(prior)
        self.assertFalse(list(self.root.glob('.*.claim')))

    def test_failed_preview_save_is_atomic_and_does_not_change_native_parent(self):
        base=self.fx.baseline();parent_bytes=(Path(base['path'])/'master.npy').read_bytes()
        loaded=appearance.load_asset(base,**self.binding)
        values,coverage,stats=development.develop_float(loaded['master'],settings())
        _close_loaded(loaded)
        operation={'operator':development.operator_for(settings()),'stats':stats,'producer':development.producer_identity()}
        with patch('raw_support.linear_preview',side_effect=OSError('Synthetic disk write failure')):
            with self.assertRaises(OSError):appearance.freeze_candidate(self.root/'failed-save',
                kind='photographic_adjustment',master=values,parent=base,coverage=coverage,
                photographic_operation=operation,**self.binding)
        self.assertFalse((self.root/'failed-save').exists())
        self.assertFalse(list(self.root.glob('.*.partial-*')));self.assertFalse(list(self.root.glob('.*.claim')))
        self.assertEqual((Path(base['path'])/'master.npy').read_bytes(),parent_bytes)
        appearance.validate_asset(base,**self.binding)


if __name__=='__main__':unittest.main()
