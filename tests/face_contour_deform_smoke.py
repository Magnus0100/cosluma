"""Pure CPU geometric inverse sampling; no detector or image generation runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import map_coordinates

import face_contour_deform as deform
from photo_shaping_geometry_smoke import synthetic_points


def detections(scale=1.,degrees=0.):
    return {'faces':[{'landmarks':synthetic_points(scale=scale,degrees=degrees).tolist()}]}


def parent():
    yy,xx=np.indices((512,512),dtype=np.float32)
    rgb=np.stack((.1+.6*xx/512,.15+.5*yy/512,.2+.3*(xx+yy)/1024),axis=-1).astype(np.float32)
    G=((xx+yy)%37/37).astype(np.float32)
    return rgb,G


class ContourDeformTests(unittest.TestCase):
    def test_rotated_scaled_maps_are_monotone_keep_protection_and_match_independent_rgb_G_sampling(self):
        rgb,G=parent()
        for scale,degrees in ((1.,0.),(.65,28.),(1.2,-36.)):
            for amount in (.15,.3,1.):
                with self.subTest(scale=scale,degrees=degrees,amount=amount):
                    mapping=deform.build_mapping(detections(scale,degrees),[512,512],amount)
                    output,transported=deform.sample_mapping(rgb,G,mapping)
                    left,top,right,bottom=mapping['box'];d=mapping['displacement'];h=np.asarray(mapping['h'],np.float64)
                    yy,xx=np.indices(d.shape,dtype=np.float64);sy=yy+top+h[1]*d;sx=xx+left+h[0]*d
                    moved=mapping['edit_support'];fixed=mapping['protected']|~moved
                    self.assertGreater(mapping['stats']['jacobian_min'],0)
                    self.assertTrue(np.any(moved));self.assertTrue(np.array_equal(output[top:bottom,left:right][fixed],rgb[top:bottom,left:right][fixed]))
                    self.assertTrue(np.array_equal(transported[top:bottom,left:right][fixed],G[top:bottom,left:right][fixed]))
                    expected_G=map_coordinates(G,[sy,sx],order=1,prefilter=False,mode='nearest')
                    np.testing.assert_array_equal(transported[top:bottom,left:right][moved],expected_G[moved])
                    for channel in range(3):
                        expected=map_coordinates(rgb[...,channel],[sy,sx],order=1,prefilter=False,mode='nearest')
                        np.testing.assert_array_equal(output[top:bottom,left:right,channel][moved],expected[moved])
                    self.assertTrue(np.any(transported!=G));self.assertFalse(np.array_equal(output,rgb))
                    self.assertFalse(np.shares_memory(output,rgb));self.assertFalse(np.shares_memory(transported,G))

    def test_amount_zero_exact_copy_and_A_B_always_consume_same_original(self):
        rgb,G=parent();original=rgb.copy();original_G=G.copy();observed=detections()
        zero=deform.build_mapping(observed,[512,512],0);restored,restored_G=deform.sample_mapping(rgb,G,zero)
        np.testing.assert_array_equal(restored,rgb);np.testing.assert_array_equal(restored_G,G)
        self.assertFalse(np.shares_memory(restored,rgb));self.assertFalse(np.shares_memory(restored_G,G))
        a=deform.build_mapping(observed,[512,512],.15);b=deform.build_mapping(observed,[512,512],.3)
        first,_=deform.sample_mapping(rgb,G,a);second,_=deform.sample_mapping(rgb,G,b)
        repeated,_=deform.sample_mapping(rgb,G,b)
        np.testing.assert_array_equal(second,repeated);self.assertFalse(np.array_equal(first,second))
        np.testing.assert_array_equal(rgb,original);np.testing.assert_array_equal(G,original_G)
        for uniform in (0,1):
            _,value=deform.sample_mapping(rgb,uniform,b);self.assertEqual(value,uniform)

    def test_current_runtime_identity_and_external_rebuild_reject_self_consistent_mapping_changes(self):
        observed=detections();mapping=deform.build_mapping(observed,[512,512],.3)
        verified=deform.verify_mapping(mapping,observed,[512,512],.3)
        self.assertEqual(verified['mapping_id'],mapping['mapping_id'])
        runtime=deform.producer_identity();self.assertIn('numpy',runtime['runtime']);self.assertIn('scipy',runtime['runtime'])
        self.assertIn('scipy.ndimage._nd_image',runtime['native_sampling_binaries'])
        forged=copy.deepcopy(mapping);forged['displacement']*=.5
        forged['edit_support']=forged['displacement']!=0
        forged['stats'].update(deform._jacobian(forged['displacement'],forged['h']));deform._seal(forged)
        with self.assertRaisesRegex(ValueError,'checked canonical'):
            deform.verify_mapping(forged,observed,[512,512],.3)
        changed=copy.deepcopy(observed);changed['faces'][0]['landmarks'][152][0]+=.5
        with self.assertRaises(ValueError):deform.verify_mapping(mapping,changed,[512,512],.3)
        with patch.object(deform,'producer_identity',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'producer'):deform.sample_mapping(*parent(),mapping)

    def test_protected_array_type_tamper_fold_nonfinite_and_out_of_canvas_are_rejected(self):
        rgb,G=parent();original=deform.build_mapping(detections(),[512,512],.3)
        variants=[]
        changed=copy.deepcopy(original);changed['displacement'][changed['protected']]=1;variants.append(changed)
        changed=copy.deepcopy(original);changed['displacement'][~changed['allowed']]=1;variants.append(changed)
        changed=copy.deepcopy(original);changed['displacement']=changed['displacement'].astype(np.float64);variants.append(changed)
        changed=copy.deepcopy(original);changed['displacement'][10,10]=np.nan;variants.append(changed)
        changed=copy.deepcopy(original);changed['h']=[1.,1.];variants.append(changed)
        changed=copy.deepcopy(original);changed['box']=[-1,0,100,100];variants.append(changed)
        changed=copy.deepcopy(original);changed['displacement']*=1000;changed['edit_support']=changed['displacement']!=0
        changed['stats'].update(deform._jacobian(changed['displacement'],changed['h']));deform._seal(changed);variants.append(changed)
        for number,variant in enumerate(variants):
            with self.subTest(number=number):
                with self.assertRaises(ValueError):deform.sample_mapping(rgb,G,variant)

    def test_unseen_self_consistent_map_rederives_actual_canonical_geometry(self):
        mapping=deform.build_mapping(detections(),[512,512],.3)
        forged=copy.deepcopy(mapping);forged['displacement']*=.7;forged['edit_support']=forged['displacement']!=0
        forged['stats'].update(deform._jacobian(forged['displacement'],forged['h']));deform._seal(forged)
        with self.assertRaisesRegex(ValueError,'canonical geometry'):deform.sample_mapping(*parent(),forged)
        forged=copy.deepcopy(mapping);forged['h']=[-value for value in forged['h']]
        forged['stats'].update(deform._jacobian(forged['displacement'],forged['h']));deform._seal(forged)
        with self.assertRaisesRegex(ValueError,'canonical geometry'):deform.sample_mapping(*parent(),forged)

    def test_serialized_mapping_with_full_original_detection_identity_survives_cold_verification(self):
        observed=detections();observed.update(source_sha256='a'*64,source_native_size=[512,512],method='synthetic-boundary')
        mapping=deform.build_mapping(observed,[512,512],.3);rgb,G=parent();expected=deform.sample_mapping(rgb,G,mapping)
        with tempfile.TemporaryDirectory(prefix='face-contour-map-') as directory:
            directory=Path(directory)
            (directory/'mapping.json').write_text(json.dumps({key:value for key,value in mapping.items() if key not in deform.ARRAYS}),encoding='utf8')
            np.savez_compressed(directory/'mapping.npz',**{key:mapping[key] for key in deform.ARRAYS})
            loaded=json.loads((directory/'mapping.json').read_text('utf8'))
            with np.load(directory/'mapping.npz',allow_pickle=False) as arrays:
                loaded.update({key:arrays[key].copy() for key in deform.ARRAYS})
            deform._VERIFIED.clear()
            actual=deform.sample_mapping(rgb,G,loaded)
            for result,reference in zip(actual,expected):np.testing.assert_array_equal(result,reference)
            self.assertEqual(deform.verify_mapping(loaded,observed,[512,512],.3)['mapping_id'],mapping['mapping_id'])

    def test_native_fold_is_rejected_even_when_coordinates_remain_inside_canvas(self):
        mapping=deform.build_mapping(detections(),[512,512],.3);mapping['displacement']*=20
        mapping['stats'].update(deform._jacobian(mapping['displacement'],mapping['h']));deform._seal(mapping)
        self.assertLess(mapping['stats']['jacobian_min'],0)
        with self.assertRaisesRegex(ValueError,'folds'):deform.sample_mapping(*parent(),mapping)

    def test_bad_requests_ambiguous_geometry_no_motion_and_no_pixel_change_fail(self):
        for value in (True,-.1,1.1,float('nan'),float('inf'),'0.3'):
            with self.subTest(amount=value):
                with self.assertRaises(ValueError):deform.build_mapping(detections(),[512,512],value)
        for shape in ((512.,512),(0,512),(512,)):
            with self.assertRaises(ValueError):deform.build_mapping(detections(),shape,.3)
        observed=detections();observed['faces']*=2
        with self.assertRaises(ValueError):deform.build_mapping(observed,[512,512],.3)
        mapping=deform.build_mapping(detections(),[512,512],.3)
        with self.assertRaisesRegex(ValueError,'no actual RGB change'):
            deform.sample_mapping(np.full((512,512,3),.2,np.float32),0,mapping)
        no_motion=copy.deepcopy(mapping);no_motion['displacement'][:]=0;no_motion['edit_support'][:]=False
        no_motion['stats'].update(deform._jacobian(no_motion['displacement'],no_motion['h']));deform._seal(no_motion)
        with self.assertRaisesRegex(ValueError,'no valid effective motion'):deform.sample_mapping(*parent(),no_motion)

    def test_cancelled_build_and_sampling_keep_parent_arrays_unchanged(self):
        observed=detections()
        with self.assertRaises(InterruptedError):deform.build_mapping(observed,[512,512],.3,cancel=lambda:True)
        mapping=deform.build_mapping(observed,[512,512],.3);rgb,G=parent();saved=rgb.copy();saved_G=G.copy()
        cancelled=[False];sample=deform.map_coordinates
        def start_sampling(*args,**kwargs):
            result=sample(*args,**kwargs);cancelled[0]=True;return result
        with patch.object(deform,'map_coordinates',side_effect=start_sampling),self.assertRaises(InterruptedError):
            deform.sample_mapping(rgb,G,mapping,cancel=lambda:cancelled[0])
        np.testing.assert_array_equal(rgb,saved);np.testing.assert_array_equal(G,saved_G)
        bad=rgb.copy();bad[0,0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'finite'):deform.sample_mapping(bad,G,mapping)


if __name__=='__main__':unittest.main()
