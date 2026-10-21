"""Offline canonical support with a mocked detector boundary and real pixels."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from current_model_registry import SCHEMA as REGISTRY_SCHEMA, PROFILES
from photo_appearance import create_source_baseline
from photo_editing import editing_scope, identity
from portrait_regions_worker import FACE_OVAL, FEATURE_POLYGONS
import photo_shaping_geometry as geometry


def synthetic_points(size=(512,512),scale=1.,degrees=0.):
    points=np.full((478,2),256.,np.float32)
    angles=np.arange(len(FACE_OVAL))*2*np.pi/len(FACE_OVAL)
    points[list(FACE_OVAL)]=np.stack((256+100*np.sin(angles),256-135*np.cos(angles)),axis=1)
    centers=[(216,230),(296,230),(256,305),(216,210),(296,210)]
    for indices,center in zip(FEATURE_POLYGONS,centers):
        angle=np.arange(len(indices))*2*np.pi/len(indices)
        points[list(indices)]=np.stack((center[0]+15*np.cos(angle),center[1]+6*np.sin(angle)),axis=1)
    angle=np.arange(len(geometry.NOSE))*2*np.pi/len(geometry.NOSE)
    points[list(geometry.NOSE)]=np.stack((256+9*np.cos(angle),266+15*np.sin(angle)),axis=1)
    angle=np.deg2rad(degrees);matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return (((points-256)@matrix.T)*scale+256)*np.array(size)/512


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='shaping-geometry-');self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.enterContext(patch.dict('os.environ',{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identity-cache')}))
        self.source=self.root/'source.png';Image.new('RGB',(512,512),'#657483').save(self.source)
        self.source_sha=sha256_file(self.source);self.scope=editing_scope({'work_type':'partial_composite'})
        self.design='d'*64
        self.binding={'source_sha256':self.source_sha,'scope':self.scope,'design_id':self.design}
        self.base=create_source_baseline(self.root/'u',source_path=self.source,**self.binding)
        self.python=self.root/'runtime/Scripts/python.exe';self.python.parent.mkdir(parents=True);self.python.write_bytes(b'fake-python')
        self.model=self.root/'face.task';self.model.write_bytes(b'fake-model')
        self.metadata=self.root/'runtime/Lib/site-packages/mediapipe-0.10.20.dist-info/METADATA'
        self.metadata.parent.mkdir(parents=True);self.metadata.write_text('Name: mediapipe\nVersion: 0.10.20\n',encoding='utf8')
        self.registry=self.root/'registry.json';self.registry.write_text(json.dumps({'schema':REGISTRY_SCHEMA,'ready':True,
            **PROFILES,'face_python':str(self.python),'face_model_path':str(self.model)}),encoding='utf8')
        self.calls=[]

    def detector(self,runtime,preview,output,cancel):
        self.calls.append((runtime,preview,output))
        document={'source_native_size':[512,512],'source_sha256':sha256_file(preview),
            'faces':[{'landmarks':synthetic_points().tolist()}],
            'method':'mediapipe_offline_multi_face_landmarks','model_sha256':runtime['model_sha256'],
            'mediapipe_version':runtime['mediapipe_version']}
        (output/'detections.json').write_text(json.dumps(document),encoding='utf8')

    def preflight(self):
        return {'checks':[
            {'key':'face_model_path.observed_identity','available':True,'path':str(self.model),'actual_sha256':sha256_file(self.model)},
            {'key':'face_python.observed_identity','available':True,'path':str(self.python),'actual_sha256':sha256_file(self.python)},
            {'key':'face_runtime.mediapipe','available':True,'path':str(self.metadata),'version':'0.10.20',
                'metadata_sha256':sha256_file(self.metadata)}]}

    def prepare(self,name='owned',**options):
        return geometry.prepare_geometry(self.root,self.base,output=self.root/name,registry_path=self.registry,
            preflight=options.pop('preflight',self.preflight()),**self.binding,**options)

    def test_face_only_registry_consumes_no_unrelated_profiles(self):
        document=json.loads(self.registry.read_text('utf8'))
        for key in PROFILES:document.pop(key,None)
        self.registry.write_text(json.dumps(document),encoding='utf8')
        with patch.object(geometry,'_run_detector',side_effect=self.detector):
            support=self.prepare()
        self.assertTrue(Path(support['path']).is_file())
        self.assertEqual(len(self.calls),1)

    def test_rotation_scale_keep_central_features_exact_zero_and_j_inside_e(self):
        for scale,degrees in ((1.,0.),(.65,28.),(1.2,-36.)):
            with self.subTest(scale=scale,degrees=degrees):
                planes,info=geometry._geometry({'faces':[{'landmarks':synthetic_points(scale=scale,degrees=degrees).tolist()}]},(512,512))
                self.assertTrue(np.any(planes['J']>0));self.assertTrue(np.all(planes['J'][planes['P']]==0))
                self.assertFalse(np.any((planes['J']>0)&~planes['E']));self.assertFalse(np.any(planes['P']&~planes['E']))
                self.assertTrue(np.all(planes['O'][planes['E']]==0));self.assertFalse(info['semantic_truth_claimed'])

    def test_no_multi_degenerate_small_clipped_and_nonfinite_stop_without_fallback(self):
        points=synthetic_points()
        invalid=[[],[{'landmarks':points.tolist()}]*2,[{'landmarks':points[:50].tolist()}],
            [{'landmarks':np.full((478,2),256.).tolist()}],[{'landmarks':synthetic_points(scale=.08).tolist()}],
            [{'landmarks':(points-[190,0]).tolist()}],[{'landmarks':np.full((478,2),np.nan).tolist()}]]
        for faces in invalid:
            with self.subTest(count=len(faces)):
                with self.assertRaises(ValueError):geometry._geometry({'faces':faces},(512,512))

    def test_canonical_oval_crossing_overlaps_and_non_neighbour_touch_are_rejected(self):
        points=synthetic_points();points[[338,297]]=points[[297,338]]
        with self.assertRaisesRegex(ValueError,'crossing'):
            geometry._geometry({'faces':[{'landmarks':points.tolist()}]},(512,512))
        for polygon in (
                [[0,0],[3,0],[1,0],[3,3],[0,3]],
                [[0,0],[3,0],[3,3],[0,0],[0,3]],
                [[0,0],[4,0],[4,4],[2,0],[0,4]],
                [[0,0],[4,0],[4,4],[1,0],[3,0],[0,4]]):
            with self.subTest(polygon=polygon):
                with self.assertRaises(ValueError):geometry._simple_polygon(polygon)

    def test_feature_point_groups_use_conservative_hulls_despite_zigzag_order(self):
        from scipy.ndimage import distance_transform_edt
        points=synthetic_points();indices=list(FEATURE_POLYGONS[3])
        points[indices]=points[indices][[0,5,1,6,2,7,3,8,4,9]]
        planes,info=geometry._geometry({'faces':[{'landmarks':points.tolist()}]},(512,512))
        box=info['box'];local=points-box[:2];size=(box[2]-box[0],box[3]-box[1])
        old=np.zeros(planes['P'].shape,bool)
        for group in (*FEATURE_POLYGONS,geometry.NOSE):old|=geometry._polygon(local[list(group)],size)
        previous_protection=distance_transform_edt(~old)<=info['feature_margin']
        self.assertFalse(np.any(previous_protection&~planes['P']))
        self.assertTrue(np.all(planes['J'][previous_protection]==0))
        self.assertEqual(len(info['normalized_protected_feature_polygons']),6)
        self.assertEqual(info['feature_protection'],geometry.PARAMETERS['feature_protection'])

    def test_feature_hull_may_cross_oval_inside_fixed_envelope_but_not_leave_envelope(self):
        points=synthetic_points();indices=list(FEATURE_POLYGONS[3]);angle=np.arange(len(indices))*2*np.pi/len(indices)
        points[indices]=np.stack((168+15*np.cos(angle),230+3*np.sin(angle)),axis=1)
        planes,info=geometry._geometry({'faces':[{'landmarks':points.tolist()}]},(512,512))
        box=info['box'];local=points-box[:2];size=(box[2]-box[0],box[3]-box[1])
        inside=geometry._polygon(local[list(FACE_OVAL)],size)
        self.assertTrue(np.any(planes['P']&~inside))
        self.assertFalse(np.any(planes['P']&~planes['E']))
        self.assertTrue(np.all(planes['J'][planes['P']]==0))
        points[indices,0]-=25
        with self.assertRaisesRegex(ValueError,'safe face envelope'):
            geometry._geometry({'faces':[{'landmarks':points.tolist()}]},(512,512))

    def test_short_nearly_collinear_feature_group_remains_protected_but_collapse_is_rejected(self):
        points=synthetic_points();indices=list(FEATURE_POLYGONS[3]);angle=np.arange(len(indices))*2*np.pi/len(indices)
        points[indices]=np.stack((216+17*np.cos(angle),210+.04*np.sin(angle)),axis=1)
        for scale,degrees in ((1.,0.),(.7,28.),(1.1,-32.)):
            radians=np.deg2rad(degrees);matrix=np.array([[np.cos(radians),-np.sin(radians)],[np.sin(radians),np.cos(radians)]])
            posed=(points-256)@matrix.T*scale+256
            planes,_=geometry._geometry({'faces':[{'landmarks':posed.tolist()}]},(512,512))
            self.assertTrue(np.any(planes['J']>0));self.assertTrue(np.all(planes['J'][planes['P']]==0))
        for collapsed in (np.tile([216.,210.],(len(indices),1)),np.stack((np.linspace(200,230,len(indices)),np.full(len(indices),210.)),axis=1)):
            points[indices]=collapsed
            with self.assertRaisesRegex(ValueError,'collapsed|Degenerate|collinear'):
                geometry._geometry({'faces':[{'landmarks':points.tolist()}]},(512,512))

    def test_oblique_compact_face_keeps_feature_protection_and_extreme_profile_stops(self):
        from scipy.ndimage import distance_transform_edt
        points=synthetic_points();points[:,0]=256+(points[:,0]-256)*.65
        angle=np.deg2rad(-27.);matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        posed=(points-256)@matrix.T+256
        planes,info=geometry._geometry({'faces':[{'landmarks':posed.tolist()}]},(512,512))
        box=info['box'];local=posed-box[:2];size=(box[2]-box[0],box[3]-box[1])
        raw=np.zeros(planes['P'].shape,bool)
        for indices in (*FEATURE_POLYGONS,geometry.NOSE):raw|=geometry._polygon(local[list(indices)],size)
        expected=distance_transform_edt(~raw)<=info['feature_margin']
        self.assertFalse(np.any(expected&~planes['P']));self.assertTrue(np.all(planes['J'][expected]==0))
        extreme=synthetic_points();extreme[:,0]=256+(extreme[:,0]-256)*.25
        with self.assertRaisesRegex(ValueError,'geometrically unreliable'):
            geometry._geometry({'faces':[{'landmarks':extreme.tolist()}]},(512,512))

    def test_owned_cache_content_runtime_preflight_and_current_u_are_bound(self):
        with patch.object(geometry,'_run_detector',side_effect=self.detector):
            first=self.prepare();second=self.prepare('owned-second')
        self.assertEqual(len(self.calls),1)
        one=json.loads(Path(first['path']).read_bytes());two=json.loads(Path(second['path']).read_bytes())
        self.assertEqual(one['binding'],two['binding'])
        self.assertEqual(one['binding']['base']['asset_id'],self.base['asset_id'])
        self.assertTrue(Path(two['masks']['path']).is_relative_to(self.root/'owned-second'))
        self.assertTrue(Path(two['detections']['path']).is_relative_to(self.root/'owned-second'))
        # No model/runtime is declared as a portable path+sha256 file record.
        def visit(value):
            if isinstance(value,dict):
                if {'path','sha256'}<=set(value):
                    self.assertNotIn(Path(value['path']),{self.python,self.model,self.metadata})
                for item in value.values():visit(item)
            elif isinstance(value,list):
                for item in value:visit(item)
        visit(two)
        frozen=self.preflight();self.metadata.write_text('Name: mediapipe\nVersion: 0.10.20\nChanged: yes\n',encoding='utf8')
        with patch.object(geometry,'_run_detector',side_effect=AssertionError('stale preflight launched detector')):
            with self.assertRaisesRegex(ValueError,'preflight'):self.prepare('stale',preflight=frozen)
        with patch.object(geometry,'_run_detector',side_effect=self.detector):self.prepare('new-runtime')
        self.assertEqual(len(self.calls),2)
        with self.assertRaises(ValueError):
            geometry.region_weight(first,base_appearance={**self.base,'asset_id':'e'*64},core_box=[120,120,390,410],
                selected_parts={'face':.3},**self.binding)

    def test_real_core_weight_and_face_body_union_cannot_reopen_face(self):
        with patch.object(geometry,'_run_detector',side_effect=self.detector):support=self.prepare()
        core=[120,110,400,430]
        face,evidence=geometry.region_weight(support,base_appearance=self.base,core_box=core,selected_parts={'face':.3},**self.binding)
        both,_=geometry.region_weight(support,base_appearance=self.base,core_box=core,selected_parts={'face':.3,'waist':.2},**self.binding)
        body,_=geometry.region_weight(support,base_appearance=self.base,core_box=core,selected_parts={'waist':.2},**self.binding)
        with np.load(evidence['core_masks']['path'],allow_pickle=False) as planes:
            j,p,e=(planes[k] for k in ('J','P','E'))
            self.assertTrue(np.array_equal(face,j));self.assertEqual(face.dtype,np.float32);self.assertEqual(face.shape,(320,280))
            for weight in (face,both,body):
                self.assertTrue(np.all(weight[p]==0));self.assertTrue(np.all(weight[e&(j==0)]==0))
            self.assertTrue(np.all(body[e]==0));self.assertTrue(np.all(both[e]==j[e]))
            self.assertTrue(np.any(both[~e]>0));self.assertTrue(np.any(body[~e]>0))
        self.assertEqual(evidence['hard_protection'],'central_face_and_features_exact_zero')
        self.assertTrue(evidence['geometry_only']);self.assertTrue(evidence['lower_contour_in_core'])
        with self.assertRaisesRegex(ValueError,'contains no safe'):
            geometry.region_weight(support,base_appearance=self.base,core_box=[0,0,50,50],selected_parts={'face':.3},**self.binding)
        with self.assertRaises(ValueError):
            geometry.region_weight(support,base_appearance=self.base,core_box=[0.,0,50,50],selected_parts={'face':.3},**self.binding)

    def test_partial_core_clips_real_segments_without_manufacturing_border_paths(self):
        points=[[-5,5],[5,5],[15,5],[15,15],[5,15],[5,5],[-5,5]]
        segments=geometry._clip_chain(points,[0,0,10,10])
        self.assertEqual(segments,[[[0.,.5],[.5,.5],[1.,.5]],[[.5,1.],[.5,.5],[0.,.5]]])
        self.assertEqual(geometry._clip_chain([[-4,-4],[-2,12]],[0,0,10,10]),[])
        with patch.object(geometry,'_run_detector',side_effect=self.detector):support=self.prepare()
        _,evidence=geometry.region_weight(support,base_appearance=self.base,core_box=[230,350,285,400],
            selected_parts={'face':.3},**self.binding)
        self.assertTrue(evidence['lower_contour_segments_in_core'])
        self.assertTrue(any(any(value<0 or value>1 for value in point) for point in evidence['lower_contour_in_core']))
        for segment in evidence['lower_contour_segments_in_core']:
            self.assertGreaterEqual(len(segment),2)
            self.assertTrue(np.all(np.asarray(segment)>=0));self.assertTrue(np.all(np.asarray(segment)<=1))

    def test_corrupt_cache_and_owned_planes_are_rejected_without_detector_retry(self):
        with patch.object(geometry,'_run_detector',side_effect=self.detector):support=self.prepare()
        document=json.loads(Path(support['path']).read_bytes());mask=Path(document['masks']['path'])
        original=mask.read_bytes();mask.write_bytes(original[:-1]+bytes([original[-1]^1]))
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            geometry.region_weight(support,base_appearance=self.base,core_box=[120,120,390,410],selected_parts={'face':.3},**self.binding)
        cache=next((self.root/'outputs/cache/shaping-geometry').glob('*/geometry.npz'));cache.write_bytes(b'corrupt')
        with patch.object(geometry,'_run_detector',side_effect=AssertionError('Corruption must not trigger inference')):
            with self.assertRaises(ValueError):self.prepare('corrupt-cache')
        self.assertFalse((self.root/'corrupt-cache').exists())

    def test_self_consistent_cache_and_owned_mask_rewrites_cannot_unlock_center(self):
        with patch.object(geometry,'_run_detector',side_effect=self.detector):support=self.prepare()
        def rewrite(path):
            saved=json.loads(path.read_text(encoding='utf8'));mask=Path(saved['masks']['path'])
            with np.load(mask,allow_pickle=False) as arrays:shape=arrays['J'].shape
            np.savez_compressed(mask,J=np.ones(shape,np.float32),P=np.zeros(shape,bool),
                E=np.ones(shape,bool),O=np.zeros(shape,np.float32))
            saved['masks']['sha256']=sha256_file(mask)
            saved['support_id']=identity({k:v for k,v in saved.items() if k!='support_id'})
            path.write_text(json.dumps(saved),encoding='utf8')
            return {'path':str(path),'sha256':sha256_file(path)}
        changed=rewrite(Path(support['path']))
        with self.assertRaisesRegex(ValueError,'deterministic canonical'):
            geometry.region_weight(changed,base_appearance=self.base,core_box=[120,110,400,430],selected_parts={'face':.3},**self.binding)
        cache=next((self.root/'outputs/cache/shaping-geometry').glob('*/support.json'));rewrite(cache)
        with patch.object(geometry,'_run_detector',side_effect=AssertionError('Never repair corrupt cache by inference')):
            with self.assertRaisesRegex(ValueError,'deterministic canonical'):self.prepare('forged-cache')
        self.assertEqual(len(self.calls),1);self.assertFalse((self.root/'forged-cache').exists())

    def test_self_consistent_owned_geometry_metadata_rewrite_is_rejected(self):
        with patch.object(geometry,'_run_detector',side_effect=self.detector):support=self.prepare()
        path=Path(support['path']);document=json.loads(path.read_bytes())
        document['geometry']['face_width']+=.125
        document['support_id']=identity({k:v for k,v in document.items() if k!='support_id'})
        path.write_text(json.dumps(document),encoding='utf8')
        changed={'path':str(path),'sha256':sha256_file(path)}
        with self.assertRaisesRegex(ValueError,'actual canonical detections'):
            geometry.region_weight(changed,base_appearance=self.base,core_box=[120,110,400,430],selected_parts={'face':.3},**self.binding)

    def test_cancel_failure_and_input_runtime_change_publish_no_partial_support(self):
        cancelled=[False]
        def detector(runtime,preview,output,cancel):
            self.detector(runtime,preview,output,cancel);cancelled[0]=True
        with patch.object(geometry,'_run_detector',side_effect=detector):
            with self.assertRaises(InterruptedError):self.prepare(cancel=lambda:cancelled[0])
        self.assertFalse((self.root/'owned').exists());self.assertFalse(list((self.root/'outputs/cache/shaping-geometry').glob('*/support.json')))
        cancelled[0]=False
        def changed(runtime,preview,output,cancel):
            self.detector(runtime,preview,output,cancel);self.model.write_bytes(b'changed-during-detector')
        with patch.object(geometry,'_run_detector',side_effect=changed):
            with self.assertRaises(ValueError):self.prepare('changed-runtime')
        self.assertFalse((self.root/'changed-runtime').exists())
        self.assertFalse(list((self.root/'outputs/cache/shaping-geometry').glob('*.partial-*')))

    def test_worker_timeout_and_cancellation_kill_only_owned_process_handle(self):
        class Process:
            returncode=None
            killed=0
            def poll(self):return self.returncode
            def wait(self,timeout=None):
                if self.returncode is None:raise subprocess.TimeoutExpired('fake-owned-detector',timeout)
                return self.returncode
            def kill(self):self.killed+=1;self.returncode=-1
        runtime=geometry._runtime(self.root,self.registry)
        for name,cancel,clock in (('timeout',lambda:False,[0.,181.]),('cancel',iter([False,True]).__next__,[0.])):
            with self.subTest(name=name):
                output=self.root/name;output.mkdir();process=Process()
                with patch.object(geometry,'external_python_popen',return_value=process) as spawn,patch.object(geometry.time,'monotonic',side_effect=clock):
                    with self.assertRaises((TimeoutError,InterruptedError)):
                        geometry._run_detector(runtime,self.source,output,cancel)
                self.assertEqual(process.killed,1)
                self.assertIn('--detect-only',spawn.call_args.args[0]);self.assertNotIn('--checkpoint',spawn.call_args.args[0])


if __name__=='__main__':unittest.main()
