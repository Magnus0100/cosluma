"""Real geometric typed save/copy/export/portable consumers, with CPU fixtures.

The canonical detector boundary is synthetic. No model, image generation,
network, user project or historical receipt is used or modified.
"""
from __future__ import annotations

import copy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import tifffile
from PIL import Image

from asset_identity import sha256_file
import appearance_assets as assets
import geometric_appearance
from generated_appearance_render import render_candidate,saved_display_transform
from portable_project import export_project,import_project
from export_artifact import export_source_run
from native_detail import saved_native_detail
from photo_editing import identity
from photo_request_budget import manual_target_key
from photo_shaping import SCHEMA,STATE_SCHEMA
import photo_shaping_geometry_smoke as geometry_fixture
import generated_appearance_portable_smoke as portable_fixture
import photo_appearance as native


class GeometricConsumerTests(unittest.TestCase):
    def setUp(self):
        self.helper=portable_fixture.PortableAppearanceTests(methodName='runTest');self.helper.setUp()
        def cleanup():
            self.helper.doCleanups();self.helper.tearDown()
        self.addCleanup(cleanup)
        self.fx,self.parent,_=self.helper.make_saved('geo-U',size=(512,512))
        self.root=self.fx.root;scope=self.parent['task_plan']['editing_scope']
        self.binding={'source_sha256':self.parent['source']['sha256'],'scope':scope,
            'design_id':self.parent['generation_design']['observed_design']['design_id']}
        baseline=assets.load_asset(self.parent['initial_source_appearance'],**self.binding)
        values=baseline['master'].copy();yy,xx=np.indices(values.shape[:2],dtype=np.float32)
        coverage=np.zeros(values.shape[:2],np.float32)
        coverage[80:440,80:400]=.15+.65*xx[80:440,80:400]/512
        values+=coverage[...,None]*np.stack((xx/512,yy/1024,(xx+yy)/1536),-1)*np.float32(.08)
        generation=assets.validate_asset(self.parent['options']['photo_appearance'],**self.binding)['generation']
        old=self.parent['options']['photo_appearance']
        self.u=native.freeze_candidate(self.root/'native-U',kind='mixed_regional',master=values,
            parent=self.parent['initial_source_appearance'],coverage=coverage,generation={
                **{role:{'path':str(Path(old['path'])/generation[role]['file']),'sha256':generation[role]['sha256']}
                    for role in ('image','provider_receipt')},'core_box':[60,60,460,460],'context_box':[0,0,512,512]},**self.binding)
        self.u_receipt=assets.validate_asset(self.u,**self.binding)
        self.parent['options']['photo_appearance']=self.u
        self.parent['reference']={'path':str(Path(self.u['path'])/'preview.png'),
            'sha256':self.u_receipt['files']['preview']['sha256']}
        self.parent['result_publication']={'state':'committed','scope':'checked_production','run_id':self.parent['run_id']}
        self.fx.save(self.parent,render_candidate(self.root,self.parent,self.fx.args))
        self.geometry=geometry_fixture.GeometryTests(methodName='runTest');self.geometry.setUp();self.addCleanup(self.geometry.doCleanups)
        self.geometry.base=self.u;self.geometry.binding=self.binding;self.geometry.scope=scope
        self.geometry.design=self.binding['design_id'];self.geometry.source=Path(self.parent['source']['path'])
        self.geometry.source_sha=self.parent['source']['sha256']
        with patch('photo_shaping_geometry._run_detector',side_effect=self.geometry.detector):
            self.support=self.geometry.prepare('geo-current-support')
        self.current=geometric_appearance.create_asset(self.root/'geometric-current',base_appearance=self.u,
            support_descriptor=self.support,amount=.3,**self.binding)
        request={'action':'apply','parts':{'face':.3}}
        auth={'schema':SCHEMA,'parent_run_id':self.parent['run_id'],
            'parent_manifest_sha256':sha256_file(self.root/'outputs/runs'/self.parent['run_id']/'manifest.json'),
            'parent_appearance':copy.deepcopy(self.u),'base_appearance':copy.deepcopy(self.u),
            'source_appearance':copy.deepcopy(self.parent['initial_source_appearance']),
            'source_sha256':self.parent['source']['sha256'],'parent_image_sha256':self.u_receipt['files']['preview']['sha256'],
            'base_image_sha256':self.u_receipt['files']['preview']['sha256'],
            'base_master_sha256':self.u_receipt['files']['master']['sha256'],
            'scope_id':scope['scope_id'],'design_id':self.binding['design_id'],
            'request':request,'manual_provider_target':None}
        auth['manual_target']=manual_target_key(source_sha256=auth['source_sha256'],base_sha256=auth['base_image_sha256'],
            scope_id=scope['scope_id'],design_id=self.binding['design_id'],shaping_request=request)
        auth['authorization_id']=identity(auth)
        self.run=copy.deepcopy(self.parent);self.run.update(run_id='geo-A',parent_run_id=self.parent['run_id'],revision=2)
        self.run['result_publication']['run_id']='geo-A'
        self.run['task_intent']['mode']='existing_reference';self.run['options']['task_intent']['mode']='existing_reference'
        self.run['task_plan'].update(mode='existing_reference',shaping_request=request,shaping_authorization=auth,
            inputs={'source':copy.deepcopy(self.run['source']),'reference':copy.deepcopy(self.run['reference'])})
        self.run['task_plan']['intent']['mode']='existing_reference'
        for key in ('editing_initialization','editing_autofinish'):self.run['task_plan'].pop(key,None)
        self.run['options'].update(photo_appearance=self.current,preserve_project_selection=True,
            shaping_state={'schema':STATE_SCHEMA,'base_appearance':copy.deepcopy(self.u),'enabled':True,
                'request':request,'authorization':auth,'selected_asset_id':self.current['asset_id']})
        self.run['reference']['photo_appearance']=copy.deepcopy(self.u)
        self.fx.save(self.run,render_candidate(self.root,self.run,self.fx.args))

    def test_native_master_and_transported_G_are_saved_exported_and_viewed_without_repreparation(self):
        loaded=assets.load_asset(self.current,**self.binding)
        np.testing.assert_array_equal(tifffile.imread(self.run['artifacts']['render_scene_master']),loaded['master'])
        self.assertEqual(sha256_file(self.run['artifacts']['auto_draft']),sha256_file(loaded['preview_path']))
        original=assets.load_asset(self.u,**self.binding)
        self.assertTrue(np.any(loaded['generated_coverage']!=original['generated_coverage']))
        self.assertTrue(self.run['render']['operator']['geometric_pixels_replayed_without_redeformation'])
        self.assertFalse(self.run['render']['render_precision']['generated_resolution']['generation_this_operation'])
        with patch('geometric_appearance.create_asset',side_effect=AssertionError('Replay authored a new deformation')), \
             patch('photo_shaping_geometry._runtime',side_effect=AssertionError('Replay inspected runtime')), \
             patch('photo_shaping_geometry._run_detector',side_effect=AssertionError('Replay detected')), \
             patch('raw_support.open_linear',side_effect=AssertionError('Replay decoded RAW')):
            repeated=render_candidate(self.root,self.run,self.fx.args)
            self.assertEqual(sha256_file(repeated['outputs']['auto_draft']),sha256_file(loaded['preview_path']))
            for format,artifact in (('PNG','auto_draft'),('TIFF16','render_master'),('TIFF32_LINEAR','render_scene_master')):
                exported=export_source_run(self.root,self.run['run_id'],directory=str(self.root/'exports'),format=format)
                self.assertEqual(sha256_file(exported['path']),sha256_file(self.run['artifacts'][artifact]))
                self.assertFalse(exported['metadata']['approved'])
            detail=saved_native_detail(self.root,self.run['run_id'],{'center':[.5,.5],'edge':128,'include_source':True})
            self.assertFalse(detail['raw_decoded']);self.assertFalse(detail['resampled'])
            self.assertTrue(detail['saved_version_only']);self.assertTrue(Path(detail['source_preview_path']).is_file())
        altered=copy.deepcopy(self.run);altered['render']['operator']['display_transform']['method']='other'
        with self.assertRaisesRegex(ValueError,'display encoding'):saved_display_transform(self.root,altered)

    def test_copy_and_portable_own_U_map_support_and_float_after_originals_disappear(self):
        from single_source_workflow import copy_source_run
        copied=copy_source_run(self.root,self.run['run_id'])
        current=copied['options']['photo_appearance'];closure=assets.declared_files(current,**self.binding)
        self.assertIn('parent/receipt.json',closure);self.assertIn('mapping.npz',closure)
        for name,digest in closure.items():self.assertEqual(sha256_file(Path(current['path'])/name),digest)
        archive=self.helper.root/'geometry.cosplayproj';export_project(self.root,self.run['run_id'],archive)
        destination=self.helper.root/'geometry-imported';imported=import_project(destination,archive)['run']
        hidden=self.root.with_name(self.root.name+'-hidden');self.root.rename(hidden)
        try:
            for path in (self.geometry.model,self.geometry.python,self.geometry.metadata):path.unlink()
            with patch('photo_shaping_geometry._runtime',side_effect=AssertionError('Imported replay inspected runtime')), \
                 patch('photo_shaping_geometry._run_detector',side_effect=AssertionError('Imported replay detected')), \
                 patch('raw_support.open_linear',side_effect=AssertionError('Imported replay decoded RAW')):
                selected=assets.load_asset(imported['options']['photo_appearance'],**self.binding)
                self.assertEqual(selected['receipt']['kind'],'geometric_transform')
                self.assertTrue(Path(imported['options']['shaping_state']['base_appearance']['path']).is_relative_to(destination))
                exported=export_source_run(destination,imported['run_id'],directory=str(destination/'exports'),format='PNG')
                self.assertEqual(sha256_file(exported['path']),selected['receipt']['files']['preview']['sha256'])
                detail=saved_native_detail(destination,imported['run_id'],{'center':[.5,.5],'include_source':True})
                self.assertTrue(Path(detail['preview_path']).is_file())
        finally:hidden.rename(self.root)

    def test_missing_nested_parent_or_mapping_and_unknown_schema_fail_atomically(self):
        archive=self.helper.root/'geometry-negative.cosplayproj';export_project(self.root,self.run['run_id'],archive)
        with zipfile.ZipFile(archive) as source:payloads={info.filename:source.read(info.filename)for info in source.infolist()}
        package=json.loads(payloads['project.json']);entry=package['photo_appearance_groups'][self.current['asset_id']]
        member=entry['archive_asset']
        with zipfile.ZipFile(BytesIO(payloads[member])) as packed:original={n:packed.read(n)for n in packed.namelist()}
        for index,missing in enumerate(('parent/master.npy','mapping.npz')):
            data=dict(original);data.pop(missing)
            stream=BytesIO()
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as packed:
                for name,value in data.items():packed.writestr(name,value)
            changed=dict(payloads);changed[member]=stream.getvalue();manifest=copy.deepcopy(package)
            manifest['assets'][member].update(sha256=hashlib.sha256(changed[member]).hexdigest(),bytes=len(changed[member]))
            changed['project.json']=json.dumps(manifest).encode();bad=self.helper.root/f'geometry-missing-{index}.cosplayproj'
            with zipfile.ZipFile(bad,'w',compression=zipfile.ZIP_STORED) as packed:
                for name,value in changed.items():packed.writestr(name,value)
            destination=self.helper.root/f'failed-geometry-{index}'
            with self.assertRaises(ValueError):import_project(destination,bad)
            self.assertFalse((destination/'outputs/runs'/self.run['run_id']).exists())
        unknown=self.helper.root/'unknown-current-type';unknown.mkdir()
        receipt={'schema':'retired_or_unknown.v99','kind':'geometric_transform','asset_id':'a'*64}
        path=unknown/'receipt.json';path.write_text(json.dumps(receipt),encoding='utf8')
        with self.assertRaises(ValueError):assets.validate_asset({'path':str(unknown),'asset_id':'a'*64,
            'receipt_sha256':sha256_file(path)},**self.binding)

    def test_hash_archived_receipt_stays_opaque_without_hiding_missing_typed_closure(self):
        original=(Path(self.u['path'])/'receipt.json').read_bytes()
        archived=self.root/'outputs/runs'/self.run['run_id']/'saved-session-evidence/files'/(self.u['receipt_sha256']+'.json')
        archived.parent.mkdir(parents=True);archived.write_bytes(original)
        self.assertFalse((archived.parent/'receipt.json').exists())
        archive=self.helper.root/'hash-archived-receipt.cosplayproj'
        export_project(self.root,self.run['run_id'],archive)
        with zipfile.ZipFile(archive) as packed:
            package=json.loads(packed.read('project.json'))
            self.assertIn(self.current['asset_id'],package['photo_appearance_groups'])
            self.assertIn(self.u['asset_id'],package['photo_appearance_groups'])
            members=[name for name,value in package['assets'].items()
                if value.get('opaque_session_evidence') and packed.read(name)==original]
            self.assertTrue(members,'Archived receipt must retain its exact original evidence bytes')
        imported=import_project(self.helper.root/'hash-archived-import',archive)['run']
        assets.validate_asset(imported['options']['photo_appearance'],**self.binding)
        # An actual typed receipt still registers its complete owned group;
        # this rule must not turn missing geometry/U files into ignored JSON.
        missing=Path(self.current['path'])/'parent/master.npy';data=missing.read_bytes();missing.unlink()
        failed=self.helper.root/'missing-real-typed-group.cosplayproj'
        try:
            with self.assertRaises((OSError,ValueError)):export_project(self.root,self.run['run_id'],failed)
            self.assertFalse(failed.exists())
        finally:missing.write_bytes(data)


if __name__=='__main__':unittest.main()
