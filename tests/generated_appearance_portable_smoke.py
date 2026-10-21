"""Typed native portable ZIP round trips, byte identity and hostile archives.

Real CPU float/PNG assets and actual encoders are used. No model or network
transport runs; the tiny source/decode/generation evidence is a synthetic fixture.
"""
from __future__ import annotations
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import warnings
import zipfile
import numpy as np
from asset_identity import sha256_file
from current_model_registry import atomic_json
from generated_appearance_render import render_candidate
from portable_project import export_project,import_project,FORMAT,APPEARANCE,CONTEXT
import photo_appearance
import generated_appearance_render_smoke as render_fixture
import photo_appearance_smoke as appearance_fixture
import user_feedback_finish_smoke as public_feedback_fixture


class PortableAppearanceTests(unittest.TestCase):
    def setUp(self):
        render_fixture.ScopedRenderTests.setUp(self)
        self.config=patch.dict(os.environ,{'COSPLAY_CONFIG_ROOT':str(self.root/'test-config')})
        self.config.start()
        self.addCleanup(self.config.stop)
        self.enterContext(patch('urllib.request.urlopen',side_effect=AssertionError('No actual network')))
    def tearDown(self):render_fixture.ScopedRenderTests.tearDown(self)

    def make_saved(self,name='portrait',size=(72,96),work='partial_composite',*,mixed=False):
        fx=render_fixture.ScopedRenderTests(methodName='runTest');fx.setUp();self.addCleanup(fx.tearDown)
        run=fx.fixture(name,size,work);run.update(root=str(fx.root),input_mode='single_source',software_version='fixture')
        from creative_task import normalize_task_intent,image_budget,INTENT_SCHEMA,PLAN_SCHEMA
        from cos_node_registry import compile_workflow
        scope=run['task_plan']['editing_scope'];request={key:scope[key] for key in ('work_type','style','target_setting','scene_target','language')}
        intent=normalize_task_intent({'schema_version':INTENT_SCHEMA,'mode':'create_reference',
            'services':{'text_vision':True,'visual_review':False}})
        run.update(task_intent=intent,execution_strategy='layered_reference_transfer')
        run['options'].update(task_intent=intent,editing_request=request,execution_strategy='layered_reference_transfer')
        run['task_plan'].update(schema_version=PLAN_SCHEMA,intent=intent,mode='create_reference',editing_request=request,
            execution_strategy='layered_reference_transfer',plan_id='a'*64,editing_autofinish={'maximum_improvements':0,'maximum_zero_image_finishes':0},
            image_budget=image_budget(intent,1,scope,autofinish={'maximum_improvements':0}),
            services={'text_vision_requests_upper_bound':2},layer_workflow=compile_workflow('create_reference',editing_scope=scope,
                editing_initialization=run['task_plan']['editing_initialization'],editing_autofinish={'maximum_improvements':0}))
        if mixed:
            scope=run['task_plan']['editing_scope']
            binding={'source_sha256':run['source']['sha256'],'scope':scope,'design_id':run['generation_design']['observed_design']['design_id']}
            previous=photo_appearance.load_asset(run['initial_source_appearance'],**binding)
            values=previous['master'].copy();del previous
            coverage=np.zeros(values.shape[:2],np.float32);coverage[5:11,6:13]=1
            values[coverage>0]+=np.float32(.035123)
            fx.source=Path(run['source']['path']);fx.source_sha=run['source']['sha256']
            generation=appearance_fixture.AppearanceTests.generation(fx,name='regional-evidence')
            asset=photo_appearance.freeze_candidate(fx.root/'regional-appearance',kind='mixed_regional',master=values,
                parent=run['initial_source_appearance'],coverage=coverage,generation=generation,**binding)
            receipt=photo_appearance.validate_asset(asset,**binding)
            run['options']['photo_appearance']=asset
            run['reference']={'path':str(Path(asset['path'])/'preview.png'),'sha256':receipt['files']['preview']['sha256']}
        result=render_candidate(fx.root,run,fx.args);fx.save(run,result)
        return fx,run,result

    def archive(self,run,source_root,**options):
        archive=self.root/(run['run_id']+str(len(list(self.root.glob('*.cosplayproj'))))+'.cosplayproj')
        export_project(source_root,run['run_id'],archive,**options)
        return archive

    def rewrite(self,original,name,*,run_change=None,package_change=None,inner_change=None,asset_change=None,extra=None):
        target=self.root/(name+'.cosplayproj')
        with zipfile.ZipFile(original) as source:
            payloads={item.filename:source.read(item.filename) for item in source.infolist()}
        package=json.loads(payloads['project.json']);run=json.loads(payloads['run.json'])
        if inner_change:
            asset_id,group=next(iter(package['photo_appearance_groups'].items()));blob_name=group['archive_asset']
            with zipfile.ZipFile(BytesIO(payloads[blob_name])) as nested:
                files={item.filename:nested.read(item.filename) for item in nested.infolist()}
            inner_change(files,asset_id,group,run)
            stream=BytesIO()
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as nested:
                for member,data in files.items():nested.writestr(member,data)
            data=stream.getvalue();payloads[blob_name]=data
            package['assets'][blob_name].update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        if run_change:run_change(run)
        if package_change:package_change(package)
        if asset_change:
            for member,entry in package['assets'].items():
                if member.endswith('.json'):
                    data=json.dumps(asset_change(json.loads(payloads[member]))).encode();payloads[member]=data
                    entry.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        payloads['project.json']=json.dumps(package).encode();payloads['run.json']=json.dumps(run).encode()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as output:
                for member,data in payloads.items():output.writestr(member,data)
                if extra:output.writestr(*extra)
        return target

    def test_typed_float_png_and_receipt_bytes_survive_source_workspace_disappearance(self):
        from export_artifact import export_source_run
        for name,size,mixed in (('portrait',(72,96),True),('wide',(120,80),False)):
            with self.subTest(name=name):
                fx,run,result=self.make_saved(name,size,'partial_composite' if mixed else 'full_composite',mixed=mixed)
                assets=[run['initial_source_appearance'],run['options']['photo_appearance']]
                original={item['asset_id']:{path.name:path.read_bytes() for path in Path(item['path']).iterdir() if path.is_file()} for item in assets}
                expected={key:sha256_file(result['outputs'][key]) for key in ('auto_draft','render_master','render_scene_master')}
                archive=self.archive(run,fx.root)
                with zipfile.ZipFile(archive) as bundle:
                    package=json.loads(bundle.read('project.json'));self.assertEqual(package['format'],FORMAT)
                    self.assertEqual(len(package['photo_appearance_groups']),2)
                    for asset_id,group in package['photo_appearance_groups'].items():
                        with zipfile.ZipFile(BytesIO(bundle.read(group['archive_asset']))) as nested:
                            self.assertTrue(all(row.compress_type==zipfile.ZIP_STORED for row in nested.infolist()))
                            for member in nested.namelist():self.assertEqual(nested.read(member),original[asset_id][member])
                imported=import_project(self.root/('target-'+name),archive,run_id='reopened-'+name);target=Path(imported['manifest']).parents[3]
                reopened=imported['run'];before=copy.deepcopy(reopened['task_plan'])
                for item in (reopened['initial_source_appearance'],reopened['options']['photo_appearance']):
                    self.assertTrue(Path(item['path']).is_relative_to(target))
                    for member,data in original[item['asset_id']].items():self.assertEqual((Path(item['path'])/member).read_bytes(),data)
                hidden=fx.root.with_name(fx.root.name+'-unavailable');fx.root.rename(hidden)
                try:
                    with patch.dict(os.environ,{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'relocated-hash-cache')}), \
                         patch('raw_support.open_rgb',side_effect=AssertionError('Decoded original')), \
                         patch('raw_support.open_linear',side_effect=AssertionError('Decoded original')):
                        replay=render_candidate(target,reopened,self.args)
                        for key,digest in expected.items():self.assertEqual(sha256_file(replay['outputs'][key]),digest)
                        reopened.update(auto_draft=replay,render=replay['metadata'],review=replay['review'],artifacts=replay['outputs'])
                        atomic_json(Path(imported['manifest']),reopened)
                        for format,key in (('PNG','auto_draft'),('TIFF16','render_master'),('TIFF32_LINEAR','render_scene_master')):
                            exported=export_source_run(target,reopened['run_id'],directory=str(target/'exports'),format=format)
                            self.assertEqual(sha256_file(exported['path']),expected[key]);self.assertFalse(exported['metadata']['approved'])
                        if name=='portrait':
                            from PySide6.QtWidgets import QApplication
                            from design_reconstruction_ui_smoke import Window,Pool
                            app=QApplication.instance() or QApplication([])
                            window=Window(target,'http://127.0.0.1:1');window.pool=Pool()
                            try:
                                window.load_source_run(reopened);app.processEvents()
                                self.assertEqual(window.previews['final'].path,Path(replay['outputs']['auto_draft']))
                                self.assertTrue(window.source_generated_appearance)
                                self.assertFalse(window.local_editor_tabs.isEnabled())
                                self.assertTrue(window.simple_workspace.export_button.isEnabled())
                            finally:window.session_timer.stop();window.close();app.processEvents()
                finally:hidden.rename(fx.root)
                self.assertEqual(reopened['task_plan'],before)

    def test_typed_package_can_omit_original_bytes_but_keeps_both_native_assets(self):
        from export_artifact import export_source_run
        fx,run,result=self.make_saved(mixed=True)
        archive=self.archive(run,fx.root,include_raw=False,include_design=False)
        target=self.root/'without-raw';imported=import_project(target,archive);reopened=imported['run']
        self.assertFalse(Path(reopened['source']['path']).exists())
        self.assertFalse(imported['project_identity_created'])
        hidden=fx.root.with_name(fx.root.name+'-unavailable');fx.root.rename(hidden)
        try:
            with patch.dict(os.environ,{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'omit-hash-cache')}), \
                 patch('raw_support.open_linear',side_effect=AssertionError('Decoded original')):
                saved=render_candidate(target,reopened,self.args)
                self.assertEqual(sha256_file(saved['outputs']['auto_draft']),sha256_file(Path(reopened['options']['photo_appearance']['path'])/'preview.png'))
                exported=export_source_run(target,reopened['run_id'],directory=str(target/'exports'),format='TIFF32_LINEAR')
                self.assertEqual(sha256_file(exported['path']),sha256_file(saved['outputs']['render_scene_master']))
        finally:hidden.rename(fx.root)

    def test_external_session_descriptors_are_packed_as_immutable_groups(self):
        fx,run,_=self.make_saved(mixed=True)
        old=run['options']['photo_appearance'];duplicate=fx.root/'parent-owned-copy';shutil.copytree(old['path'],duplicate)
        descriptor={**old,'path':str(duplicate)}
        history=fx.root/'external-session.json';history.write_text(json.dumps({'candidate':{'appearance_asset':descriptor}}),encoding='utf8')
        run['artifacts']['photo_edit_session']=str(history);atomic_json(fx.root/'outputs/runs'/run['run_id']/'manifest.json',run)
        archive=self.archive(run,fx.root);imported=import_project(self.root/'history-target',archive)
        record=json.loads(Path(imported['run']['artifacts']['photo_edit_session']).read_text(encoding='utf8'))
        asset=record['candidate']['appearance_asset']
        self.assertEqual(asset['asset_id'],old['asset_id']);self.assertNotEqual(asset['path'],str(duplicate))
        self.assertEqual((Path(asset['path'])/'receipt.json').read_bytes(),(duplicate/'receipt.json').read_bytes())

    def test_public_session_history_survives_portable_relocation_and_feedback_start(self):
        from backend_server import Backend
        from photo_editing import identity
        from photo_edit_session import snapshot
        fx=public_feedback_fixture.UserFeedbackFinishTests(methodName='runTest')
        self.addCleanup(fx.doCleanups);fx.setUp()
        parent=fx.parent;path=Path(parent['photo_edit_session']['path'])
        original=json.loads(path.read_text('utf8'));original_bytes=path.read_bytes()
        archive=fx.root.parent/'public-session.cosplayproj'
        export_project(fx.root,parent['run_id'],archive)
        self.assertEqual(path.read_bytes(),original_bytes,'Packaging must not alter the saved parent')
        omitted=fx.root.parent/'public-without-raw.cosplayproj'
        export_project(fx.root,parent['run_id'],omitted,include_raw=False)
        with zipfile.ZipFile(omitted) as bundle:
            package=json.loads(bundle.read('project.json'))
            self.assertNotIn(parent['source']['sha256'],{entry['sha256'] for entry in package['assets'].values()},
                'Omitting RAW must apply to session evidence as well as ordinary project assets')
        def unsafe_member(package):
            group=next(iter(package['photo_session_groups'].values()))
            group['files']['../outside']=next(iter(group['files'].values()))
        bad=self.rewrite(archive,'unsafe-session',package_change=unsafe_member)
        unsafe_target=fx.root.parent/'unsafe-ported'
        with self.assertRaisesRegex(ValueError,'session member'):import_project(unsafe_target,bad)
        self.assertFalse((unsafe_target/'outputs/runs'/parent['run_id']).exists())
        def change_context(value):
            if isinstance(value,dict) and value.get('schema')=='cosluma_photo_edit_session.v1':
                value['attempts'][0]['context']['unexpected']='tampered history'
            return value
        bad=self.rewrite(archive,'changed-session',asset_change=change_context)
        with self.assertRaisesRegex(ValueError,'historical context'):import_project(unsafe_target,bad)
        self.assertFalse((unsafe_target/'outputs/runs'/parent['run_id']).exists())
        target=fx.root.parent/'ported';reopened=import_project(target,archive)['run']
        imported_path=Path(reopened['photo_edit_session']['path'])
        state=json.loads(imported_path.read_text('utf8'))
        self.assertEqual({k:v for k,v in state.items() if k!='evidence'},
                         {k:v for k,v in original.items() if k!='evidence'})
        self.assertEqual(state['session_id'],identity(state['binding']))
        self.assertTrue(all(row['context_sha256']==identity(row['context']) for row in state['attempts']))
        self.assertEqual(reopened['photo_edit_session']['independent_evaluation'],parent['photo_edit_session']['independent_evaluation'])
        registry=target/'registry.json';shutil.copyfile(fx.fx.registry,registry)
        hidden=fx.root.with_name('original-unavailable');fx.root.rename(hidden)
        try:
            with patch.dict(os.environ,{'COSPLAY_MODEL_REGISTRY':str(registry)}):
                snapshot(imported_path.parent)
                from cos_workflow_graph import build_graph
                graph=build_graph(reopened)
                self.assertTrue(graph['edited_output'])
                self.assertEqual(next(row['status'] for row in graph['nodes'] if row['id']=='selected_edit_tool'),'complete')
                body=fx.request();body.update(source=reopened['source']['path'],reference=reopened['artifacts']['auto_draft'],
                    sample=reopened['sample'],parent_run_id=reopened['run_id'],model_registry=str(registry))
                backend=Backend(target,embedded=True);before=len(fx.fx.image_requests),len(fx.fx.vision_requests)
                checked=backend.task_plan(body)
                self.assertTrue(checked['plan']['ready'],checked['plan'].get('blocking_reasons'))
                self.assertEqual(before,(len(fx.fx.image_requests),len(fx.fx.vision_requests)))
                result=backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})
                self.assertTrue(result.get('ok'),result.get('stderr'))
                self.assertEqual(result['run']['result_publication']['state'],'committed')
                self.assertEqual(len(fx.fx.image_requests)-before[0],1)
                self.assertEqual(len(fx.fx.vision_requests)-before[1],2)
                self.assertFalse(result['run']['photo_edit_session']['human_approved'])
                child_state=snapshot(Path(result['run']['photo_edit_session']['path']).parent)
                self.assertEqual(child_state['binding']['user_intent']['user_feedback'],fx.feedback)
                self.assertEqual(child_state['candidates'][0]['review'],fx.parent_review)
                self.assertEqual(child_state['candidates'][0]['review_objective'],'baseline')
        finally:hidden.rename(fx.root)

    def test_modified_group_members_and_rehashed_receipts_are_rejected_atomically(self):
        fx,run,_=self.make_saved(mixed=True);archive=self.archive(run,fx.root)
        def corrupt_member(files,*args):
            data=files['master.npy'];files['master.npy']=data[:-1]+bytes([data[-1]^1])
        def corrupt_receipt(files,asset_id,group,run):
            record=json.loads(files['receipt.json']);record['physical_relighting_verified']=True
            data=json.dumps(record).encode();old=group['receipt_sha256'];new=hashlib.sha256(data).hexdigest()
            files['receipt.json']=data;group['receipt_sha256']=new
            def replace(value):
                if isinstance(value,dict):return {key:(new if key=='receipt_sha256' and item==old else replace(item)) for key,item in value.items()}
                if isinstance(value,list):return [replace(item) for item in value]
                return value
            changed=replace(run);run.clear();run.update(changed)
        def missing(files,*args):files.pop('preview.png')
        def traversal(files,*args):files['../escape.npy']=b'forbidden'
        for name,change in (('member',corrupt_member),('receipt',corrupt_receipt),('missing',missing),('nested-path',traversal)):
            with self.subTest(name=name):
                changed=self.rewrite(archive,name,inner_change=change);target=self.root/('reject-'+name)
                with self.assertRaises((ValueError,zipfile.BadZipFile)):import_project(target,changed)
                self.assertFalse((target/'outputs/runs'/run['run_id']).exists())
                self.assertFalse((target/'project.json').exists());self.assertFalse((target/'escape.npy').exists())

    def test_archive_duplicates_undeclared_paths_and_locator_traversal_are_rejected(self):
        fx,run,_=self.make_saved();archive=self.archive(run,fx.root)
        attacks=[('extra',{'extra':('../escape.txt',b'forbidden')}),('duplicate',{'extra':('run.json',b'{}')}),
            ('context',{'run_change':lambda value:value.update(extra=CONTEXT+'../outside')}),
            ('external',{'run_change':lambda value:value.update(extra='cosplay-external://../outside')}),
            ('group-member',{'run_change':lambda value:value.update(extra=APPEARANCE+value['options']['photo_appearance']['asset_id']+'/../outside')})]
        for name,changes in attacks:
            with self.subTest(name=name):
                changed=self.rewrite(archive,name,**changes);target=self.root/('unsafe-'+name)
                with self.assertRaises(ValueError):import_project(target,changed)
                self.assertFalse((target/'outputs/runs'/run['run_id']).exists())
        with self.assertRaises(ValueError):import_project(self.root/'unsafe-id',archive,run_id='../outside')
        self.assertFalse((self.root/'outside').exists())

    def test_retired_packages_and_untyped_creation_are_rejected_without_compatibility(self):
        fx,run,_=self.make_saved();archive=self.archive(run,fx.root)
        changed=self.rewrite(archive,'retired-format',package_change=lambda value:value.update(format='cosplay.portable-project.v2'))
        with self.assertRaisesRegex(ValueError,'Unsupported portable project format'):import_project(self.root/'old-format',changed)
        changed=self.rewrite(archive,'untyped-create',run_change=lambda value:value['options'].pop('photo_appearance'))
        with self.assertRaisesRegex(ValueError,'current typed appearance'):import_project(self.root/'old-create',changed)

    def test_rehashed_invalid_float_master_rolls_back_without_a_windows_mapping_leak(self):
        from photo_editing import identity
        fx,run,_=self.make_saved();archive=self.archive(run,fx.root);changed_ids={};replacements={}
        def invalid_float(files,asset_id,group,run):
            record=json.loads(files['receipt.json']);values=np.load(BytesIO(files['master.npy']),allow_pickle=False)
            values[0,0,0]=np.nan;stream=BytesIO();np.save(stream,values,allow_pickle=False)
            files['master.npy']=stream.getvalue();record['files']['master']['sha256']=hashlib.sha256(files['master.npy']).hexdigest()
            record.pop('asset_id');record['asset_id']=identity(record);data=json.dumps(record).encode()
            old_sha=group['receipt_sha256'];group['receipt_sha256']=hashlib.sha256(data).hexdigest();files['receipt.json']=data
            changed_ids[asset_id]=record['asset_id']
            replacements.update({asset_id:record['asset_id'],old_sha:group['receipt_sha256']})
            def relocate(value):
                if isinstance(value,str):return value.replace(asset_id,record['asset_id']).replace(old_sha,group['receipt_sha256'])
                if isinstance(value,list):return [relocate(item) for item in value]
                if isinstance(value,dict):return {key:relocate(item) for key,item in value.items()}
                return value
            replacement=relocate(run);run.clear();run.update(replacement)
        def update_index(package):
            for old,new in changed_ids.items():package['photo_appearance_groups'][new]=package['photo_appearance_groups'].pop(old)
        def update_assets(value):
            if isinstance(value,str):
                for old,new in replacements.items():value=value.replace(old,new)
                return value
            if isinstance(value,list):return [update_assets(item) for item in value]
            if isinstance(value,dict):return {key:update_assets(item) for key,item in value.items()}
            return value
        changed=self.rewrite(archive,'invalid-float',inner_change=invalid_float,package_change=update_index,asset_change=update_assets)
        target=self.root/'bad-float'
        with self.assertRaisesRegex(ValueError,'finite normalized'):import_project(target,changed)
        self.assertFalse((target/'outputs/runs'/run['run_id']).exists())

    def test_export_failure_removes_archive_and_group_staging(self):
        fx,run,_=self.make_saved();cache=fx.root/'outputs/cache';cache.mkdir(parents=True,exist_ok=True)
        before=set(cache.glob('portable-appearance-*'));archive=self.root/'failed.cosplayproj'
        actual=photo_appearance.validate_asset;created=[]
        def fail_after_first(*args,**kwargs):
            if list(cache.glob('portable-appearance-*/*.zip')):
                created.append(True)
                raise ValueError('Synthetic packaging validation failure')
            return actual(*args,**kwargs)
        with patch('photo_appearance.validate_asset',side_effect=fail_after_first),self.assertRaises(ValueError):export_project(fx.root,run['run_id'],archive)
        self.assertFalse(archive.exists());self.assertEqual(set(cache.glob('portable-appearance-*')),before)
        self.assertTrue(any(created),'Failure must occur after a real opaque group was staged')
        normal=zipfile.ZipFile.write
        def outer_failure(bundle,*args,**kwargs):
            if Path(bundle.filename)==archive:raise OSError('Synthetic archive write failure')
            return normal(bundle,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'write',outer_failure),self.assertRaises(OSError):export_project(fx.root,run['run_id'],archive)
        self.assertFalse(archive.exists());self.assertEqual(set(cache.glob('portable-appearance-*')),before)


if __name__=='__main__':unittest.main()
