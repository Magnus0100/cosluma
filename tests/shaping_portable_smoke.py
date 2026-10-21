"""Real typed-pixel shaping ownership/portable checks with synthetic model evidence.

These checks do not claim a real model execution or natural shaping quality.
"""
from __future__ import annotations

import argparse
import copy
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import tifffile

from asset_identity import sha256_file
from generated_appearance_render import render_candidate, validate_run
from photo_editing import identity
from photo_request_budget import manual_target_key
from photo_shaping import STATE_SCHEMA, SCHEMA, validate_state
from portable_project import import_project
import photo_appearance as appearance
import generated_appearance_portable_smoke as portable_fixture
import photo_appearance_smoke as appearance_fixture


class ShapingPortableTests(unittest.TestCase):
    setUp=portable_fixture.PortableAppearanceTests.setUp
    tearDown=portable_fixture.PortableAppearanceTests.tearDown
    make_saved=portable_fixture.PortableAppearanceTests.make_saved
    archive=portable_fixture.PortableAppearanceTests.archive
    rewrite=portable_fixture.PortableAppearanceTests.rewrite

    def shaped(self):
        fx,run,_=self.make_saved('shaping',mixed=True)
        base=copy.deepcopy(run['options']['photo_appearance']);scope=run['task_plan']['editing_scope']
        design=run['generation_design']['observed_design']['design_id']
        binding={'source_sha256':run['source']['sha256'],'scope':scope,'design_id':design}
        old=appearance.load_asset(base,**binding);values=old['master'].copy()
        coverage=np.zeros(values.shape[:2],np.float32);coverage[7:10,8:12]=.5
        values[coverage>0]+=np.float32(.021123)
        del old
        generation=appearance_fixture.AppearanceTests.generation(fx,name='shaping-evidence')
        selected=appearance.freeze_candidate(fx.root/'new-shape',kind='mixed_regional',master=values,
            parent=base,coverage=coverage,generation=generation,**binding)
        receipt=appearance.validate_asset(base,**binding)
        request={'action':'apply','parts':{'waist':.25}}
        auth={'schema':SCHEMA,'parent_run_id':run['run_id'],'parent_manifest_sha256':sha256_file(
                fx.root/'outputs/runs'/run['run_id']/'manifest.json'),
            'parent_appearance':copy.deepcopy(base),'base_appearance':copy.deepcopy(base),
            'source_appearance':copy.deepcopy(run['initial_source_appearance']),
            'source_sha256':run['source']['sha256'],'parent_image_sha256':receipt['files']['preview']['sha256'],
            'base_image_sha256':receipt['files']['preview']['sha256'],'base_master_sha256':receipt['files']['master']['sha256'],
            'scope_id':scope['scope_id'],'design_id':design,'request':request,'manual_provider_target':None}
        auth['manual_target']=manual_target_key(source_sha256=run['source']['sha256'],
            base_sha256=auth['base_image_sha256'],scope_id=scope['scope_id'],design_id=design,
            provider_target=None,shaping_request=request)
        auth['authorization_id']=identity(auth)
        state={'schema':STATE_SCHEMA,'base_appearance':copy.deepcopy(base),'enabled':True,'request':request,
            'authorization':auth,'selected_asset_id':selected['asset_id']}
        run['run_id']='shaped';run['revision']=2
        run['task_plan'].update(mode='existing_reference',shaping_request=request,shaping_authorization=auth)
        run['task_plan'].pop('editing_initialization',None)
        run['task_plan']['inputs']={'source':copy.deepcopy(run['source']),'reference':copy.deepcopy(run['reference'])}
        run['task_intent']['mode']='existing_reference'
        run['options'].update(photo_appearance=selected,shaping_state=state)
        run['reference']['photo_appearance']=copy.deepcopy(base)
        result=render_candidate(fx.root,run,fx.args);fx.save(run,result)
        return fx,run,binding

    def test_default_is_off_and_graph_models_apply_and_exact_restore(self):
        from cos_node_registry import compile_workflow
        from scoped_workflow import build_scoped_graph
        fx,run,binding=self.shaped()
        self.assertNotIn('shaping_state',json.loads((fx.root/'outputs/runs/shaping/manifest.json').read_bytes())['options'])
        scope=binding['scope'];request=run['task_plan']['shaping_request']
        graph=compile_workflow('existing_reference',editing_scope=scope,visual_review=True,shaping_request=request)
        self.assertEqual(graph['request_limits'],{'image_submissions':1,'vision_requests':3})
        for left,right in zip(('shaping_inputs','baseline_review','decide_edit','selected_edit_tool','independent_recheck'),
                              ('baseline_review','decide_edit','selected_edit_tool','independent_recheck','select_best')):
            self.assertLess(graph['order'].index(left),graph['order'].index(right))
        restore=compile_workflow('existing_reference',editing_scope=scope,shaping_request={'action':'restore'})
        self.assertEqual(restore['request_limits'],{'image_submissions':0,'vision_requests':0})
        self.assertEqual(restore['order'],['shaping_inputs','restore_baseline','save_appearance','export'])
        displayed=build_scoped_graph(run,scope,True)
        self.assertEqual(displayed['display_basis'],'owned_shaping_baseline_and_current_session_facts')
        self.assertNotIn('observe_design',[node['id'] for node in displayed['nodes']])

    def test_unadopted_history_never_becomes_startup_and_explicit_adoption_persists(self):
        import workbench_store as store
        from saved_result_selection_smoke import write_saved
        root=self.root/'startup-selection'
        write_saved(root,'before-shaping','2026-10-20T01:00:00+00:00')
        before=store.saved_result_state(root)
        self.assertEqual(before['latest_saved_run_id'],'before-shaping')
        for number in range(22):
            identifier=f'shaping-candidate-{number:02}'
            request={'action':'restore'} if number%2 else {'action':'apply','parts':{'face':.25}}
            run=write_saved(root,identifier,f'2026-10-20T02:{number:02}:00+00:00',
                task_plan={'shaping_request':request},options={'preserve_project_selection':True})
            self.assertFalse(store.record_recent(root,sample=run['sample'],run_id=identifier)['recorded'])
            self.assertEqual(store.select_saved_result(root,identifier),before)
        self.assertEqual(store.saved_result_state(root),before)
        self.assertIn('shaping-candidate-21',[row['run_id'] for row in store.list_recent(root)])
        selected=store.select_saved_result(root,'shaping-candidate-21',explicit=True)
        self.assertEqual(selected['selected_saved_run_id'],'shaping-candidate-21')
        self.assertEqual(selected['latest_saved_run_id'],'shaping-candidate-21')
        self.assertTrue(selected['selection_explicit'])
        self.assertEqual(store.saved_result_state(root),selected)
        self.assertTrue(store.record_recent(root,sample='portrait',run_id='shaping-candidate-21')['recorded'])

    def test_copy_owns_u_preserves_authorization_and_current_preview_is_a(self):
        import single_source_workflow as workflow
        fx,run,binding=self.shaped();auth=copy.deepcopy(run['task_plan']['shaping_authorization'])
        for snapshot in (False,True):
            with self.subTest(snapshot=snapshot):
                path,child=workflow._source_revision(fx.root,run['run_id'],snapshot=snapshot)
                state=child['options']['shaping_state']
                validate_state(state,current=child['options']['photo_appearance'],**binding)
                self.assertTrue(Path(state['base_appearance']['path']).is_relative_to(path.parent))
                self.assertEqual(state['authorization'],auth)
                self.assertEqual(child['task_plan']['shaping_authorization'],auth)
                self.assertEqual(child['reference']['sha256'],auth['base_image_sha256'])
                _,_,_,current_preview=validate_run(fx.root,child)
                current_receipt=appearance.validate_asset(child['options']['photo_appearance'],**binding)
                self.assertEqual(sha256_file(current_preview),current_receipt['files']['preview']['sha256'])
                self.assertNotEqual(sha256_file(current_preview),auth['base_image_sha256'])
                self.assertIsNone(child['review']['reviewer'])

    def test_portable_owned_u_and_current_survive_entire_original_workspace_loss(self):
        fx,run,binding=self.shaped();auth=copy.deepcopy(run['task_plan']['shaping_authorization'])
        archive=self.archive(run,fx.root)
        expected={}
        for role,descriptor in (('base',run['options']['shaping_state']['base_appearance']),('current',run['options']['photo_appearance'])):
            loaded=appearance.load_asset(descriptor,**binding)
            expected[role]={'master':loaded['master'].copy(),'generated_coverage':np.array(loaded['generated_coverage']),
                'edit_coverage':np.array(loaded['edit_coverage']),'source_coverage':np.array(loaded['source_coverage']),
                'preview':Path(loaded['preview_path']).read_bytes()}
            del loaded
        cache=patch.dict(os.environ,{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'import-identity-cache')})
        cache.start();self.addCleanup(cache.stop)
        original=fx.root;hidden=original.with_name(original.name+'-unavailable')
        original.rename(hidden)
        try:
            target=self.root/'imported'
            with patch('raw_support.open_linear',side_effect=AssertionError('No RAW or raster re-decode')):
                imported=import_project(target,archive)
                imported=json.loads(Path(imported['manifest']).read_bytes())
                self.assertEqual(imported['options']['shaping_state']['authorization'],auth)
                for role,descriptor in (('base',imported['options']['shaping_state']['base_appearance']),('current',imported['options']['photo_appearance'])):
                    loaded=appearance.load_asset(descriptor,**binding)
                    for plane in ('master','generated_coverage','edit_coverage','source_coverage'):
                        np.testing.assert_array_equal(loaded[plane],expected[role][plane])
                    self.assertEqual(Path(loaded['preview_path']).read_bytes(),expected[role]['preview']);del loaded
                # Restore materializes the same immutable U, without a decoding or model boundary.
                restored=copy.deepcopy(imported);state=restored['options']['shaping_state']
                restore_auth=copy.deepcopy(auth);restore_auth.update(request={'action':'restore'},manual_target=None,manual_provider_target=None)
                restore_auth['authorization_id']=identity({k:v for k,v in restore_auth.items() if k!='authorization_id'})
                state.update(enabled=False,request={'action':'restore'},authorization=restore_auth,
                    selected_asset_id=state['base_appearance']['asset_id'])
                restored['options']['photo_appearance']=copy.deepcopy(state['base_appearance'])
                restored['task_plan'].update(shaping_request={'action':'restore'},shaping_authorization=restore_auth)
                result=render_candidate(target,restored,argparse.Namespace(cancel_request_path=None))
                self.assertEqual(Path(result['outputs']['auto_draft']).read_bytes(),expected['base']['preview'])
                np.testing.assert_array_equal(tifffile.imread(result['outputs']['render_scene_master']),expected['base']['master'])
                self.assertIsNone(result['review']['reviewer'])
        finally:
            hidden.rename(original)

    def test_portable_rejects_shaping_state_and_owned_u_tampering_atomically(self):
        fx,run,binding=self.shaped();archive=self.archive(run,fx.root)
        cases={
            'missing-u':lambda r:r['options']['shaping_state'].pop('base_appearance'),
            'wrong-current':lambda r:r['options']['shaping_state'].update(selected_asset_id='f'*64),
            'changed-u-binding':lambda r:r['options']['shaping_state']['base_appearance'].update(receipt_sha256='e'*64),
            'changed-request':lambda r:r['options']['shaping_state']['request']['parts'].update(face=.75),
            'changed-authorization':lambda r:r['task_plan']['shaping_authorization'].update(base_image_sha256='d'*64),
        }
        for name,change in cases.items():
            with self.subTest(name=name):
                changed=self.rewrite(archive,name,run_change=change);target=self.root/name
                with self.assertRaises((ValueError,KeyError)):
                    import_project(target,changed)
                self.assertFalse(list((target/'outputs/runs').glob('*/manifest.json')))

    def test_every_owned_baseline_file_is_required_and_content_verified(self):
        from portable_project import _appearance_files
        fx,run,binding=self.shaped();archive=self.archive(run,fx.root)
        base=run['options']['shaping_state']['base_appearance']
        receipt=appearance.validate_asset(base,**binding)
        with zipfile.ZipFile(archive) as original:
            original_payloads={item.filename:original.read(item.filename) for item in original.infolist()}
        package=json.loads(original_payloads['project.json'])
        member=package['photo_appearance_groups'][base['asset_id']]['archive_asset']
        with zipfile.ZipFile(BytesIO(original_payloads[member])) as group:
            original_files={item.filename:group.read(item.filename) for item in group.infolist()}
        files=_appearance_files(receipt,base['receipt_sha256'])
        self.assertIn('master.npy',files);self.assertIn('preview.png',files)
        self.assertTrue(any('coverage' in name for name in files))
        self.assertIn(receipt['generation']['provider_receipt']['file'],files)
        for index,name in enumerate(files):
            for action in ('remove','change'):
                with self.subTest(file=name,action=action):
                    payloads=copy.deepcopy(original_payloads);changed_files=copy.deepcopy(original_files)
                    if action=='remove':changed_files.pop(name)
                    else:
                        data=bytearray(changed_files[name]);data[-1]^=1;changed_files[name]=bytes(data)
                    stream=BytesIO()
                    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as packed:
                        for filename,data in changed_files.items():packed.writestr(filename,data)
                    payloads[member]=stream.getvalue();changed_package=copy.deepcopy(package)
                    changed_package['assets'][member].update(sha256=hashlib.sha256(payloads[member]).hexdigest(),bytes=len(payloads[member]))
                    payloads['project.json']=json.dumps(changed_package).encode()
                    target=self.root/f'baseline-{index}-{action}';changed=target.with_suffix('.cosplayproj')
                    with zipfile.ZipFile(changed,'w',compression=zipfile.ZIP_STORED) as packed:
                        for filename,data in payloads.items():packed.writestr(filename,data)
                    with self.assertRaises(ValueError):import_project(target,changed)
                    self.assertFalse(list((target/'outputs/runs').glob('*/manifest.json')))


if __name__=='__main__':unittest.main()
