"""Existing Beauty fast paths own frozen bytes on checked and source-only creation."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import single_source_workflow as workflow
from portrait_beauty_backend_smoke import project, prepared
from portrait_beauty_layer import identity_for_run, load_result, apply_for_run
from creative_task import INTENT_SCHEMA, PLAN_SCHEMA, normalize_task_intent, image_budget
from cos_node_registry import compile_workflow


def main():
    with tempfile.TemporaryDirectory(prefix='beauty-existing-owner-') as temporary:
        root=Path(temporary)
        parent,_=project(root/'project')
        workspace=root/'project'
        generated=prepared(parent,workspace/'outputs/runs/parent-preparation')
        control={'result':generated['result'],'strength':1.,'disabled_faces':[]}
        identity=identity_for_run({**parent,'options':{**parent['options'],
            'appearance_revision':{'portrait_beauty':control}}})
        reference=workspace/'reference.png'
        Image.new('RGB',(32,24),(120,100,80)).save(reference)
        owners=[]
        with ExitStack() as stack:
            # No installed GPU/model runtime is needed for frozen-layer
            # ownership; only capability discovery is substituted here.
            stack.enter_context(patch('runtime_preflight.preflight_for_run',return_value={'status':'ready','scope':'CPU fixture'}))
            stack.enter_context(patch('source_alpha.source_silhouette',side_effect=lambda root,path,image,**kwargs:
                (Image.new('L',image.size,255),{'backend':'CPU fixture'})))
            stack.enter_context(patch('beautygrpo_prepare.prepare_for_workbench',side_effect=AssertionError('Existing Beauty ran a model')))
            stack.enter_context(patch('beautygrpo_prepare.prepare_cached_for_workbench',side_effect=AssertionError('Existing Beauty probed inference cache')))
            for mode in ('existing_reference','local_portrait'):
                args=workflow.parse_args(['--root',str(workspace),'--source',parent['source']['path'],
                    '--run-id',mode,'--parent-run-id',parent['run_id']])
                args.appearance_revision={'portrait_beauty':copy.deepcopy(control)}
                incoming=copy.deepcopy(args.appearance_revision)
                intent=normalize_task_intent({'schema_version':INTENT_SCHEMA,'mode':mode})
                args.task_intent=intent
                args.task_plan={'schema_version':PLAN_SCHEMA,'intent':intent,'mode':mode,'ready':True,
                    'image_budget':image_budget(intent),'plan_id':'a'*64,
                    'execution_strategy':'layered_reference_transfer','layer_workflow':compile_workflow(mode),
                    'inputs':{'source':{'sha256':parent['source']['sha256']},
                              'reference':{'sha256':workflow.sha256(reference)}}}
                if mode=='existing_reference':
                    args.reference=reference
                    def render(root,run,args):
                        actual=Path(run['options']['appearance_revision']['portrait_beauty']['result'])
                        assert actual.is_relative_to(workspace/'outputs/runs'/mode)
                        assert run['portrait_beauty_preparation']['status']=='frozen_layer_reused'
                        return {'metadata':{'run_id':run['run_id']},'outputs':{},'review':{}}
                    # The reference numerical pipeline is outside this
                    # intake regression. Its no-inference Beauty gate is real.
                    with patch.object(workflow,'_prepare_task_boundary_light',return_value=False), \
                            patch.object(workflow,'render_candidate',side_effect=render):
                        run=workflow.create_run(args)
                else:
                    # Actual source-only render/save consumes the owned Beauty.
                    run=workflow.create_run(args)
                assert run['status']=='needs_visual_review',run.get('error')
                assert args.appearance_revision==incoming,'Ownership mutated the submitted request'
                assert identity_for_run(run)==identity
                path=Path(run['options']['appearance_revision']['portrait_beauty']['result'])
                assert path.is_relative_to(workspace/'outputs/runs'/mode)
                layer,_=load_result(path)
                for face in layer['faces']:
                    assert Path(face['pixels']['path']).is_relative_to(path.parent)
                    assert Path(face['model_output']['path']).is_relative_to(path.parent)
                owners.append(run)
            different=workspace/'different-source.png'
            Image.new('RGB',(32,24),(10,20,30)).save(different)
            invalid=workflow.parse_args(['--root',str(workspace),'--source',str(different),
                '--run-id','rejected-owner','--local-only'])
            invalid.appearance_revision={'portrait_beauty':copy.deepcopy(control)}
            try:
                workflow.create_run(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('A layer bound to another source was adopted')
            assert not (workspace/'outputs/runs/rejected-owner/inputs/portrait-beauty').exists()
        # The closures survive removal of the isolated preparation fixture;
        # this deletes no actual photograph, delivery, or user program.
        fixture=workspace/'outputs/runs/parent-preparation'
        assert fixture.resolve().is_relative_to(root.resolve())
        shutil.rmtree(fixture)
        for run in owners:
            assert identity_for_run(run)==identity
            before=np.full((24,32,3),.05,np.float32)
            after,receipt=apply_for_run(before,run) if run['reference'] is None else (None,None)
            if after is not None:
                assert receipt['applied'] and not np.array_equal(before,after)
    print(json.dumps({'ok':True,'checked_retry_and_source_only_own_existing_beauty':True,
        'no_model_or_cache_requests':True,'request_unchanged':True,
        'appearance_identity_unchanged':True,'source_only_actual_render_saved':True,
        'removed_fixture_does_not_break_owned_closures':True}))


if __name__=='__main__':main()
