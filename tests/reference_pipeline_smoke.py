"""Check reference preparation boundaries without models or network calls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

import design_reconstruction_pipeline as design_pipeline
import reference_pipeline
import single_source_workflow as workflow


def _snapshot(directory):
    return {path.relative_to(directory).as_posix():path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_reference_boundaries_') as folder:
        root=Path(folder);source=root/'source.jpg';candidate=root/'candidate.png'
        Image.new('RGB',(96,64),(70,100,140)).save(source)
        Image.new('RGB',(96,64),(115,150,180)).save(candidate)
        common=['--root',str(root),'--source',str(source),'--alpha-backend','birefnet']
        # Saved/source-only operations have no automatic reference dependency.
        with patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Source-only operation prepared reference assets')), \
             patch.object(design_pipeline,'ready',side_effect=AssertionError('Source-only operation required local reference models')), \
             patch.object(workflow,'generate_candidate',side_effect=AssertionError('Local-only operation generated a candidate')), \
             patch('source_alpha.source_silhouette',side_effect=lambda _root,_source,image,**kw:(Image.new('L',image.size,255),{'backend':'synthetic_test_matte'})):
            local=workflow.create_run(workflow.parse_args(common+['--run-id','local','--local-only']))
            assert local['status']=='needs_visual_review' and local['reference'] is None
            assert local['options']['render_mode']=='native_source_design'
            local_directory=root/workflow.RUNS_DIR/'local';saved_bytes=_snapshot(local_directory)
            copied=workflow.copy_source_run(root,'local')
            assert copied['reference'] is None and copied['saved_render_copy']['render_executed'] is False
            assert Path(copied['artifacts']['render_scene_master']).read_bytes()==Path(local['artifacts']['render_scene_master']).read_bytes()
            edited=workflow.adjust_source_run(root,copied['run_id'],appearance_revision={'params':{'exposure':.35}})
            assert edited['reference'] is None and edited['continuation_edit']['render_state']=='saved'
            assert Path(edited['artifacts']['auto_draft']).read_bytes()!=Path(local['artifacts']['auto_draft']).read_bytes()
            assert _snapshot(local_directory)==saved_bytes
        # A genuine provider failure preserves source-only fallback/copy/edit.
        args=workflow.parse_args(common+['--run-id','provider-failure','--generate'])
        args.character_profile={'identification':'fixture','recommended_styles':[]}
        with patch.object(workflow,'prepare_bound_design',return_value={'status':'unavailable', 'effective_prompt':'Source preservation fallback fixture'}), \
             patch.object(workflow,'generate_candidate',side_effect=RuntimeError('provider unavailable')) as generation, \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Missing reference prepared automatic assets')):
            fallback=workflow.create_run(args)
            assert generation.call_count==1
            assert fallback['status']=='blocked' and fallback['reference'] is None
            assert not fallback.get('auto_draft') and not fallback.get('fallback')
            assert fallback['error']['kind']=='generation_failed'
            assert generation.call_count==1
        # Importing a candidate into a workspace without models stays blocked;
        # neither the history fixture nor acceptance may bypass that contract.
        with patch.object(design_pipeline,'ready',return_value=False), \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Unavailable preparation was invoked')), \
             patch.object(workflow,'generate_candidate',side_effect=AssertionError('Retained candidate was regenerated')):
            blocked=workflow.create_run(workflow.parse_args(common+['--run-id','candidate-no-runtime','--reference',str(candidate)]))
        assert blocked['status']=='blocked' and blocked['error']['kind']=='local_reconstruction_failed'
        assert blocked['error']['candidate_retained'] and not blocked['error']['automatic_image_retry']
        assert Path(blocked['reference']['path']).read_bytes()==candidate.read_bytes()
        assert not blocked.get('auto_draft') and not blocked.get('fallback')
        try:
            workflow.accept_run(root,blocked['run_id'],reviewer='fixture',checks={key:'pass' for key in workflow.VISUAL_CHECK_KEYS})
            raise AssertionError('A blocked reference run passed the unchanged acceptance gate')
        except RuntimeError as exc:
            assert 'not awaiting visual review' in str(exc)
        # The public reconstruction entry point delegates to the reference
        # pipeline, preserving the explicit rebuild choice and caller input.
        call={'run_id':'boundary'};result={'ok':True,'design_reconstruction_plan':{}}
        with patch.object(reference_pipeline,'prepare',return_value=result) as prepare:
            assert design_pipeline.prepare(root,call,rebuild_subject=True) is result
            prepare.assert_called_once_with(root,call,rebuild=True)
        print(json.dumps({'ok':True,'local_source_without_reference_models':True,
            'source_only_copy_and_edit':True,'provider_failure_blocked_without_fake_final':True,
            'candidate_retained_when_models_missing':True,'blocked_accept_rejected':True,
            'public_prepare_delegates':True},ensure_ascii=False))


if __name__=='__main__':
    main()
