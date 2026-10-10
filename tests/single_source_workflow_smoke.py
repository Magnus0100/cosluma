#!/usr/bin/env python3
"""Check source-only entry states and the explicit saved-master delivery gate.

The native source-only fixture uses the current source baseline without local models.
Reference colour preparation belongs to reference_pipeline_smoke.py; an imported
candidate in this temporary workspace must remain blocked and retained.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import design_reconstruction_pipeline as design_pipeline
import reference_pipeline
import single_source_workflow as workflow


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='cosplay-source-only-') as value:
        root = Path(value)
        source, candidate = root/'source.jpg', root/'candidate.png'
        Image.new('RGB', (128, 96), (120, 80, 160)).save(source, quality=92)
        Image.new('RGB', (128, 96), (160, 120, 200)).save(candidate)
        common = ['--root',str(root),'--source',str(source),'--alpha-backend','birefnet']
        with patch.object(workflow,'generate_candidate',side_effect=AssertionError('Local entry generated a candidate')), \
             patch.object(reference_pipeline,'prepare',side_effect=AssertionError('Unavailable reference preparation was invoked')), \
             patch.object(design_pipeline,'ready',return_value=False), \
             patch('source_alpha.source_silhouette',side_effect=lambda _root,_source,image,**kw: (Image.new('L',image.size,255),{'backend':'synthetic_test_matte'})):
            awaiting = workflow.create_run(workflow.parse_args(common+['--run-id','run-awaiting']))
            assert awaiting['status']=='awaiting_reference' and awaiting['reference'] is None
            assert not awaiting.get('auto_draft')
            blocked = workflow.create_run(workflow.parse_args(common+['--run-id','run-candidate','--reference',str(candidate)]))
            assert blocked['status']=='blocked' and blocked['error']['candidate_retained'] is True
            assert blocked['error']['automatic_image_retry'] is False
            assert Path(blocked['reference']['path']).read_bytes()==candidate.read_bytes()
            assert not blocked.get('auto_draft') and not blocked.get('fallback')
            local = workflow.create_run(workflow.parse_args(common+['--run-id','run-local','--local-only']))
            assert local['status']=='needs_visual_review' and local['reference'] is None
            assert local['options']['render_mode']=='native_source_design'
            assert local['delivery_status']=='local_source_draft'
        checks = {key:'pass' for key in workflow.VISUAL_CHECK_KEYS}
        for unavailable in ('run-awaiting','run-candidate'):
            try:
                workflow.accept_run(root,unavailable,reviewer='fixture',checks=checks)
                raise AssertionError('An unrendered entry passed the delivery gate')
            except RuntimeError as exc:
                assert 'not awaiting visual review' in str(exc), exc
        masters = {role:Path(local['artifacts'][role]) for role in ('render_master','render_scene_master')}
        assert all(path.is_file() for path in masters.values())
        assert local['render']['render_precision']['calibration_rgb8'] is False
        review_path = root/'outputs/runs/run-local/visual_review.json'
        review = json.loads(review_path.read_text(encoding='utf8'))
        assert all(value=='not_checked' for value in review['checks'].values())
        review['revision'] = 999
        review_path.write_text(json.dumps(review),encoding='utf8')
        try:
            workflow.accept_run(root,'run-local',reviewer='fixture',checks=checks)
            raise AssertionError('Delivery accepted review from another revision')
        except RuntimeError as exc:
            assert 'revision' in str(exc), exc
        review['revision'] = local['revision']
        review_path.write_text(json.dumps(review),encoding='utf8')
        try:
            workflow.accept_run(root,'run-local',reviewer='fixture')
            raise AssertionError('A successful render inherited visual acceptance')
        except RuntimeError as exc:
            assert 'visual checks' in str(exc), exc
        delivered = workflow.accept_run(root,'run-local',reviewer='fixture',checks=checks)
        assert delivered['status']=='delivered'
        delivery = root/'outputs/final_delivery/source_only/source/run-local'
        receipt = json.loads((delivery/'delivery.json').read_text(encoding='utf8'))
        assert receipt['run_id']=='run-local' and receipt['checks']==checks
        assert receipt['source_sha256']==workflow.sha256(source) and receipt['candidate_sha256'] is None
        assert receipt['software_version'] and receipt['fallback'] is None
        for path in masters.values():
            assert (delivery/path.name).read_bytes()==path.read_bytes()
            assert receipt['artifacts'][path.name]==workflow.sha256(path)
        assert receipt['render_master_sha256']==workflow.sha256(masters['render_master'])
        assert not (delivery/'candidate.png').exists()
        bad = root/'bad.txt'; bad.write_text('not an image',encoding='utf8')
        try:
            workflow.create_run(workflow.parse_args(['--root',str(root),'--source',str(bad),'--run-id','run-bad']))
            raise AssertionError('Unsupported source extension was accepted')
        except ValueError: pass
        print(json.dumps({'ok':True,'awaiting_reference_without_render':True,'blocked_candidate_retained':True,
            'explicit_native_source_only_fixture':True,'unrendered_accept_rejected':True,
            'review_revision_and_eight_checks_required':True,'saved_master_delivery_bytes_and_hashes':True,
            'no_generation_or_reference_models':True,'unsupported_source_rejected':True,
            'visual_quality_acceptance_simulated':True}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
