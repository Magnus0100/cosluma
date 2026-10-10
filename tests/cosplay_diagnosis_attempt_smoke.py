"""CPU regressions for immutable failed, interrupted and changed diagnosis attempts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from generation_design import DESIGN_VERSION
import single_source_workflow as workflow


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix='diagnosis-attempt-') as temporary:
        root = Path(temporary)
        source, reference, final = [root/(name+'.png') for name in ('source', 'reference', 'final')]
        for path, colour in ((source, '#804020'), (reference, '#304070'), (final, '#605050')):
            Image.new('RGB', (32, 40), colour).save(path)
        run = {'run_id': 'fixture', 'revision': 1, 'status': 'needs_visual_review',
            'source': {'path': str(source), 'sha256': workflow.sha256(source)},
            'reference': {'path': str(reference), 'sha256': workflow.sha256(reference)},
            'artifacts': {'auto_draft': str(final)},
            'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready',
                'design_brief': {'summary': 'CPU fixture intent'}}, 'options': {}}
        args = SimpleNamespace(generate=False, complete_post=True, local_only=False,
            provider=None, api_base=None, api_url=None, api_key_env='FIXTURE_NO_SECRET')
        output = root/workflow.RUNS_DIR/run['run_id']/'cosplay-post-review'
        with patch('cosplay_native_review.prepare_cosplay_native_review',
                   side_effect=ValueError('Original missing source semantic evidence')) as prepare:
            first = workflow.diagnose_cosplay_post(root, run, args)
            assert first['error'] == 'Original missing source semantic evidence'
            assert first['run_binding']['final_sha256'] == workflow.sha256(final)
            snapshot = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
            second = workflow.diagnose_cosplay_post(root, run, args)
            assert second == first and prepare.call_count == 1
            assert all(p.read_bytes() == data for p, data in snapshot.items())
            cases += ['original_preparation_error_preserved', 'same_failed_endpoint_no_request_or_overwrite']
            changed = copy.deepcopy(run)
            changed['generation_design']['design_brief']['summary'] += ' changed intent'
            blocked = workflow.diagnose_cosplay_post(root, changed, args)
            assert blocked['status'] == 'unavailable' and blocked['text_vision_submissions'] == 0
            original = final.read_bytes()
            Image.new('RGB', (32, 40), '#204030').save(final)
            blocked = workflow.diagnose_cosplay_post(root, run, args)
            assert blocked['text_vision_submissions'] == 0 and prepare.call_count == 1
            final.write_bytes(original)
            assert all(p.read_bytes() == data for p, data in snapshot.items())
            cases += ['changed_design_cannot_reuse_attempt', 'changed_pixels_cannot_overwrite_evidence']
            for role in ('source', 'reference'):
                path = Path(run[role]['path']); original = path.read_bytes()
                Image.new('RGB', (32, 40), '#203040').save(path)
                fresh = copy.deepcopy(run); fresh['run_id'] = 'changed-'+role
                blocked = workflow.diagnose_cosplay_post(root, fresh, args)
                assert blocked['text_vision_submissions'] == 0 and prepare.call_count == 1
                assert not (root/workflow.RUNS_DIR/fresh['run_id']/'cosplay-post-review').exists()
                path.write_bytes(original)
            cases.append('changed_source_or_reference_blocked_before_preparation')
            # An interrupted reserved attempt never grants another service call.
            attempt = workflow.read_json(output/'attempt.json')
            attempt.update(status='reserved'); attempt.pop('review_sha256')
            workflow.write_json(output/'attempt.json', attempt)
            saved = (output/'review.json').read_bytes()
            blocked = workflow.diagnose_cosplay_post(root, run, args)
            assert blocked['text_vision_submissions'] == 0 and 'interrupted' in blocked['error']
            assert (output/'review.json').read_bytes() == saved and prepare.call_count == 1
            cases.append('interrupted_attempt_no_implicit_retry')
            child = copy.deepcopy(run); child['run_id'] = 'new-child'; child['revision'] = 2
            child.pop('cosplay_post_review', None)
            workflow.diagnose_cosplay_post(root, child, args)
            assert prepare.call_count == 2
            cases.append('new_owned_child_gets_its_own_attempt')
            legacy = copy.deepcopy(run); legacy['run_id'] = 'existing-owned-evidence'
            legacy_output = root/workflow.RUNS_DIR/legacy['run_id']/'cosplay-post-review'
            workflow.write_json(legacy_output/'native-selection.json', {'existing': True})
            before = (legacy_output/'native-selection.json').read_bytes()
            blocked = workflow.diagnose_cosplay_post(root, legacy, args)
            assert blocked['text_vision_submissions'] == 0 and prepare.call_count == 2
            assert (legacy_output/'native-selection.json').read_bytes() == before
            assert not (legacy_output/'review.json').exists()
            cases.append('existing_unreserved_evidence_unchanged')
            interrupted = copy.deepcopy(run); interrupted['run_id'] = 'partial-old-preparation'
            old_output = root/workflow.RUNS_DIR/interrupted['run_id']/'cosplay-post-review'
            old_output.mkdir(parents=True)
            owned = old_output/'source-display.png'; owned.write_bytes(source.read_bytes())
            before = owned.read_bytes()
            blocked = workflow.diagnose_cosplay_post(root, interrupted, args)
            assert blocked['text_vision_submissions'] == 0 and prepare.call_count == 2
            assert owned.read_bytes() == before and not (old_output/'attempt.json').exists()
            cases.append('partial_unreserved_preparation_asset_unchanged')
        cancelled = copy.deepcopy(run); cancelled['run_id'] = 'cancel-during-native'
        signal = root/'cancel.json'; args.cancel_request_path = signal

        def cancel_preparation(*_args, **_kwargs):
            signal.write_text('{}', encoding='utf8')
            return {'receipt': {'status': 'unknown'}}

        with patch('cosplay_native_review.prepare_cosplay_native_review', side_effect=cancel_preparation), \
             patch('cosplay_post_review.review_cosplay_post', side_effect=AssertionError('No request after cancel')) as sender:
            result = workflow.diagnose_cosplay_post(root, cancelled, args)
            assert result['status'] == 'unavailable' and result['error'] == 'generation cancelled'
            assert sender.call_count == 0
            workflow.diagnose_cosplay_post(root, cancelled, args)
            assert sender.call_count == 0
            cases.append('cancel_during_preparation_prevents_vision_request')
        signal.unlink(); args.cancel_request_path = None
        changed = copy.deepcopy(run); changed['run_id'] = 'changed-during-native'
        original = reference.read_bytes()

        def change_preparation(*_args, **_kwargs):
            Image.new('RGB', (32, 40), '#202030').save(reference)
            return {'receipt': {'status': 'unknown'}}

        with patch('cosplay_native_review.prepare_cosplay_native_review', side_effect=change_preparation), \
             patch('cosplay_post_review.review_cosplay_post', side_effect=AssertionError('No request for changed endpoint')) as sender:
            result = workflow.diagnose_cosplay_post(root, changed, args)
            assert result['status'] == 'unavailable' and 'during native preparation' in result['error']
            assert sender.call_count == 0
            reference.write_bytes(original)
            cases.append('changed_endpoint_during_preparation_prevents_request')
    print(json.dumps({'status': 'passed', 'cases': cases, 'network_requests': 0,
                      'model_inferences': 0, 'controlled_preparation_failure': True}))


if __name__ == '__main__':
    main()
