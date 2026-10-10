"""Offline checks for paid submission limits; never contacts a provider."""
from pathlib import Path
from unittest.mock import patch
import io
import json
import tempfile
import urllib.error

from PIL import Image
import reference_provider
import single_source_workflow as workflow
from backend_server import Backend


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_budget_') as folder:
        root = Path(folder)
        source = root / 'source.png'
        Image.new('RGB', (24, 32), '#806050').save(source)
        for failure in (TimeoutError('response lost'), urllib.error.HTTPError(
                'https://example.invalid', 429, 'limit', {}, io.BytesIO(b'limited'))):
            events = []
            with patch('reference_provider.urllib.request.urlopen', side_effect=failure) as transport:
                try:
                    reference_provider.request_edit('https://example.invalid', 'fixture-key', source,
                        'fixture-model', 'test', None, 1,
                        retry={'max_attempts': 8, 'backoff_seconds': 1, 'retry_statuses': [429]}, events=events)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('failure should be reported')
                assert transport.call_count == 1
                assert len(events) == 1 and events[0]['retrying'] is False

        calls = []
        def provider(_root, run, args):
            calls.append(args.candidate_out_dir.name)
            if len(calls) == 2:
                raise RuntimeError('lost response: may have been billed')
            args.candidate_out_dir.mkdir(parents=True)
            candidate = args.candidate_out_dir / (run['sample'] + '.png')
            Image.new('RGB', (24, 32), '#806050').save(candidate)
            return candidate

        args = workflow.parse_args(['--root', str(root), '--source', str(source), '--generate'])
        args.variant_count = 3
        args.effective_prompt = 'Exact saved English brief.'
        args.preserve_effective_prompt = True
        from generation_design import DESIGN_VERSION
        args.generation_design = {'schema_version':DESIGN_VERSION,'status':'ready',
            'effective_prompt':args.effective_prompt,'input_prompt':'offline budget fixture',
            'design_brief':{},'character_research':{}}
        draft = {'outputs': {'auto_draft':str(source)}, 'metadata': {'fixture': True}, 'review': {}}
        with patch('single_source_workflow.generate_candidate', side_effect=provider), \
             patch('single_source_workflow.render_candidate', return_value=draft), \
             patch('single_source_workflow.diagnose_cosplay_post', return_value=None), \
             patch('character_profile.analyze_character', side_effect=AssertionError('no paid planning')), \
             patch('generation_design.prepare_generation_design', side_effect=AssertionError('no paid planning')):
            run = workflow.create_run(args)
        assert calls == ['v1', 'v2']
        budget = run['generation_budget']
        assert budget['max_image_submissions'] == 3 and budget['image_submissions'] == 2
        assert [item['status'] for item in budget['submissions']] == ['succeeded', 'failed_or_outcome_unknown']
        assert run['selected_variant_id'] == 'v1' and len(run['variants']) == 1
        assert run['reference'] and run['auto_draft'] and run['status'] == 'needs_visual_review'
        saved = json.loads((root / run['manifest']).read_text(encoding='utf-8'))
        assert saved['generation_budget'] == budget

        args = workflow.parse_args(['--root', str(root), '--source', str(source), '--local-only', '--run-id', 'local-fixture'])
        with patch('single_source_workflow.render_local_source', return_value=draft), \
             patch('single_source_workflow.diagnose_cosplay_post', return_value=None), \
             patch('single_source_workflow.generate_candidate', side_effect=AssertionError('local-only must not generate')), \
             patch('character_profile.analyze_character', side_effect=AssertionError('local-only must not recognize')):
            local = workflow.create_run(args)
        assert local['delivery_status'] == 'local_source_draft' and 'generation_budget' not in local
        body = {'source': str(source), 'local_only': True, 'generate': False, 'async': False}
        backend = Backend(root, embedded=True)
        with patch('single_source_workflow.render_local_source', side_effect=RuntimeError('models not installed')), \
             patch('single_source_workflow.generate_candidate', side_effect=AssertionError('must not pay')):
            failed_local = backend.one_click(body)['run']
        assert failed_local['status'] == 'blocked' and failed_local['next_action'] == 'retry_local_processing'
        assert 'fallback' not in failed_local and 'generation_budget' not in failed_local
        assert '--local-only' in backend._one_click_args(body, str(source))
    print(json.dumps({'ok': True, 'paid_calls': 0, 'old_retry_config_ignored': True,
        'partial_candidates_reused': True, 'local_only_no_gpt': True}))


if __name__ == '__main__':
    main()
