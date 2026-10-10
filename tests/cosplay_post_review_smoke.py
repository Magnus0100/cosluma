"""CPU-only mocked contract regressions; no network or image generation."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import cosplay_post_review as review


def verdict(target, *, status='pass', cause='none', stage=None, action='', panels=None):
    return {'status': status, 'cause': cause, 'correction_stage': stage, 'action': action,
            'reason': 'Visible photographic evidence is limited to the supplied panels.',
            'evidence': [] if status == 'unknown' else [
                {'basis': 'visible_pixels', 'panels': panels or [target],
                 'observation': 'The displayed material has continuous texture and a distinct edge.'}]}


def response(has_final):
    return {'schema': review.OBSERVATION_SCHEMA, 'checks': {
        key: {'reference': verdict('reference'),
              'final': verdict('final') if has_final else verdict('final', status='unknown', cause='unresolved')}
        for key in review.CRITERIA}}


def fake_transport(payload, *, provenance_mutation=None, after_request=None):
    def request(source, output, prompt, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        sent = output/'mock-api-input.jpg'
        with Image.open(source) as image:
            image.convert('RGB').save(sent, quality=93, subsampling=0)
            size = list(image.size)
        provenance = {'schema_version': 'vision_transport.v1', 'provider': 'cpu_mock',
                      'model': 'mock_vision', 'response_id': 'mock_completed_response',
                      'protocol': 'responses', 'network_used': True, 'generation_used': False,
                      'source': {'path': str(source), 'sha256': review._sha(source)},
                      'api_input': {'path': str(sent), 'sha256': review._sha(sent), 'size': size,
                                    'source': {'sha256': review._sha(source)}},
                      'input_sha256': review._sha(sent),
                      'prompt_sha256': review._digest((prompt+review.TRANSPORT_SUFFIX).encode('utf-8'))}
        if provenance_mutation:
            provenance_mutation(provenance)
        if after_request:
            after_request()
        return copy.deepcopy(payload), provenance
    return request


def multi_fake_transport(payload=None, *, malformed_group=None, provenance_mutation=None):
    """Inspect actual per-board panel facts; cite every matched native pair."""
    def request(source, output, prompt, **kwargs):
        facts = json.loads(prompt.split('VISIBLE PANEL FACTS:\n', 1)[1].split('\nINTENDED BRIEF', 1)[0])
        if facts.get('pairs'):
            current = {'schema': review.NATIVE_OBSERVATION_SCHEMA, 'pairs': [
                {'pair_id': row['pair_id'], 'checks': {key: verdict(row['final_panel'],
                    panels=[row['source_panel'], row['final_panel']]) for key in review.LOCAL_CRITERIA}}
                for row in facts['pairs']]}
            if malformed_group and Path(source).stem == malformed_group:
                current['pairs'][0]['checks'].pop('native_detail')
            return fake_transport(current, provenance_mutation=provenance_mutation)(source, output, prompt, **kwargs)
        current = copy.deepcopy(payload or response(True))
        native = [panel['id'] for panel in facts['panels'] if panel.get('displayed_at_100_percent')]
        current['checks']['native_detail']['reference'] = verdict('reference', status='unknown', cause='unresolved')
        current['checks']['native_detail']['final'] = (verdict('final', panels=['final', *native]) if native else
                                                       verdict('final', status='unknown', cause='unresolved'))
        if malformed_group and Path(source).stem == malformed_group:
            current['checks'].pop('effects_space')
        return fake_transport(current, provenance_mutation=provenance_mutation)(source, output, prompt, **kwargs)
    return request


def main():
    from cosplay_native_review_smoke import make_native_fixture
    from cosplay_native_review import prepare_cosplay_native_review
    with tempfile.TemporaryDirectory(prefix='current-comprehensive-live-') as directory:
        root=Path(directory);run=make_native_fixture(root/'inputs')
        prepared=prepare_cosplay_native_review(run,root/'sampling',comprehensive=True)
        design={'selected_style_id':'fixture','design_brief':{'summary':'Current CPU fixture',
          'primary_effect_plan':{'mode':'none','description':'No effect'}}}
        def produce(name,transport):
            with patch.object(review.vision_transport,'request_vision_json',transport):
                return review.review_cosplay_post(prepared['source_path'],Path(run['reference']['path']),root/name,
                    design,final=prepared['final_path'],native_crops=prepared['native_crops'],native_selection=prepared['receipt'])
        current=produce('valid',multi_fake_transport())
        assert current['schema']=='cosplay_post_review.v2' and current['status']=='reviewed'
        assert current['actual_api_evidence_valid'] and len(current['reviews'])>1
        assert current['scope_contract'] == review.SCOPE_CONTRACT
        assert current['checks']['native_detail']['final']['status'] == 'pass'
        assert not current['whole_image_native_acceptance']
        assert current['native_sampling_coverage']['unsupported_regions']
        assert all(board['checks']['effects_space']['final']['status'] == 'unknown'
                   for board in current['reviews'] if board['board_scope'] == 'native')
        deferred = copy.deepcopy(design)
        deferred['design_brief']['foreground_effect_plan'] = {'enabled': True, 'layers': [{
            'id': 'additional_sparse_haze', 'slot': 'front', 'operation': 'over',
            'increment': True, 'baked_in_reference': False,
            'effect_design': {'material': 'atmospheric_haze', 'colour_rgb8': [240, 170, 95],
                              'extent_2d': [0, .6, .4, .9], 'density': 'sparse'}}]}
        assert review._brief(deferred)['intended_brief']['foreground_effect_plan'] == deferred['design_brief']['foreground_effect_plan']
        prompt = review._prompt(deferred, [], [], True)
        assert 'absence of those deferred increments is not a reference' in prompt
        assert 'final.effects_space against ALL required effects' in prompt
        assert 'primary background effect remains required' in prompt
        untyped = copy.deepcopy(deferred)
        untyped['design_brief']['foreground_effect_plan']['layers'][0].pop('effect_design')
        assert 'REFERENCE EFFECT SCOPE:' not in review._prompt(untyped, [], [], True)
        with patch.object(review.vision_transport, 'request_vision_json', multi_fake_transport()):
            independent = review.review_cosplay_post(prepared['source_path'], Path(run['reference']['path']),
                root/'independent-effect', deferred, final=prepared['final_path'],
                native_crops=prepared['native_crops'], native_selection=prepared['receipt'])
        assert any(board['board_scope'] == 'context' for board in independent['reviews'])
        partial=produce('partial',multi_fake_transport(malformed_group='native-01'))
        assert partial['status']=='partial' and not partial['actual_api_evidence_valid']
        assert not partial['visual_acceptance'] and not partial['whole_image_native_acceptance']
        try:review.review_cosplay_post(prepared['source_path'],Path(run['reference']['path']),root/'old',design)
        except ValueError:pass
        else:raise AssertionError('Retired single-board QA accepted')
        transport = multi_fake_transport()
        def local_failure(source, output, prompt, **kwargs):
            payload, provenance = transport(source, output, prompt, **kwargs)
            if payload['schema'] == review.NATIVE_OBSERVATION_SCHEMA and Path(source).stem == 'native-01':
                row = payload['pairs'][0]
                row['checks']['scene_integration'] = verdict('final', status='fail',
                    cause='composition_defect', stage='scene_composition', action='Repair this local halo.',
                    panels=row['checks']['native_detail']['evidence'][0]['panels'])
            return payload, provenance
        failed=produce('local-failure',local_failure)
        assert failed['checks']['scene_integration']['final']['status']=='fail'
        assert failed['checks']['concept_hierarchy']['final']['status']=='pass'
        assert failed['checks']['effects_space']['final']['status']=='pass'
        assert any(a.get('pair_id') and a['status']=='fail' for a in failed['required_actions'])
        def borrowed_pair(source, output, prompt, **kwargs):
            payload, provenance = transport(source, output, prompt, **kwargs)
            if payload['schema'] == review.NATIVE_OBSERVATION_SCHEMA and len(payload['pairs'])>1:
                payload['pairs'][0]['checks']['native_detail']['evidence'] = payload['pairs'][1]['checks']['native_detail']['evidence']
            return payload, provenance
        borrowed=produce('borrowed-pair',borrowed_pair)
        assert borrowed['status']=='partial' and not borrowed['actual_api_evidence_valid']
        assert not borrowed['whole_image_native_acceptance']
        try:produce('valid',multi_fake_transport())
        except FileExistsError:pass
        else:raise AssertionError('Existing actual QA evidence was overwritten')
    print(json.dumps({'status':'passed','current_schema':review.SCHEMA,'scope_contract':review.SCOPE_CONTRACT,
        'cases_passed':11,'network_requests':0}))


if __name__=='__main__':main()
