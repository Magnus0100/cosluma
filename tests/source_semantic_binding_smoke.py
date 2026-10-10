"""Offline QA of cropped source semantics, invalid bindings and whole-only QA."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cosplay_native_review as native
import cosplay_post_review as visual
from cosplay_native_review_smoke import add_frozen_feature_fixture, make_native_fixture
from cosplay_post_review_smoke import multi_fake_transport
from source_semantic_binding import bind_source_semantics


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix='source-semantics-qa-') as folder:
        root = Path(folder)
        run = make_native_fixture(root/'inputs', native_size=(660, 840))
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        labels = plan['layers'][0]['assets']['source_shoe_semantics']
        owned_contact = copy.deepcopy(plan['layers'][0])
        add_frozen_feature_fixture(run, root/'feature-observation')
        plan['mask_roles'] = {'source_semantics': bind_source_semantics(
            {**labels, 'labels_path': labels['path']}, source_sha256=run['source']['sha256'],
            native_size=plan['canvas']['source_oriented_size'], proxy_size=plan['canvas']['source_proxy_size'])}
        plan['layers'] = [layer for layer in plan['layers'] if layer['kind'] != 'contact_shadow']
        with patch.object(native, '_current_source_feature_observation', side_effect=AssertionError('No model allowed')):
            prepared = native.prepare_cosplay_native_review(run, root/'cropped')
        assert prepared['receipt']['status'] == 'ready', prepared['receipt']['reason']
        assert {row['region'] for row in prepared['receipt']['regions']} >= {'eyes', 'lips', 'left_hand', 'right_hand'}
        assert not any(row['region'] in {'left_shoe', 'right_shoe'} for row in prepared['receipt']['regions'])
        assert prepared['receipt']['semantic_binding']['source_sha256'] == run['source']['sha256']
        cases.append('cropped_no_feet_ready_without_contact_layer')

        for name, mutation in (
                ('bad_hash', lambda a: a.update(sha256='0'*64)),
                ('bad_model', lambda a: a.update(model='unrecognized/parser')),
                ('bad_model_revision', lambda a: a.update(model_revision='changed')),
                ('bad_weight_identity', lambda a: a.update(checkpoint_sha256='0'*64)),
                ('bad_taxonomy', lambda a: a.update(taxonomy_sha256='0'*64)),
                ('wrong_source', lambda a: a.update(source_sha256='0'*64)),
                ('wrong_native_canvas', lambda a: a.update(native_size=[1, 2])),
                ('wrong_proxy_canvas', lambda a: a.update(size=[1, 2]))):
            changed = copy.deepcopy(run)
            mutation(changed['options']['appearance_revision']['design_reconstruction_plan']['mask_roles']['source_semantics'])
            with patch.object(native, '_current_source_feature_observation', side_effect=AssertionError('No model allowed')):
                checked = native.prepare_cosplay_native_review(changed, root/name)
            assert checked['receipt']['status'] == 'unknown' and not checked['native_crops'], name
            assert checked['receipt']['context_regions'] == []
            cases.append(name+'_cannot_grant_native_evidence')

        damaged = copy.deepcopy(run)
        damaged_plan = damaged['options']['appearance_revision']['design_reconstruction_plan']
        damaged_plan['layers'].append(owned_contact)
        damaged_plan['mask_roles']['source_semantics']['sha256'] = '0'*64
        checked = native.prepare_cosplay_native_review(damaged, root/'no-fallback')
        assert checked['receipt']['status'] == 'unknown' and not checked['native_crops']
        cases.append('invalid_explicit_semantics_never_falls_back_to_contact')

        missing = copy.deepcopy(run)
        missing['options']['appearance_revision']['design_reconstruction_plan']['mask_roles'].pop('source_semantics')
        unknown = native.prepare_cosplay_native_review(missing, root/'unknown')
        assert unknown['receipt']['status'] == 'unknown' and not unknown['native_crops']
        design = {'design_brief': {'summary': 'Synthetic whole QA despite missing semantics',
                                  'primary_effect_plan': {'mode': 'scene_effect'}}}
        # Also covers old unknown receipts that predate an explicit empty context list.
        unknown['receipt'].pop('context_regions')
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=multi_fake_transport()) as sender:
            review = visual.review_cosplay_post(unknown['source_path'], Path(run['reference']['path']),
                root/'whole-only', design, final=unknown['final_path'],
                native_crops=unknown['native_crops'], native_selection=unknown['receipt'])
        assert sender.call_count == 1 and len(review['reviews']) == 1
        assert review['reviews'][0]['board_scope'] == 'whole' and review['actual_api_evidence_valid']
        assert review['checks']['native_detail']['final']['status'] == 'unknown'
        assert review['diagnostic_gate'] == 'unknown' and not review['whole_image_native_acceptance']
        assert review['native_sampling_coverage']['requested_pairs'] == []
        assert any(row['criterion'] == 'native_sampling' for row in review['required_actions'])
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No retry/replay network allowed')):
            saved, proof = visual.validate_saved_comprehensive_review(unknown['source_path'],
                run['reference']['path'], root/'whole-only/review.json', design)
        assert saved == review and proof
        cases.append('unknown_semantics_whole_QA_saved_replay_native_stays_unknown')
        print(json.dumps({'ok': True, 'cases': cases, 'synthetic_assets': True,
            'vision_transport_mocked': True, 'inference_verified': False,
            'network_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__': main()
