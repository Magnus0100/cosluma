"""CPU regressions for consuming saved v2 QA; isolated synthetic artifacts only."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cosplay_post_review as visual
import reference_refinement as refinement
from cosplay_native_review import prepare_cosplay_native_review
from cosplay_native_review_smoke import make_native_fixture, add_frozen_feature_fixture
from cosplay_post_review_smoke import multi_fake_transport, response, verdict


def main():
    passed = []
    rejected = {}
    with tempfile.TemporaryDirectory(prefix='saved-comprehensive-review-smoke-') as temporary:
        root = Path(temporary)
        run = make_native_fixture(root/'inputs')
        add_frozen_feature_fixture(run, root/'frozen-observation')
        alpha = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['foreground_alpha']
        np.savez_compressed(alpha['path'], values=np.full((1400, 1100), .75, np.float32))
        alpha['sha256'] = visual._sha(alpha['path'])
        prepared = prepare_cosplay_native_review(run, root/'selection', comprehensive=True)
        assert prepared['receipt']['status'] == 'ready'
        design = {'selected_style_id': 'cpu_fixture', 'design_brief': {
            'summary': 'A source-owned native costume photograph; intention is not evidence.',
            'primary_effect_plan': {'mode': 'scene_effect', 'description': 'A source-bound restrained background smoke effect'}}}
        payload = response(True)
        payload['checks']['skin_materials']['reference'] = verdict('reference', status='fail',
            cause='reference_defect', stage='reference_generation', action='Correct the reference material colour.',
            panels=['source', 'reference'])
        payload['checks']['scene_integration']['final'] = verdict('final', status='fail',
            cause='composition_defect', stage='scene_composition', action='FINAL_ONLY_CORRECT_THE_CONTACT.',
            panels=['source', 'reference', 'final'])

        transport = multi_fake_transport(payload)
        def native_failure(source, output, prompt, **kwargs):
            current, provenance = transport(source, output, prompt, **kwargs)
            if Path(source).stem == 'native-02':
                row = current['pairs'][0]
                row['checks']['identity_makeup'] = verdict('final', status='fail',
                    cause='unresolved', stage='native_retouch',
                    panels=row['checks']['native_detail']['evidence'][0]['panels'],
                    action='Restore the source-owned local shape, not the reference identity.')
            return current, provenance

        def produce(directory, mock=native_failure):
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=mock):
                return visual.review_cosplay_post(prepared['source_path'], Path(run['reference']['path']), directory,
                    design, final=prepared['final_path'], native_crops=prepared['native_crops'],
                    native_selection=prepared['receipt'], max_workers=2)

        qa = root/'cosplay-post-review'
        result = produce(qa)
        assert result['status'] == 'reviewed' and len(result['reviews']) >= 3
        path = qa/'review.json'
        result['run_binding'] = {**prepared['receipt']['run_binding'],
            'final_sha256': result['binding']['images']['final']['sha256'],
            'render_artifacts': {'auto_draft': result['binding']['images']['final']['sha256']}}
        visual._write(path, result)
        snapshot = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}

        def consume(source=prepared['source_path'], reference=run['reference']['path'], doc=design):
            with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No API allowed')), \
                 patch('local_photo_analysis.prepare_local_appearance', side_effect=AssertionError('No model allowed')):
                return refinement._bound_review(source, reference, path, doc)

        def read(file=path):
            return json.loads(file.read_text(encoding='utf-8-sig'))

        def edit(file, operation):
            value = read(file)
            operation(value)
            visual._write(file, value)

        def sync_board(index, operation):
            current = read()
            operation(current['reviews'][index])
            visual._write(qa/current['reviews'][index]['id']/'review.json', current['reviews'][index])
            visual._write(path, current)

        def sync_selection(operation):
            current = read()
            operation(current['native_selection'])
            current['binding']['sampling_contract_sha256'] = visual._digest(visual._json_bytes(current['native_selection']))
            visual._write(qa/'sampling-contract.json', current['native_selection'])
            visual._write(path, current)

        def reset():
            for p, data in snapshot.items():
                if not p.is_file() or p.read_bytes() != data:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
            for p in root.rglob('*'):
                if p.is_file() and p not in snapshot:
                    p.unlink()

        def reject(name, mutate=lambda: None, **supplied):
            try:
                mutate()
                consume(**supplied)
            except ValueError as exc:
                rejected[name] = str(exc)
                passed.append(name)
            else:
                raise AssertionError('Accepted tampered evidence: '+name)
            finally:
                reset()

        checked, actions, proof = consume()
        assert checked == result
        assert {row['criterion'] for row in actions} == {'skin_materials'}
        assert all(row['correction_stage'] == 'reference_generation' for row in actions)
        assert all('FINAL_ONLY' not in row['action'] for row in actions)
        assert checked['checks']['identity_makeup']['reference']['status'] == 'pass'
        assert checked['checks']['identity_makeup']['final']['status'] == 'fail'
        assert checked['checks']['native_detail']['final']['status'] == 'pass'
        assert not checked['whole_image_native_acceptance']
        assert checked['native_sampling_coverage']['unobserved_pairs'] == []
        assert len(proof) == len({row['path'] for row in proof})
        assert all(visual._sha(row['path']) == row['sha256'] for row in proof)
        assert {str(qa/board['id']/'model-response.json') for board in checked['reviews']} <= {row['path'] for row in proof}
        assert all(p.read_bytes() == data for p, data in snapshot.items())
        passed.append('valid_v2_reconstructed_and_read_only_reference_only_actions')
        reject('missing_current_scope_contract', lambda: edit(path, lambda r: r.pop('scope_contract')))
        reject('local_pass_promoted_to_global_scope', lambda: edit(path, lambda r:
            r['criterion_scopes'].update(native_detail='whole_image_native_quality')))
        reject('context_selection_changed_with_rehashed_snapshot', lambda: sync_selection(lambda s:
            s['context_regions'][0].update(native_box=[1, 0, 1000, 1000])))
        reject('frozen_source_feature_mask_bytes_changed', lambda: Path(
            result['native_selection']['feature_binding']['mask_records']['eyes']['path']).write_bytes(b'changed-eye-mask'))
        reject('frozen_material_observation_origin_changed', lambda: Path(
            result['native_selection']['feature_binding']['origin_records'][1]['path']).write_bytes(b'changed-frozen-input'))
        reject('erased_specific_local_failure', lambda: edit(path, lambda r:
            r['local_observations'].clear()))

        # Aggregate claims never outrank a saved response or board observation.
        reject('softened_aggregate_reference_failure', lambda: edit(path, lambda r:
            r['checks']['skin_materials'].update(reference=verdict('reference'))))
        reject('changed_aggregate_reference_action', lambda: edit(path, lambda r:
            r['checks']['skin_materials']['reference'].update(action='A different unauthorised correction.')))
        reject('erased_native_final_failure', lambda: edit(path, lambda r:
            r['checks']['identity_makeup'].update(final=verdict('final'))))
        reject('changed_final_gate', lambda: edit(path, lambda r: r.update(diagnostic_gate='pass')))
        reject('deleted_required_actions', lambda: edit(path, lambda r: r.update(required_actions=[])))
        reject('falsified_aggregate_sampling_coverage', lambda: edit(path, lambda r:
            r['native_sampling_coverage'].update(actually_observed_pairs=[])))
        reject('promoted_whole_image_acceptance', lambda: edit(path, lambda r:
            r.update(whole_image_native_acceptance=True)))
        reject('promoted_automatic_acceptance', lambda: edit(path, lambda r: r.update(automatic_accepted=True)))
        reject('plan_text_as_execution', lambda: edit(path, lambda r: r.update(plan_text_is_execution_evidence=True)))
        reject('partial_receipt', lambda: edit(path, lambda r: r.update(status='partial')))
        reject('missing_board', lambda: edit(path, lambda r: r['reviews'].pop()))
        reject('duplicate_board', lambda: edit(path, lambda r: r['reviews'].append(r['reviews'][-1])))
        reject('reordered_boards', lambda: edit(path, lambda r: r['reviews'].reverse()))
        reject('embedded_board_changed', lambda: edit(path, lambda r:
            r['reviews'][0]['checks']['skin_materials']['reference'].update(action='Forged board action.')))
        reject('standalone_board_changed', lambda: edit(qa/'whole/review.json', lambda r: r.update(error='changed')))
        reject('board_verdict_changed_in_both_receipts', lambda: sync_board(0, lambda b:
            b['checks']['skin_materials'].update(reference=verdict('reference'))))
        reject('raw_response_changed', lambda: edit(qa/'whole/model-response.json', lambda r:
            r['checks']['skin_materials']['reference'].update(action='Edited raw response.')))

        def edit_response_and_board():
            changed = verdict('reference')
            edit(qa/'whole/model-response.json', lambda r: r['checks']['skin_materials'].update(reference=changed))
            sync_board(0, lambda b: b['checks']['skin_materials'].update(reference=changed))
        reject('response_and_embedded_check_changed_without_original_hash', edit_response_and_board)

        reject('provenance_changed', lambda: edit(qa/'whole/vision-provenance.json', lambda r:
            r.update(network_used=False)))
        reject('native_observed_pairs_invented', lambda: sync_board(1, lambda b:
            b.update(native_pairs_actually_observed=['UNOBSERVED_PAIR'])))
        reject('native_scope_promoted', lambda: sync_board(1, lambda b:
            b.update(native_detail_scope='whole_image_native_quality')))
        reject('native_panel_label_changed', lambda: sync_board(1, lambda b:
            b['comparison']['panels'][-1].update(pair_id='WRONG_PAIR')))
        reject('raw_prompt_changed', lambda: (qa/'whole/request-prompt.txt').write_text('A replacement request.', encoding='utf-8'))
        reject('saved_design_changed', lambda: edit(qa/'design.json', lambda r:
            r['design_brief'].update(summary='Changed intended scene.')))
        reject('different_supplied_design', doc={'design_brief': {'summary': 'A different design'}})
        reject('different_supplied_source', source=prepared['final_path'])
        reject('different_supplied_reference', reference=prepared['source_path'])

        def change_pixel(file):
            with Image.open(file) as image:
                image = image.convert('RGB')
                image.putpixel((0, 0), (0, 1, 2))
                image.save(file)
        reject('bound_source_pixels_changed', lambda: change_pixel(prepared['source_path']))
        reject('bound_reference_pixels_changed', lambda: change_pixel(Path(run['reference']['path'])))
        reject('bound_final_pixels_changed', lambda: change_pixel(prepared['final_path']))
        reject('native_crop_pixels_changed', lambda: change_pixel(Path(result['native_crops'][0]['asset']['path'])))
        reject('native_crop_id_changed', lambda: edit(path, lambda r: r['native_crops'][0].update(id='native_wrong')))
        reject('native_crop_box_changed', lambda: edit(path, lambda r: r['native_crops'][0].update(box=[0, 0, 312, 286])))
        reject('sampling_snapshot_changed', lambda: edit(qa/'sampling-contract.json', lambda r: r.update(reason='Changed snapshot')))
        reject('baseline_asset_changed', lambda: Path(alpha['path']).write_bytes(b'not-the-bound-alpha'))
        reject('semantic_asset_changed', lambda: Path(prepared['receipt']['semantic_binding']['path']).write_bytes(b'wrong labels'))
        reject('sampling_regions_changed_with_rehashed_snapshot', lambda: sync_selection(lambda s:
            s['regions'][0].update(native_box=[0, 0, 312, 286])))
        reject('sampling_source_identity_changed', lambda: sync_selection(lambda s:
            s['run_binding'].update(source_sha256='0'*64)))
        reject('sampling_alpha_rebound', lambda: sync_selection(lambda s:
            s.update(alpha_binding=s['source_baseline_assets']['old_environment'])))
        reject('sampling_coverage_promoted', lambda: sync_selection(lambda s:
            s['coverage'][0].update(native_region_completely_certified=True)))
        reject('sampling_pair_list_reordered', lambda: sync_selection(lambda s: s['native_crops'].reverse()))
        reject('run_revision_changed', lambda: edit(path, lambda r: r['run_binding'].update(revision=999)))
        reject('run_final_hash_changed', lambda: edit(path, lambda r: r['run_binding'].update(final_sha256='0'*64)))
        reject('comparison_pixels_changed', lambda: change_pixel(qa/'comparison.png'))

        def forge_board_hash():
            change_pixel(qa/'comparison.png')
            sync_board(0, lambda b: b['comparison'].update(sha256=visual._sha(qa/'comparison.png')))
        reject('comparison_forged_hash_still_not_bound_pixel_replay', forge_board_hash)

        reject('api_input_pixels_changed', lambda: change_pixel(Path(result['reviews'][0]['api_input']['path'])))
        def forge_api_hash():
            sent = Path(result['reviews'][0]['api_input']['path'])
            change_pixel(sent)
            provenance_file = qa/'whole/vision-provenance.json'
            provenance = read(provenance_file)
            provenance['api_input']['sha256'] = visual._sha(sent)
            provenance['input_sha256'] = visual._sha(sent)
            visual._write(provenance_file, provenance)
            sync_board(0, lambda b: b.update(vision_provenance=provenance,
                vision_provenance_sha256=visual._sha(provenance_file), api_input=provenance['api_input']))
        reject('api_forged_hash_still_not_bound_board_transport_pixels', forge_api_hash)

        def native_unobserved(source, output, prompt, **kwargs):
            current, provenance = transport(source, output, prompt, **kwargs)
            if current['schema']==visual.NATIVE_OBSERVATION_SCHEMA:
                for row in current['pairs']:
                    row['checks']['native_detail'] = verdict('final', status='unknown', cause='unresolved')
            return current, provenance
        unknown = produce(root/'unknown', native_unobserved)
        with patch.object(visual.vision_transport, 'request_vision_json', side_effect=AssertionError('No API allowed')):
            checked_unknown, unknown_actions, _ = refinement._bound_review(prepared['source_path'],
                run['reference']['path'], root/'unknown/review.json', design)
        assert checked_unknown == unknown
        assert checked_unknown['checks']['native_detail']['final']['status'] == 'unknown'
        assert checked_unknown['native_sampling_coverage']['unobserved_pairs']
        assert any(a['criterion'] == 'skin_materials' for a in unknown_actions)
        passed.append('uncited_native_pairs_remain_unknown_with_valid_reference_diagnosis')

        # The production material consumer must accept v2 via the unchanged ABI.
        import automatic_material_completion as material
        run['cosplay_post_review'] = result
        run['generation_design'] = design
        checked_paths = material._diagnosis(root, run, root)
        assert checked_paths == (path, prepared['source_path'], Path(run['source']['path']), Path(run['reference']['path']))
        passed.append('automatic_material_completion_production_diagnosis_accepts_v2')
        import reference_material_targets as targets
        source_binding = {'image': result['binding']['images']['source']}
        diagnosis = {'evidence': refinement._file(path), 'regions': ['face', 'left_arm', 'right_arm']}
        checked_target = targets._checked_diagnosis(diagnosis, source_binding)
        assert checked_target['reason'] == checked['checks']['skin_materials']['reference']['reason']
        passed.append('reference_material_targets_production_diagnosis_accepts_v2')
        try:
            targets._checked_diagnosis(diagnosis, {'image': {'sha256': '0'*64}})
        except ValueError:
            passed.append('reference_material_targets_rejects_wrong_source_owner')
        else:
            raise AssertionError('Material target entry accepted a different source owner')
        for name, operation in (
                ('softened_reference', lambda r: r['checks']['skin_materials'].update(reference=verdict('reference'))),
                ('forged_native_pair_observation', lambda r: r['reviews'][1].update(native_pairs_actually_observed=['invented']))):
            edit(path, operation)
            changed_diagnosis = {**diagnosis, 'evidence': refinement._file(path)}
            try:
                targets._checked_diagnosis(changed_diagnosis, source_binding)
            except ValueError:
                passed.append('reference_material_targets_rejects_v2_'+name)
            else:
                raise AssertionError('Material target entry bypassed v2 evidence validation: '+name)
            finally:
                reset()
    print(json.dumps({'schema': 'saved_comprehensive_review_smoke.v1', 'status': 'passed',
        'cases_passed': len(passed), 'cases': passed, 'tamper_rejection_reasons': rejected,
        'network_requests': 0, 'model_requests': 0, 'generation_requests': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
