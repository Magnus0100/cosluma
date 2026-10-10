"""Test preparation cache identity and file closure without fitting or models."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import reference_pipeline as pipeline
from display_transform import source_transform_for_precision
from raw_support import RGB_DECODE_VERSION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def asset(root, name, content=b'fixture'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {'path': str(path), 'sha256': pipeline._sha(path), 'size': [4, 6]}


def main():
    with tempfile.TemporaryDirectory(prefix='reference-input-identity-') as temporary, \
            patch('runtime_resources.code_path', side_effect=lambda name: Path(temporary)/'scripts'/name):
        root = Path(temporary)
        for name in pipeline.CODE_DEPENDENCIES:
            asset(root, 'scripts/' + name, name.encode())
        source, reference, alpha, protection, background, validity, fp = [asset(root, name, name.encode())
            for name in ('source', 'reference', 'alpha', 'protection', 'background', 'validity', 'fp')]
        source_mask, reference_mask = [asset(root, name, name.encode())
            for name in ('source-mask', 'reference-mask')]
        objects = {'schema': 'reference_object_masks.v1', 'object': 'bag', 'label': 200,
            'visual_reviewed': True, 'roles': {}}
        for role, image, mask in (('source', source, source_mask), ('reference', reference, reference_mask)):
            objects['roles'][role] = {'input_sha256': image['sha256'], 'mask_path': mask['path'],
                'mask_sha256': mask['sha256'], 'box_prompt': [0, 0, 3, 5]}
        edit = {'schema': 'reference_authored_chroma.v1', 'source_sha256': source['sha256'],
            'reference_sha256': reference['sha256'], 'working_domain': 'display_linear_sRGB',
            'adjustments': [{'strength': .2, 'colour': [.1, .2, .3]}]}
        material_source, material_reference, material_qualification = [asset(root, name, name.encode())
            for name in ('material-source-observation', 'material-reference-observation', 'material-qualification')]
        material = {'schema': 'material_target_inputs.v2',
            'source_observation': {'binding': {'image': source, 'semantic_labels': material_source}},
            'reference_observation': {'binding': {'image': reference, 'semantic_labels': material_reference}},
            'qualification': material_qualification,
            'diagnosis': {'regions': ['face'], 'evidence': material_source}}
        paths = {key: root / (key + '.json') for key in pipeline.EXTRA_INPUTS}
        records = {'object_masks': objects, 'authored_colour_edit': edit,
                   'material_targets': material}
        for key, record in records.items(): write_json(paths[key], record)
        plan = {'schema_version':'design_reconstruction_plan.v1',
            'source_sha256': source['sha256'], 'candidate_sha256': reference['sha256'],
            'enabled': True, 'canvas': {'source_oriented_size': [4, 6], 'source_proxy_size': [2, 3],
                'candidate_mapping': {'candidate_original_to_final_original': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
            'mask_roles': {'foreground_alpha': alpha, 'identity_protection': protection},
            'working_domain': 'display_linear_sRGB',
            'source_baseline': {'schema':'appearance_source_baseline.v1', 'working_domain':'display_linear_sRGB',
                'source_sha256':source['sha256'], 'source_precision':'raw16_linear', 'decode_version':RGB_DECODE_VERSION,
                'source_display_applied_once':True, 'source_display':source_transform_for_precision('raw16_linear'),
                'assets': {'foreground_premultiplied': fp, 'foreground_alpha': alpha}},
            'layers': [{'kind': 'background_design', 'assets': {'background': background, 'validity': validity}}]}
        run = {'source': source, 'reference': reference,
            'reference_pipeline_inputs': {key: str(path) for key, path in paths.items()},
            'options': {'appearance_revision': {'design_reconstruction_plan': plan}}}
        plan['provenance']={'local_model_registry':pipeline._registry_identity(root,run)}
        original = copy.deepcopy(run)
        identity = pipeline._input_identity(root, run)
        assert identity['schema'] == pipeline.INPUT_CONTRACT
        assert run == original
        result_plan = copy.deepcopy(plan)
        snapshot_path = root / 'new-output/preparation-snapshot.json'
        write_json(snapshot_path, {'recipe': pipeline.RECIPE})
        snapshot_asset = {'path': str(snapshot_path), 'sha256': pipeline._sha(snapshot_path)}
        result_plan['layers'].insert(0, {'kind': 'reference_appearance_stack', 'params': {'alpha_sha256': alpha['sha256']},
            'assets': {'preparation_snapshot': snapshot_asset}, 'enabled': True, 'strength': 1.})
        result = {'ok': True, 'design_reconstruction_plan': result_plan}
        pipeline._stamp_inputs(root, result, identity)
        with patch('reference_appearance_stack.validate_layer'):
            assert pipeline._prepared_matches(root, run, result)
            snapshot = json.loads(snapshot_path.read_text(encoding='utf8'))
            snapshot['pipeline_inputs']['code']['foreground_effects.py'] = 'retired-fx-consumer'
            write_json(snapshot_path, snapshot)
            snapshot_asset['sha256'] = pipeline._sha(snapshot_path)
            assert not pipeline._prepared_matches(root, run, result), 'Retired identity keys must invalidate cache'
            snapshot['pipeline_inputs']['code'].pop('foreground_effects.py')
            write_json(snapshot_path, snapshot)
            snapshot_asset['sha256'] = pipeline._sha(snapshot_path)
            saved = copy.deepcopy(snapshot)
            snapshot['pipeline_inputs']['code']['reference_colour_blend.py'] = 'changed-core-producer'
            write_json(snapshot_path, snapshot)
            snapshot_asset['sha256'] = pipeline._sha(snapshot_path)
            assert not pipeline._prepared_matches(root, run, result), 'Actual appearance producer changed'
            write_json(snapshot_path, saved)
            snapshot_asset['sha256'] = pipeline._sha(snapshot_path)

            # Changing paths, including paths inside JSON receipts, preserves
            # the identity when every effective choice and byte stays equal.
            moved = copy.deepcopy(run)
            for role in ('source', 'reference'):
                item = moved[role]
                target = root / 'moved' / role
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(item['path']).read_bytes())
                item['path'] = str(target)
            for key, record in records.items():
                record = copy.deepcopy(record)
                if key == 'object_masks':
                    for role, value in record['roles'].items():
                        target = root / 'moved' / (role + '-mask')
                        target.write_bytes(Path(value['mask_path']).read_bytes())
                        value['mask_path'] = str(target)
                target = root / 'moved' / (key + '.json')
                write_json(target, record)
                moved['reference_pipeline_inputs'][key] = str(target)
            assert pipeline._input_identity(root, moved) == identity
            assert pipeline._prepared_matches(root, moved, result)

            # Content changes with stale declarations must fail closed, never
            # receive the old preparation simply because filenames match.
            for item in (source, reference, alpha, protection, background, validity, source_mask, reference_mask,
                         material_source, material_reference, material_qualification):
                path = Path(item['path']); old = path.read_bytes(); path.write_bytes(old + b' changed')
                assert not pipeline._prepared_matches(root, run, result), path
                path.write_bytes(old)
            for key in records:
                record = copy.deepcopy(records[key])
                if key == 'object_masks': record['visual_reviewed'] = False
                elif key == 'authored_colour_edit': record['adjustments'][0]['strength'] = .7
                elif key == 'material_targets': record['diagnosis']['regions'] = ['hands']
                write_json(paths[key], record)
                assert not pipeline._prepared_matches(root, run, result), key
                write_json(paths[key], records[key])
            for role, item in (('source', source_mask), ('reference', reference_mask)):
                path = Path(item['path']); old = path.read_bytes(); path.write_bytes(old + b' revised')
                key = 'object_masks'
                record = copy.deepcopy(records[key])
                record['roles'][role]['mask_sha256'] = pipeline._sha(path)
                write_json(paths[key], record)
                assert not pipeline._prepared_matches(root, run, result), 'Correctly re-declared changed dependency'
                path.write_bytes(old); write_json(paths[key], records[key])
            for name, item in [('source', material_source), ('reference', material_reference),
                               ('qualification', material_qualification)]:
                path = Path(item['path']); old = path.read_bytes(); path.write_bytes(old+b' new accepted content')
                record = copy.deepcopy(material)
                if name == 'qualification': record['qualification']['sha256'] = pipeline._sha(path)
                else:
                    record[name+'_observation']['binding']['semantic_labels']['sha256'] = pipeline._sha(path)
                    if name == 'source': record['diagnosis']['evidence']['sha256'] = pipeline._sha(path)
                write_json(paths['material_targets'], record)
                assert not pipeline._prepared_matches(root, run, result), 'Changed material '+name+' file closure reused old preparation'
                path.write_bytes(old); write_json(paths['material_targets'], material)
            for change in ('display', 'mapping', 'proxy', 'asset_metadata'):
                altered = copy.deepcopy(run)
                ap = altered['options']['appearance_revision']
                if change == 'display': ap['design_reconstruction_plan']['source_baseline']['source_display'] = {'method': 'luminance_soft_shoulder', 'shoulder_start': .6}
                elif change == 'mapping': ap['design_reconstruction_plan']['canvas']['candidate_mapping']['candidate_original_to_final_original'][0][2] = 2
                elif change == 'proxy': ap['design_reconstruction_plan']['canvas']['source_proxy_size'] = [3, 4]
                else: ap['design_reconstruction_plan']['mask_roles']['foreground_alpha']['sampling'] = 'nearest'
                assert not pipeline._prepared_matches(root, altered, result), change
            altered = copy.deepcopy(run); altered.pop('reference_pipeline_inputs')
            assert not pipeline._prepared_matches(root, altered, result)
            altered = copy.deepcopy(run); altered['reference_pipeline_inputs']['unconsumed_option'] = True
            assert not pipeline._prepared_matches(root, altered, result)
            retired = copy.deepcopy(run); retired['reference_pipeline_inputs']['floor_repair'] = str(root/'retired.json')
            try: pipeline._input_identity(root, retired)
            except ValueError: pass
            else: raise AssertionError('Retired fixed-photo producer input was accepted')
            changed_code = root / 'scripts/evaluate_reference_colour.py'
            old = changed_code.read_bytes(); changed_code.write_bytes(old + b' change')
            assert not pipeline._prepared_matches(root, run, result)
            changed_code.write_bytes(old)

            # Every explicit registry must use the one current profile set.
            registry = root/'retired-models.json'
            write_json(registry, {'semantics_profile':'sapiens2_seg_1b','ready':False})
            altered = copy.deepcopy(run)
            altered['options']['model_registry'] = str(registry)
            try: pipeline._input_identity(root, altered)
            except ValueError: pass
            else: raise AssertionError('Retired registry was accepted')
            missing_registry = root/'missing-registry.json'
            altered = copy.deepcopy(run)
            altered['options']['model_registry'] = str(missing_registry)
            try: pipeline._input_identity(root, altered)
            except RuntimeError: pass
            else: raise AssertionError('Missing explicit registry silently used default models')
            with patch.dict(os.environ, {'COSPLAY_MODEL_REGISTRY':str(missing_registry)}):
                try: pipeline._input_identity(root, run)
                except RuntimeError: pass
                else: raise AssertionError('Missing environment registry silently used default models')

            # An unchanged prepared plan uses its original (possibly repaired)
            # environment input, while manual replay choices are not refitted.
            saved = copy.deepcopy(run)
            saved['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(result_plan)
            saved_plan = saved['options']['appearance_revision']['design_reconstruction_plan']
            saved_plan['working_domain'] = 'display_linear_sRGB'
            saved_plan['source_baseline']['source_display'] = source_transform_for_precision('raw16_linear')
            saved_plan['layers'][0]['strength'] = .3
            saved['options']['manual_layers'] = {'brush': {'strength': .9}}
            assert pipeline._input_identity(root, saved) == identity
            assert pipeline._prepared_matches(root, saved, result)
            saved_plan['source_baseline']['source_display']['shoulder_start'] = .6
            assert not pipeline._prepared_matches(root, saved, result)

            # Legacy saved snapshots remain untouched and replayable, but no
            # longer stand in for an exact dependency-bound prepare cache.
            stamped_bytes = snapshot_path.read_bytes()
            legacy = {'recipe': pipeline.RECIPE}
            write_json(snapshot_path, legacy)
            legacy_result = copy.deepcopy(result)
            legacy_result['design_reconstruction_plan']['layers'][0]['assets']['preparation_snapshot']['sha256'] = pipeline._sha(snapshot_path)
            assert not pipeline._prepared_matches(root, run, legacy_result)
            assert snapshot_path.read_bytes() == json.dumps(legacy).encode()
            snapshot_path.write_bytes(stamped_bytes)

            # prepare itself takes the cache fast path before any fitting, and
            # preserves the current edited alpha without rebuilding a scene.
            cache = root / 'outputs/cache/reference-pipeline/exact/appearance/prepared-plan.json'
            write_json(cache, result)
            with patch.object(pipeline, '_process', side_effect=AssertionError('Cache ran a model')), \
                 patch('design_reconstruction_pipeline.prepare_scene', side_effect=AssertionError('Valid alpha was reset')):
                hit = pipeline.prepare(root, run)
            assert hit['cache_hit'] is True and run == original
            assert hit['design_reconstruction_plan']['mask_roles']['foreground_alpha']['sha256'] == alpha['sha256']
            seed = root / 'outputs/inspection/reference-match-v2'
            weights = asset(root, 'outputs/inspection/reference-match-v2/source-weights.npz', b'weights')
            parameters = {'binding': {'source': source['sha256'], 'reference': reference['sha256'],
                'alpha': alpha['sha256'], 'algorithm_sha256': pipeline._sha(root / 'scripts/reference_colour_fit.py'),
                'evaluation_sha256': pipeline._sha(root / 'scripts/evaluate_reference_colour.py'),
                'source_weights_sha256': weights['sha256']}}
            write_json(seed / 'parameters.json', parameters)
            write_json(seed / 'composite-receipt.json', {'parameters_sha256': pipeline._sha(seed / 'parameters.json'),
                'renderer_sha256': pipeline._sha(root / 'scripts/render_reference_colour_composite.py'), 'scene_balance': True})
            write_json(seed / 'reference-colour-layer.json', {})
            write_json(seed / 'scene-balance-layer.json', {})
            assert not pipeline._diagnostic_ready(root, run, seed), 'Otherwise-valid legacy seed still needs input closure'
            write_json(seed / 'pipeline-inputs.json', identity)
            assert pipeline._diagnostic_ready(root, run, seed)
            altered = copy.deepcopy(run); altered.pop('reference_pipeline_inputs')
            assert not pipeline._diagnostic_ready(root, altered, seed), 'Same photo/matte does not imply same extras'
        print(json.dumps({'ok': True, 'content_and_indirect_file_closure': True,
            'path_independent_reuse': True, 'effective_parameters_and_code_invalidate': True,
            'exact_prepare_cache_hit_without_models': True, 'edited_alpha_not_reset': True,
            'legacy_cache_and_seed_not_trusted': True, 'saved_replay_inputs_not_mutated': True}))


if __name__ == '__main__':
    main()
