"""Offline native diagnostic input checks using real frozen baseline assets."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import cosplay_native_review as native
import cosplay_post_review as review
from cosplay_post_review_smoke import fake_transport, response, verdict


def make_native_fixture(root, native_size=(1100, 1400)):
    """A fully bound synthetic native plan, never a real/cache model sample."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    width, height = native_size
    source, reference, final = [root/(name+'.png') for name in ('source', 'reference', 'auto_draft')]
    Image.new('RGB', native_size, (8, 20, 35)).save(source)
    Image.new('RGB', (110, 140), (90, 60, 20)).save(reference)
    y, x = np.indices((height, width))
    final_values = np.stack((x % 256, y % 256, (x+y) % 256), -1).astype(np.uint8)
    Image.fromarray(final_values).save(final)
    source_sha = native._sha(source)
    assets = {}
    for key, values in (
            ('foreground_premultiplied', np.full((height, width, 3), .475, np.float32)),
            ('foreground_alpha', np.full((height, width), .75, np.float32)),
            ('old_environment', np.full((height, width, 3), .95, np.float32))):
        path = root/(key+'.npz')
        np.savez_compressed(path, values=values)
        assets[key] = {'path': str(path), 'sha256': native._sha(path),
                       'size': list(native_size), 'coordinate_space': 'source_canvas'}
    labels = np.zeros((140, 110), np.uint8)
    labels[30:60, 35:65] = 3
    labels[1:3, 1:3] = 3  # Tiny disjoint noise must not select the face window.
    labels[100:123, 80:98] = 6
    labels[100:105, 15:20] = 15
    semantic_path = root/'frozen-source-labels.npz'
    np.savez_compressed(semantic_path, labels=labels, native_labels=np.zeros((4, 3), np.uint8))
    semantic_sha = native._sha(semantic_path)
    from sapiens_model_profiles import profile
    from local_semantics_worker import CLASS_NAMES
    spec=profile('sapiens2_seg_0.4b','segmentation')
    semantic_metadata={'model':spec['repository'],'model_profile':'sapiens2_seg_0.4b','model_revision':spec['revision'],'checkpoint_sha256':spec['checkpoint_sha256'],'taxonomy_sha256':hashlib.sha256(json.dumps(CLASS_NAMES,separators=(',',':')).encode()).hexdigest()}
    identity = {'source_sha256': source_sha, 'review_semantics_sha256': semantic_sha}
    layer = {'kind': 'contact_shadow', 'assets': {'source_shoe_semantics': {
        'path': str(semantic_path), 'sha256': semantic_sha, 'coordinate_space': 'source_proxy',
        'role': 'source_cached_left_right_shoe_semantics',**semantic_metadata}}, 'provenance': {
        'contract_schema': native.CONTACT_CONTRACT, 'producer_recipe': native.CONTACT_RECIPE,
        'source_sha256': source_sha, 'asset_bindings': {'source_shoe_semantics_sha256': semantic_sha}}}
    plan = {'schema_version':'design_reconstruction_plan.v1','source_sha256': source_sha, 'working_domain': 'display_linear_sRGB',
        'canvas': {'source_oriented_size': list(native_size), 'source_proxy_size': [110, 140]},
        'execution': {'source_reference': 'paired_native_baseline'},
        'source_baseline': {'schema':'appearance_source_baseline.v1','working_domain':'display_linear_sRGB','source_sha256': source_sha,
                            'source_display_applied_once': True, 'assets': assets,
                            'foreground_bundle_identity': copy.deepcopy(identity)},
        'provenance': {'source_foreground': {'identity': copy.deepcopy(identity)}}, 'layers': [layer]}
    return {'run_id': 'native-offline-fixture', 'revision': 3,
        'source': {'path': str(source), 'sha256': source_sha, 'oriented_size': list(native_size)},
        'reference': {'path': str(reference), 'sha256': native._sha(reference)},
        'artifacts': {'auto_draft': str(final)},
        'options': {'render_mode':'native_source_design','appearance_revision': {'design_reconstruction_plan': plan}}}


def add_frozen_feature_fixture(run, root):
    """Synthetic source-owned observation chain; no inferred/photo coordinates."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    labels_record = plan['layers'][0]['assets']['source_shoe_semantics']
    from local_semantics_worker import CLASS_NAMES
    from sapiens_model_profiles import profile
    specification = profile('sapiens2_seg_0.4b', 'segmentation')
    display = root/'source-display.png'
    with Image.open(run['source']['path']) as image:
        image.resize((110, 140), Image.Resampling.LANCZOS).save(display)
    analysis_path = root/'analysis.json'
    analysis = {'source': run['source'], 'analysis_role': 'source', 'semantic_result': {
        'model_profile': 'sapiens2_seg_0.4b', 'model': specification['repository'],
        'model_revision': specification['revision'], 'checkpoint_sha256': specification['checkpoint_sha256'],
        'class_names': CLASS_NAMES, 'labels_path': labels_record['path']}}
    analysis_path.write_text(json.dumps(analysis), encoding='utf-8')
    masks = {}
    for name in ('eyes', 'lips'):
        values = np.zeros((140, 110), np.float32)
        if name == 'eyes':
            values[40:44, 40:44] = 1; values[40:44, 56:60] = 1
        else:
            values[52:55, 47:53] = 1
        path = root/(name+'.npz'); np.savez_compressed(path, values=values)
        masks[name] = {'path': str(path), 'sha256': native._sha(path)}
    spec = {'role': 'source', 'domain': 'display_linear_sRGB', 'binding': {
        'image': {'path': str(display), 'sha256': native._sha(display)}, 'identity_root': run['source'],
        'analysis': {'path': str(analysis_path), 'sha256': native._sha(analysis_path)},
        'semantic_labels': labels_record, 'feature_masks': masks}}
    inputs_path = root/'inputs.json'
    inputs_path.write_text(json.dumps({'schema': 'material_target_inputs.v2', 'source_observation': spec}), encoding='utf-8')
    receipt_path = root/'receipt.json'
    receipt_path.write_text(json.dumps({'schema': 'qualified_material_finish.v1', 'input': {
        'path': str(inputs_path), 'sha256': native._sha(inputs_path)}}), encoding='utf-8')
    plan['layers'].append({'kind': 'reference_appearance_stack', 'assets': {'material_finish_receipt': {
        'path': str(receipt_path), 'sha256': native._sha(receipt_path)}}})
    return spec


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='current-native-review-') as folder:
        root=Path(folder);run=make_native_fixture(root/'inputs')
        original={str(path):native._sha(path) for path in (root/'inputs').iterdir()}
        with patch('local_photo_analysis.prepare_local_appearance',side_effect=AssertionError('No models')), patch('vision_transport.request_vision_json',side_effect=AssertionError('No API')):
            first=native.prepare_cosplay_native_review(run,root/'first')
            second=native.prepare_cosplay_native_review(run,root/'second')
        assert first['receipt']['schema']=='cosplay_native_review_inputs.v2' and first['receipt']['status']=='ready'
        assert native._sha(first['source_path'])==native._sha(second['source_path'])
        assert [(row['pair_id'],row['native_box']) for row in first['receipt']['regions']]==[(row['pair_id'],row['native_box']) for row in second['receipt']['regions']]
        pairs={}
        for item in first['native_crops']:
            pairs.setdefault(item['pair_id'],set()).add(item['role'])
            raster,parent=review._raster(item['path']),review._raster(item['native_image'])
            assert raster.tobytes()==parent.crop(item['box']).tobytes()
            assert raster.width<=312 and raster.height<=286
        assert pairs['left_hand']==pairs['right_hand']=={'source','final'}
        assert all(roles=={'source','final'} for roles in pairs.values())
        assert not first['receipt']['coverage_limits']['whole_native_image_verified']
        assert any(row['region']=='hair' for row in first['receipt']['unsupported_regions'])
        checks.extend(('current_v2_only','deterministic_source_bytes_and_windows','exact_paired_pixels','unobserved_regions_unknown'))
        def plan(r):return r['options']['appearance_revision']['design_reconstruction_plan']
        for label,mutation in (
            ('missing_model_identity',lambda p:p['layers'][0]['assets']['source_shoe_semantics'].pop('model_profile')),
            ('wrong_model_hash',lambda p:p['layers'][0]['assets']['source_shoe_semantics'].update(checkpoint_sha256='0'*64)),
            ('wrong_taxonomy',lambda p:p['layers'][0]['assets']['source_shoe_semantics'].update(taxonomy_sha256='0'*64)),
            ('missing_source_labels',lambda p:p['layers'][0]['assets'].pop('source_shoe_semantics')),
            ('retired_contact_contract',lambda p:p['layers'][0]['provenance'].update(contract_schema='contact_receiver_contract.v3')),
            ('wrong_source_semantics_binding',lambda p:p['provenance']['source_foreground']['identity'].update(review_semantics_sha256='0'*64))):
            changed=copy.deepcopy(run);mutation(plan(changed))
            unknown=native.prepare_cosplay_native_review(changed,root/('mutation-'+label))
            assert unknown['receipt']['status']=='unknown' and not unknown['native_crops']
            assert not list((root/('mutation-'+label)).glob('native-*.png')),label
            checks.append(label)
        for label,mutation in (
            ('retired_scene_baseline',lambda r:plan(r).update(working_domain='scene_linear_sRGB')),
            ('missing_current_baseline_schema',lambda r:plan(r)['source_baseline'].pop('schema')),
            ('changed_baseline_asset',lambda r:plan(r)['source_baseline']['assets']['foreground_alpha'].update(sha256='0'*64))):
            changed=copy.deepcopy(run);mutation(changed)
            try:native.prepare_cosplay_native_review(changed,root/'invalid')
            except ValueError:pass
            else:raise AssertionError(label+' accepted')
            checks.append(label)
        try:native.prepare_cosplay_native_review(run,root/'retired',comprehensive=False)
        except ValueError:pass
        else:raise AssertionError('retired narrow-v1 QA accepted')
        checks.append('retired_narrow_v1_rejected')
        assert original=={str(path):native._sha(path) for path in (root/'inputs').iterdir()}
        feature_run=make_native_fixture(root/'feature-inputs')
        add_frozen_feature_fixture(feature_run, root/'feature-observation')
        before={str(path):native._sha(path) for path in (root/'feature-observation').iterdir()}
        featured=native.prepare_cosplay_native_review(feature_run,root/'featured')
        assert featured['receipt']['status']=='ready',featured['receipt']['reason']
        assert {r['pair_id'] for r in featured['receipt']['regions']} >= {'eyes_0','eyes_1','lips'}
        assert len(featured['receipt']['feature_binding']['origin_records'])==2
        assert featured['receipt']['context_regions']
        assert before=={str(path):native._sha(path) for path in (root/'feature-observation').iterdir()}
        checks.append('automatic_frozen_source_observation_selects_both_eyes_and_lips')
        changed=copy.deepcopy(feature_run)
        plan(changed)['layers'][-1]['assets']['material_finish_receipt']['sha256']='0'*64
        invalid=native.prepare_cosplay_native_review(changed,root/'invalid-feature')
        assert invalid['receipt']['status']=='unknown' and not invalid['native_crops']
        checks.append('bad_frozen_feature_origin_fails_closed_without_model_fallback')
        try:native.prepare_cosplay_native_review(feature_run,root/'featured')
        except FileExistsError:pass
        else:raise AssertionError('Existing native evidence overwritten')
        checks.append('existing_native_evidence_is_immutable')
    print(json.dumps({'ok':True,'checks':checks,'input_assets_unchanged':True,'api_calls':0,'gpu_calls':0}))


if __name__=='__main__':main()
