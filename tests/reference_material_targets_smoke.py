"""Current v2 evidence and qualification regressions; no validator mocks."""
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
import cosplay_post_review as visual
import reference_material_targets as evidence
import reference_material_targets_v2 as targets
from current_material_fixture import prepare_fixture
from reference_colour_blend import apply_colour_layer
from colour_space import to_lab


def main():
    cases=0
    with tempfile.TemporaryDirectory(prefix='current-material-v2-') as directory:
        fixture=prepare_fixture(Path(directory));root=Path(directory)
        source,reference,qualification,diagnosis,actual_review,anchor_review,record=[fixture[key] for key in
          ('source','reference','qualification','diagnosis','anchor_review','anchor_review_factory','record')]
        fitted = targets.fit_qualified_skin_colour(source, reference, qualification, diagnosis)
        field = fitted['source_ab_field']; original = source['rgb']; result = apply_colour_layer(original, field)
        cases += 1
        assert field.shape == (*source['labels'].shape, 3) and np.isfinite(field).all()
        clean_ab = np.array([.032, .018]); before = to_lab(original); after = to_lab(result)
        for name, ids in [('left_arm', [7]), ('right_arm', [16])]:
            owner = np.isin(source['labels'], ids)
            old_error = np.median(np.linalg.norm(before[owner, 1:]-clean_ab, axis=-1))
            new_error = np.median(np.linalg.norm(after[owner, 1:]-clean_ab, axis=-1))
            assert new_error < old_error*.55, (name, old_error, new_error)
        left = np.median(after[source['labels'] == 7, 1:], axis=0)
        right = np.median(after[source['labels'] == 16, 1:], axis=0)
        assert np.linalg.norm(left-right) < .012
        cases += 1
        assert np.max(abs(before[..., 0]-after[..., 0])) < 3e-6
        assert result.min() >= 0 and result.max() <= 1
        assert np.array_equal(result[source['labels'] == 0], original[source['labels'] == 0])
        # A low-frequency ab field may change ambient pigment appearance; it
        # retains feature-native texture instead of replacing it with skin.
        cases += 1
        assert np.linalg.norm(field[source['feature_masks']['eyes'] > .5, :2], axis=-1).mean() > .003
        assert np.linalg.norm(field[source['labels'] == 24, :2], axis=-1).mean() > .003
        assert np.median(after[source['labels'] == 24, 1]) > .08
        for feature in (source['feature_masks']['eyes'] > .5, source['labels'] == 24):
            assert np.std(after[feature, 1]) >= np.std(before[feature, 1])*.85
        cases += 1
        dx = np.diff(field[..., :2], axis=1); dy = np.diff(field[..., :2], axis=0)
        assert np.linalg.norm(dx, axis=-1).max() <= .04+1e-6
        assert np.linalg.norm(dy, axis=-1).max() <= .04+1e-6
        assert np.linalg.norm(field[..., :2], axis=-1).max() <= targets.SETTINGS['maximum_ab']+1e-6
        cases += 1
        rejected_statuses = {row['anchor_id']: 'rejected' for row in actual_review['anchors']}
        rejected = anchor_review('rejected-anchors', rejected_statuses)
        rejected_targets = targets.qualify_material_targets(source, reference, diagnosis, rejected)
        rejected_field = targets.fit_qualified_skin_colour(source, reference, rejected_targets, diagnosis)['source_ab_field']
        assert not rejected_field.any()
        unknown_statuses = {row['anchor_id']: 'unknown' for row in actual_review['anchors']}
        unknown = anchor_review('unknown-anchors', unknown_statuses)
        assert not targets.fit_qualified_skin_colour(source, reference,
            targets.qualify_material_targets(source, reference, diagnosis, unknown), diagnosis)['source_ab_field'].any()
        cases += 1
        forged = copy.deepcopy(rejected_targets)
        forged['targets'] = copy.deepcopy(qualification['targets'])
        forged['qualification_sha256'] = evidence._digest({key: value for key, value in forged.items() if key != 'qualification_sha256'})
        try: targets.fit_qualified_skin_colour(source, reference, forged, diagnosis)
        except ValueError: pass
        else: raise AssertionError('Rejected actual crop promoted into clean endpoint')
        cases += 1
        tampered = copy.deepcopy(qualification)
        tampered['targets']['skin'][0]['cells'][0][1] += .01
        tampered['qualification_sha256'] = evidence._digest({key: value for key, value in tampered.items() if key != 'qualification_sha256'})
        try: targets.fit_qualified_skin_colour(source, reference, tampered, diagnosis)
        except ValueError: pass
        else: raise AssertionError('Target cells altered after the actual crop review')
        for label, mutator in [('wrong-domain', lambda obs: obs.update(domain='scene_linear_RGB')),
                ('wrong-rgb', lambda obs: obs['rgb'].__setitem__((30, 60, 1), .8)),
                ('missing-feature-evidence', lambda obs: obs['binding']['feature_masks'].pop('eyes')),
                ('wrong-labels', lambda obs: obs['labels'].__setitem__((30, 60), 200))]:
            cases += 1; modified = copy.deepcopy(source); mutator(modified)
            try: evidence.candidate_material_pools(modified)
            except ValueError: pass
            else: raise AssertionError(label)
        cases += 1
        edited_review = copy.deepcopy(actual_review); edited_review['anchors'][0]['status'] = 'unknown'
        try: targets.qualify_material_targets(source, reference, diagnosis, edited_review)
        except ValueError: pass
        else: raise AssertionError('Anchor verdict edited after actual API evidence')
        cases += 1
        wrong_crop = copy.deepcopy(actual_review); wrong_crop['anchors'][0]['crop']['box'][0] += 1
        try: targets.qualify_material_targets(source, reference, diagnosis, wrong_crop)
        except ValueError: pass
        else: raise AssertionError('Crop from another material canvas silently used')
        cases += 1
        for label, mutator in [
                ('old-target-schema', lambda value:value.update(schema='qualified_material_targets.v1')),
                ('old-wrapper', lambda value:value.update(evidence_v1={})),
                ('changed-crop-panel', lambda value:value['targets']['skin'][0]['crop']['panel_box'].__setitem__(0, 1)),
                ('changed-cell-sampling', lambda value:value['targets']['skin'][0].update(cell_size=3)),
                ('wrong-material-region', lambda value:value['targets']['skin'][0].update(region='lips'))]:
            changed=copy.deepcopy(qualification);mutator(changed)
            changed['qualification_sha256']=evidence._digest({key:value for key,value in changed.items() if key!='qualification_sha256'})
            try:targets.fit_qualified_skin_colour(source,reference,changed,diagnosis)
            except ValueError:pass
            else:raise AssertionError(label)
            cases+=1
        base = np.clip(original*.92, 0, 1).astype(np.float32)
        base_path = root/'current-base.npz'; np.savez_compressed(base_path, values=base)
        recipe_path = root/'base-recipe.json'; visual._write(recipe_path, {'parent': source['binding']['image'], 'mode': 'frozen synthetic T base'})
        base_argument = {'rgb': base, 'binding': {'asset': record(base_path), 'recipe': record(recipe_path),
            'parent_source_sha256': source['binding']['image']['sha256']}}
        base_result = targets.fit_qualified_skin_colour(source, reference, qualification, diagnosis, source_base=base_argument)
        assert base_result['parameters']['source_colour_base']['basis'] == 'frozen_current_colour_base'
        assert not np.array_equal(base_result['source_ab_field'], field)
        cases += 1
        altered = copy.deepcopy(base_argument); altered['binding']['parent_source_sha256'] = '0'*64
        try: targets.fit_qualified_skin_colour(source, reference, qualification, diagnosis, source_base=altered)
        except ValueError: pass
        else: raise AssertionError('Current base attached to another original')

    print(json.dumps({'status':'passed','current_contract':'qualified_material_targets.v2','cases_passed':cases,
                      'network_requests':0,'gpu_requests':0}))


if __name__=='__main__':main()
