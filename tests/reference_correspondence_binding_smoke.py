"""Reject cross-photo/observation/model/asset substitutions at the B/S boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import reference_local_correspondence as producer
from reference_automatic_layers import (pack_correspondence, load_maps, sha,
                                       MAP_REPRESENTATION)
from reference_appearance_prepare import provenance_only


def rejected(operation):
    try:
        operation()
    except (ValueError, OSError, KeyError, RuntimeError):
        return
    raise AssertionError('A substituted local correspondence observation was accepted')


def main():
    identity = {'schema': producer.SCHEMA, 'algorithm_profile':producer.DINO3_L_PROFILE,
        'model': {'repository': 'ground-truth-fixture', 'revision': '1',
                  'files_sha256': {'weights': 'fixture-model'}},
        'preprocess': {'version': 'ground-truth-pixels'},
        'matching': {'minimum_confidence': .15}}
    with tempfile.TemporaryDirectory(prefix='correspondence-binding-') as folder, \
            patch.object(producer, 'local_correspondence_identity', return_value=identity):
        root = Path(folder); size = [10, 8]
        labels = np.full((8, 10), 1, np.uint8)
        faces = {'method': 'fixture', 'size': size, 'landmarks': [[1., 2.]],
                 'protected_regions': [], 'model_sha256': 'fixture-face', 'mediapipe_version': 'fixture'}
        files = {name: root/(name+'.bin') for name in ('source', 'reference', 'other')}
        for name, path in files.items():
            path.write_bytes(name.encode())
        run = {role: {'path': str(files[role]), 'sha256': sha(files[role]),
                      'oriented_size': size} for role in ('source', 'reference')}
        yy, xx = np.indices(labels.shape, dtype=np.float32)
        maps = {'ref_x': xx, 'ref_y': yy, 'confidence': np.ones(labels.shape, np.float32),
                'valid': np.ones(labels.shape, bool), 'material_labels': labels.astype(np.int32)}
        asset = root/'maps.npz'; np.savez_compressed(asset, **maps)
        binding = {'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'source_labels_sha256': producer.array_sha(labels),
            'reference_labels_sha256': producer.array_sha(labels), 'proxy_size': size,
            'source_faces_sha256': producer.canonical_sha(producer._face_observation(faces)),
            'reference_faces_sha256': producer.canonical_sha(producer._face_observation(faces)),
            'producer_identity_sha256': producer.canonical_sha(identity),
            'cached_rgb_sha256': {'source': None, 'reference': None}}
        receipt = {'schema': producer.SCHEMA, 'binding': binding, 'identity': identity,
            'source_size': size, 'reference_size': size, 'proxy_size': size,
            'coordinate_space': 'source_proxy', 'reference_coordinate_space': 'original_oriented_reference_pixels',
            'confidence_is_spatial_evidence_not_effect_dose': True, 'person_geometry_warped': False,
            'reference_texture_imported': False, 'background_H_used': False,
            'asset': {'path': str(asset), 'sha256': sha(asset), 'representation': MAP_REPRESENTATION},
            'regions': {'clothing': {'accepted_local_pixels': 80}},
            'diagnostics': {'triangles': [{'source': [[1, 1], [2, 1], [1, 2]]}]}}
        receipt_path = root/'receipt.json'

        def observed(value=receipt):
            receipt_path.write_text(json.dumps(value), encoding='utf8')
            return {'receipt': value, 'receipt_path': str(receipt_path), 'asset': value['asset']}

        context = {'root': root, 'source_labels': labels, 'reference_labels': labels,
                   'source_faces': faces, 'reference_faces': faces}
        params = {'binding': {role: run[role]['sha256'] for role in ('source', 'reference')}}
        result = observed()
        record = pack_correspondence(result, run, size, **context)
        np.testing.assert_array_equal(load_maps(record, labels.shape, labels.shape, params, **context)['valid'], maps['valid'])

        other_run = copy.deepcopy(run)
        other_run['source'].update(path=str(files['other']), sha256=sha(files['other']))
        rejected(lambda: pack_correspondence(result, other_run, size, **context))
        other_params = copy.deepcopy(params); other_params['binding']['source'] = sha(files['other'])
        relabelled_record = copy.deepcopy(record)
        relabelled_record['inputs']['source_sha256'] = sha(files['other'])
        rejected(lambda: load_maps(relabelled_record, labels.shape, labels.shape, other_params, **context))
        other_params = copy.deepcopy(params); other_params['binding']['reference'] = sha(files['other'])
        rejected(lambda: load_maps(record, labels.shape, labels.shape, other_params, **context))
        changed_labels = labels.copy(); changed_labels[2, 3] = 4
        for role in ('source_labels', 'reference_labels'):
            rejected(lambda role=role: load_maps(record, labels.shape, labels.shape, params,
                **{**context, role: changed_labels}))
        rejected(lambda: load_maps(record, labels.shape, labels.shape, params,
            **{**context, 'source_labels': labels.astype(np.int32)}))
        rejected(lambda: pack_correspondence(result, run, [8, 10], **context))
        changed_face = copy.deepcopy(faces); changed_face['landmarks'][0][0] += 1
        rejected(lambda: load_maps(record, labels.shape, labels.shape, params,
            **{**context, 'source_faces': changed_face}))
        for field in ('model', 'preprocess'):
            changed = copy.deepcopy(identity); changed[field]['revision' if field == 'model' else 'version'] = 'changed'
            with patch.object(producer, 'local_correspondence_identity', return_value=changed):
                rejected(lambda: load_maps(record, labels.shape, labels.shape, params, **context))
        different_asset = root/'other-maps.npz'
        np.savez_compressed(different_asset, **{**maps, 'confidence': np.full(labels.shape, .9, np.float32)})
        different = {'path': str(different_asset), 'sha256': sha(different_asset), 'representation': MAP_REPRESENTATION}
        rejected(lambda: pack_correspondence({**result, 'asset': different}, run, size, **context))
        rejected(lambda: load_maps({**record, 'asset': different}, labels.shape, labels.shape, params, **context))
        bad_body = copy.deepcopy(record); bad_body['receipt']['diagnostics'] = {}
        rejected(lambda: load_maps(bad_body, labels.shape, labels.shape, params, **context))
        for change in ('low_confidence', 'wrong_owner', 'nonfinite'):
            bad_maps = {key: value.copy() for key, value in maps.items()}
            if change == 'low_confidence': bad_maps['confidence'][0, 0] = .01
            elif change == 'wrong_owner': bad_maps['material_labels'][0, 0] = 4
            else: bad_maps['ref_x'][0, 0] = np.nan
            np.savez_compressed(asset, **bad_maps)
            bad_receipt = copy.deepcopy(receipt); bad_receipt['asset']['sha256'] = sha(asset)
            rejected(lambda: pack_correspondence(observed(bad_receipt), run, size, **context))
        frozen = provenance_only(record)
        assert frozen['receipt'] == record['receipt']
        assert producer.canonical_sha(frozen['receipt']) == frozen['receipt_content_sha256']
        assert frozen['receipt']['diagnostics']['triangles'] == receipt['diagnostics']['triangles']
        assert 'path' not in frozen['asset'] and 'receipt_path' not in frozen
        assert 'path' not in frozen['receipt']['asset']
    print(json.dumps({'ok': True, 'cross_photo_rejected': True,
        'exact_labels_dtype_proxy_faces_bound': True, 'current_model_preprocess_bound': True,
        'receipt_asset_substitution_rejected': True, 'minimum_confidence_owners_finite_checked': True,
        'path_independent_receipt_body_hash_preserved': True}))


if __name__ == '__main__':
    main()
