"""Reject ambiguous, cross-material, folded and corrupt local correspondence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from reference_local_correspondence import (local_correspondence_identity, validate_local_correspondence,
    array_sha, file_sha, SCHEMA, canonical_sha, _face_observation, DINO3_L_PROFILE)
from reference_local_correspondence_worker import (mutual_matches, geometric_matches, build_maps,
    _rasterize_triangles)


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Invalid local correspondence accepted')


def main():
    yy, xx = np.mgrid[:6, :6]
    points = np.stack([xx*16+16, yy*16+16], axis=-1).reshape(-1, 2).astype(float)
    source = {'features': np.eye(36, dtype=np.float32), 'points': points,
              'labels': np.ones(36, np.int32), 'size': (112, 112), 'patch_size':16}
    reference = {**source, 'points': points+7}
    labels = np.ones((112, 112), np.uint8)
    rows, _ = mutual_matches(source, reference)
    assert len(rows) == 36
    accepted, _ = geometric_matches(rows, source, reference)
    assert len(accepted) == 36
    maps, reports = build_maps(source, reference, labels, labels)
    assert maps['valid'].sum() > 4000
    yy, xx = np.nonzero(maps['valid'])
    assert np.max(abs(maps['ref_x'][yy, xx]-xx-7)) < 1e-4
    assert np.max(abs(maps['ref_y'][yy, xx]-yy-7)) < 1e-4
    assert not maps['valid'][0].any() and reports['regions']['clothing']['status'] == 'partial'
    assert reports['regions']['face']['status'] == 'unsupported'
    assert reports['regions']['lips']['status'] == 'unsupported'
    ambiguous = {**reference, 'features': np.ones_like(reference['features'])/6}
    assert not mutual_matches(ambiguous, ambiguous)[0]
    different = {**reference, 'labels': np.full(36, 4, np.int32)}
    assert not mutual_matches(source, different)[0]
    folded = {**reference, 'points': reference['points'].copy()}
    folded['points'][:, 0] = 112-folded['points'][:, 0]
    assert not geometric_matches(rows, source, folded)[0]
    damaged = {**reference, 'points': reference['points'].copy()}
    damaged['points'][14] += [40, -30]
    bad, _ = geometric_matches(rows, source, damaged)
    assert not any(row['source_index'] == 14 for row in bad)
    empty = {'ref_x': np.full(labels.shape, -1., np.float32), 'ref_y': np.full(labels.shape, -1., np.float32),
             'confidence': np.zeros(labels.shape, np.float32), 'valid': np.zeros(labels.shape, bool)}
    ref_labels = np.full_like(labels, 4)
    _rasterize_triangles(points, points+7, np.ones(36), labels, ref_labels, [1], empty, max_edge=56)
    assert not empty['valid'].any(), 'Raster sampling must also verify reference material'
    # A canonical landmark mesh may certify source feature colours without
    # DINO eye/lip matching; local folding must discard those triangles.
    gy, gx = np.mgrid[:18, :26]
    face_points = np.c_[gx.reshape(-1)*2.5+23, gy.reshape(-1)*3.5+23]
    face = {'landmarks': face_points.tolist(), 'size': [112, 112]}
    target_face = {**face, 'landmarks': (face_points+4).tolist()}
    face_labels = np.full_like(labels, 3)
    no_dino = {**source, 'labels': np.zeros(36, np.int32)}
    facial, diag = build_maps(no_dino, no_dino, face_labels, face_labels, face, target_face)
    assert facial['valid'].sum() > 2000 and diag['face']['valid_landmarks'] > 400
    # Eyes live inside semantic class 3, while class 2 denotes eyeglasses.
    # Canonical source/ref eye polygons are an extra exact ownership gate.
    source_eye = [[33,33],[43,33],[43,43],[33,43]]
    face['protected_regions']=[{'region':'left_eye','polygon':source_eye}]
    target_face['protected_regions']=[{'region':'left_eye','polygon':(np.asarray(source_eye)+4).tolist()}]
    facial, diag = build_maps(no_dino, no_dino, face_labels, face_labels, face, target_face)
    assert diag['regions']['left_eye']['source_pixels'] > 50 and diag['regions']['left_eye']['valid_pixels'] > 50
    shifted = copy.deepcopy(target_face);shifted['protected_regions'][0]['polygon']=(np.asarray(source_eye)+40).tolist()
    facial, diag = build_maps(no_dino, no_dino, face_labels, face_labels, face, shifted)
    assert diag['regions']['left_eye']['valid_pixels'] == 0
    distorted = copy.deepcopy(target_face)
    distorted['landmarks'] = (np.c_[112-face_points[:, 0], face_points[:, 1]]).tolist()
    folded_face, _ = build_maps(no_dino, no_dino, face_labels, face_labels, face, distorted)
    assert not folded_face['valid'].any()
    missing, _ = build_maps(no_dino, no_dino, face_labels, face_labels)
    assert not missing['valid'].any(), 'No face observations cannot imply face or feature mapping'
    with tempfile.TemporaryDirectory(prefix='local_correspondence_') as directory:
        asset = Path(directory)/'maps.npz'
        np.savez_compressed(asset, **maps)
        receipt = {'schema': SCHEMA, 'identity':{'algorithm_profile':DINO3_L_PROFILE}, 'binding': {'proxy_size': [112, 112]}, 'reference_size': [112, 112],
                   'asset': {'path': str(asset), 'sha256': file_sha(asset)}}
        assert validate_local_correspondence(receipt)['valid'].sum() > 4000
        rejected(lambda:validate_local_correspondence(receipt, binding={'proxy_size':[56,56]}))
        corrupt = {key:value.copy() for key,value in maps.items()}
        corrupt['ref_x'][20, 20] = float('nan')
        np.savez_compressed(asset, **corrupt)
        rejected(lambda:validate_local_correspondence(receipt))
        receipt['asset']['sha256'] = file_sha(asset)
        rejected(lambda:validate_local_correspondence(receipt))
    assert array_sha(labels) != array_sha(labels.astype(np.int32))
    assert array_sha(labels) != array_sha(labels.T.copy()[:56])
    root = Path(__file__).resolve().parents[1]
    identity = local_correspondence_identity(root)
    assert canonical_sha(_face_observation({**face,'cache_hit':True,'field_path':'A:/x'})) == canonical_sha(
        _face_observation({**face,'cache_hit':False,'field_path':'B:/x'})), 'Locators/cache flags are not face evidence'
    import reference_local_correspondence as module
    from current_model_registry import load_registry
    installed = load_registry(root)
    with tempfile.TemporaryDirectory() as empty_model:
        with patch('current_model_registry.load_registry', return_value={**installed, 'dinov3_model_root':empty_model}):
            rejected(lambda:module.local_correspondence_identity(root))
    changed = copy.deepcopy(identity);changed['matching']['minimum_cosine'] += .01
    assert canonical_sha(changed) != canonical_sha(identity)
    assert identity['model']['revision'] and identity['model']['files_sha256']['model.safetensors']
    print(json.dumps({'ok': True, 'mutual_and_ambiguity': True, 'exact_semantic_gate': True,
        'local_neighbour_heldout_geometry': True, 'no_folded_triangle': True,
        'no_extrapolation': True, 'face_landmark_only_features': True,
        'changed_input_recipe_model_bound': True, 'corrupt_assets_rejected': True}))


if __name__ == '__main__':
    main()
