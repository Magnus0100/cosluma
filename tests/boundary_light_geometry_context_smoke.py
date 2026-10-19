"""Exact EDT handoff and request/source/stat/coordinate geometry isolation."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.ndimage import distance_transform_edt

from boundary_light_geometry_context import (COORDINATE_RECIPE, ExactMaskDistance,
    NativeClothGeometry, current_geometry_context, geometry_key, request_geometry_scope)
from reference_boundary_light_transfer import _contour


def main():
    mask = np.zeros((16, 21), bool); mask[3:13, 4:17] = True
    distance = ExactMaskDistance.compute(mask.copy())
    assert np.array_equal(distance.for_mask(mask), distance_transform_edt(mask))
    alpha = mask.astype(np.float32)
    original = _contour(alpha)
    shared = _contour(alpha, inside_depth=distance)
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(original, shared))
    invalid = mask.copy(); invalid[4, 6] = False
    try:
        distance.for_mask(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('A different native mask consumed a retained EDT')
    try:
        _contour(alpha, inside_depth=distance.distance)
    except TypeError:
        pass
    else:
        raise AssertionError('A saved distance array was accepted as live geometry')
    backing = np.pad(mask, ((1, 1), (1, 1)))
    view = backing[1:-1, 1:-1]
    view_distance = ExactMaskDistance.compute(view)
    assert view.flags.writeable and view_distance.mask.flags.owndata and view_distance.mask.base is None
    backing[5, 7] = False
    try:
        view_distance.for_mask(view)
    except ValueError:
        pass
    else:
        raise AssertionError('A writable mask backing changed a retained EDT')

    with tempfile.TemporaryDirectory(prefix='boundary-geometry-') as temporary:
        root = Path(temporary)
        path = root/'input.bin'; path.write_bytes(b'observed current input')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        asset = {'path': str(path), 'sha256': digest}
        run = {'source': copy.deepcopy(asset), 'reference': copy.deepcopy(asset)}
        plan = {'source_sha256': digest, 'candidate_sha256': digest,
            'canvas': {'source_oriented_size': [21, 16], 'source_proxy_size': [21, 16]},
            'mask_roles': {name: copy.deepcopy(asset) for name in
                ('foreground_alpha', 'source_semantics', 'identity_protection', 'candidate_occupancy')},
            'layers': [{'kind': 'reference_appearance_stack', 'assets': {'delta': copy.deepcopy(asset)}},
                       {'kind': 'background_design', 'assets': {'validity': copy.deepcopy(asset)}}]}
        proof = {name: copy.deepcopy(asset) for name in ('map', 'receipt', 'reference_labels')}
        proof['input_binding'] = {'source_sha256': digest, 'reference_sha256': digest,
                                  'source_labels_sha256': digest, 'reference_labels_sha256': digest}
        owners = np.where(mask, 23, 0).astype(np.uint8)
        yy, xx = np.indices(mask.shape, dtype=np.float32)
        geometry = NativeClothGeometry(alpha.copy(), owners, {'ref_x': xx, 'ref_y': yy,
            'confidence': mask.astype(np.float32), 'valid': mask.copy()}, np.zeros(mask.shape, np.float32),
            np.ones(mask.shape, np.float32), ExactMaskDistance.compute(owners == 23))
        assert geometry.owners is owners
        assert geometry.maps['ref_x'] is not xx and geometry.maps['ref_x'].flags.owndata
        alias_buffers = [np.pad(values, ((1, 1), (1, 1))) for values in
            (alpha, owners, np.zeros(mask.shape, np.float32), np.ones(mask.shape, np.float32))]
        alias_fields = [values[1:-1, 1:-1] for values in alias_buffers]
        alias_map_backing = np.pad(yy, ((1, 1), (1, 1)))
        alias_map = alias_map_backing[1:-1, 1:-1]
        detached = NativeClothGeometry(alias_fields[0], alias_fields[1], {'ref_y': alias_map},
            alias_fields[2], alias_fields[3], ExactMaskDistance.compute(alias_fields[1] == 23))
        retained = [getattr(detached, name).copy() for name in ('alpha', 'owners', 'protection', 'validity')]
        retained_map = detached.maps['ref_y'].copy()
        for values in alias_buffers:
            values[4, 6] = 99
        alias_map_backing[4, 6] = 99
        for name, values in zip(('alpha', 'owners', 'protection', 'validity'), retained):
            assert np.array_equal(getattr(detached, name), values)
            assert getattr(detached, name).base is None and getattr(detached, name).flags.owndata
        assert np.array_equal(detached.maps['ref_y'], retained_map) and detached.maps['ref_y'].base is None
        for values in (geometry.alpha, geometry.owners, geometry.validity, *geometry.maps.values()):
            assert not values.flags.writeable
        try:
            geometry.maps['valid'] = np.zeros(mask.shape, bool)
        except TypeError:
            pass
        else:
            raise AssertionError('Readonly native maps were replaced')
        held = {}
        @request_geometry_scope
        def check():
            context = current_geometry_context()
            key = geometry_key(root, run, plan, proof, (0, 0, 21, 16), (21, 16), (21, 16), maps=geometry.maps)
            changed = [replace(key, source_sha256='a'*64), replace(key, reference_sha256='b'*64),
                replace(key, box=(1, 0, 21, 16)), replace(key, coordinate_recipe=COORDINATE_RECIPE+'.changed'),
                replace(key, assets=key.assets[:-1]), replace(key, reference_size=(22, 16)),
                replace(key, observation_reference_size=(11, 8)), replace(key, submitted_geometry='changed')]
            for wrong in changed:
                context.store(key, geometry)
                assert context.take(wrong) is None
            # Adding the newly prepared hair layer does not change geometry.
            after = copy.deepcopy(plan)
            after['layers'].append({'kind': 'source_boundary_light', 'assets': {}})
            same = geometry_key(root, run, after, proof, (0, 0, 21, 16), (21, 16), (21, 16), maps=geometry.maps)
            assert same == key
            context.store(key, geometry)
            assert context.take(same) is geometry and context.take(same) is None
            context.store(key, geometry)
            changed_plan = copy.deepcopy(plan)
            changed_plan['mask_roles']['foreground_alpha']['sha256'] = 'c'*64
            wrong = geometry_key(root, run, changed_plan, proof, (0, 0, 21, 16), (21, 16), (21, 16), maps=geometry.maps)
            assert wrong != key and context.take(wrong) is None
            changed_maps = dict(geometry.maps)
            changed_maps['ref_x'] = geometry.maps['ref_x'].copy()
            changed_maps['ref_x'][0, 0] += np.float32(.25)
            wrong = geometry_key(root, run, plan, proof, (0, 0, 21, 16), (21, 16), (21, 16), maps=changed_maps)
            assert wrong != key
            context.store(key, geometry); assert context.take(wrong) is None
            path.write_bytes(b'changed current input!')
            changed_stat = geometry_key(root, run, plan, proof, (0, 0, 21, 16), (21, 16), (21, 16), maps=geometry.maps)
            assert changed_stat != key
            context.store(key, geometry); assert context.take(changed_stat) is None
            context.store(changed_stat, geometry)
            held.update(context=context, key=changed_stat)
        check()
        assert current_geometry_context() is None
        assert held['context']._cloth is None
        try:
            held['context'].take(held['key'])
        except ValueError:
            pass
        else:
            raise AssertionError('Geometry from a completed request was replayed')
        @request_geometry_scope
        def next_request():
            context = current_geometry_context()
            try:
                context.store(held['key'], geometry)
            except ValueError:
                pass
            else:
                raise AssertionError('Geometry key crossed preparation requests')
        next_request()
        @request_geometry_scope
        def failed_request():
            raise RuntimeError('synthetic stage failure')
        try:
            failed_request()
        except RuntimeError:
            pass
        assert current_geometry_context() is None
    # Exercise the actual production samplers and geometry builder: the RGB
    # stage consumes the exact live preflight object, with no second transform.
    import cloth_boundary_worker as cloth_worker
    import design_reconstruction as design
    from cloth_boundary_runtime_smoke import fixture
    with tempfile.TemporaryDirectory(prefix='boundary-native-geometry-') as temporary:
        root = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            run, _, _, _, _, _, _ = fixture(root/'inputs', with_hair=False)
        plan = design.plan_for_run(run)
        plan['mask_roles']['candidate_occupancy'] = plan['mask_roles']['foreground_alpha']
        proof = {name: plan['mask_roles']['source_semantics'] for name in ('map', 'receipt', 'reference_labels')}
        proof['input_binding'] = {'source_sha256': run['source']['sha256'],
                                  'reference_sha256': run['reference']['sha256']}
        yy, xx = np.indices((32, 48), dtype=np.float32)
        maps = {'ref_x': xx, 'ref_y': yy, 'confidence': np.ones_like(xx), 'valid': np.ones(xx.shape, bool)}
        original_map = cloth_worker.map_coordinates
        original_edt = distance_transform_edt
        @request_geometry_scope
        def current_pair():
            with patch.object(cloth_worker, 'map_coordinates', side_effect=original_map) as sampled, \
                 patch('scipy.ndimage.distance_transform_edt', side_effect=original_edt) as edt:
                preflight = cloth_worker.native_geometry(root, run, plan, maps, proof, (0, 0, 48, 32), (48, 32), (48, 32))
                actual = cloth_worker.native_geometry(root, run, plan, maps, proof, (0, 0, 48, 32), (48, 32), (48, 32),
                                                     reuse_preflight=True)
                assert actual is preflight and sampled.call_count == 4 and edt.call_count == 1
                assert all(values.flags.owndata and values.base is None for values in
                           (actual.alpha, actual.owners, actual.protection, actual.validity, *actual.maps.values()))
            return actual
        actual = current_pair()
        with patch.object(cloth_worker, 'map_coordinates', side_effect=original_map) as sampled, \
             patch('scipy.ndimage.distance_transform_edt', side_effect=original_edt) as edt:
            independent = cloth_worker.native_geometry(root, run, plan, maps, proof, (0, 0, 48, 32), (48, 32), (48, 32),
                                                      reuse_preflight=True)
            assert sampled.call_count == 4 and edt.call_count == 1
        for name in ('alpha', 'owners', 'protection', 'validity'):
            assert np.array_equal(getattr(actual, name), getattr(independent, name))
        assert all(np.array_equal(actual.maps[name], independent.maps[name]) for name in maps)
        assert np.array_equal(actual.owner_distance.distance, independent.owner_distance.distance)
    print(json.dumps({'ok': True, 'exact_scipy_contour_bits': True, 'readonly_native_geometry': True,
        'source_reference_asset_stat_roi_coordinates_bound': True, 'no_cross_request_reuse': True,
        'exception_scope_cleared': True, 'actual_preflight_to_RGB_four_maps_once_EDT_once': True,
        'models': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
