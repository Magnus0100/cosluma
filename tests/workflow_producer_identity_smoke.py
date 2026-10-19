"""An implementation upgrade invalidates planned reuse and its descendants."""
from unittest.mock import patch
import copy
import json

from cos_node_registry import compile_workflow
from cos_workflow_cache import compare, producer_identity_for_node


def main():
    old = compile_workflow('existing_reference')
    observed = []

    def changed(node_id, implementation):
        identity = copy.deepcopy(producer_identity_for_node(node_id, implementation))
        if node_id == 'colour':
            identity['code']['reference_adaptive_fusion.py'] = 'updated-producer-sha'
        observed.append(node_id)
        return identity

    with patch('cos_workflow_cache.producer_identity_for_node', side_effect=changed):
        new = compile_workflow('existing_reference')
    difference = compare(old, new)
    assert difference['changed'] == ['colour'], difference
    assert 'colour' in difference['invalidated']
    assert {'boundary', 'composite', 'export'} <= set(difference['invalidated'])
    assert {'source', 'matte', 'reference', 'beauty', 'background'} <= set(difference['reused'])
    assert old['graph_id'] != new['graph_id']
    assert old['cache']['signature_reuse_is_not_a_verified_pixel_cache_hit']
    assert set(observed) == set(new['order'])
    for dependency in ('reference_boundary_light_transfer.py',
                       'reference_cloth_boundary_transfer.py'):
        def boundary_changed(node_id, implementation):
            identity = copy.deepcopy(producer_identity_for_node(node_id, implementation))
            if node_id == 'boundary':
                assert dependency in identity['code']
                identity['code'][dependency] = 'updated-boundary-sha'
            return identity
        with patch('cos_workflow_cache.producer_identity_for_node', side_effect=boundary_changed):
            updated = compile_workflow('existing_reference')
        boundary_difference = compare(old, updated)
        assert boundary_difference['changed'] == ['boundary'], boundary_difference
        assert set(boundary_difference['invalidated']) == {'boundary', 'composite', 'export'}, boundary_difference
    print(json.dumps({'ok': True, 'producer_change_invalidates_descendants': True,
        'unrelated_nodes_reusable': True, 'plan_identity_changes': True,
        'pixel_validation_owned_by_worker': True}))


if __name__ == '__main__':
    main()
