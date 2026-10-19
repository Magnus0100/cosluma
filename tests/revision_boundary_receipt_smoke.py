"""Preserve historical boundary facts and own their real qualification file."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import single_source_workflow as workflow


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay_boundary_receipt_') as directory:
        root = Path(directory)
        parent_dir = root/workflow.RUNS_DIR/'parent'
        parent_dir.mkdir(parents=True)
        qualification = parent_dir/'boundary-coordination-preparation/qualification.json'
        qualification.parent.mkdir()
        payload = b'{"state":"no_change","model_tile_inferences_this_attempt":0}\n'
        qualification.write_bytes(payload)
        boundary = {'status': 'no_change', 'elapsed_seconds': 12.5,
            'qualification_path': str(qualification), 'receipt': {'state': 'no_change',
                'model_tile_inferences_this_attempt': 0, 'producer_run_id': 'parent',
                'historical_asset': str(parent_dir/'historical.json')}}
        parent = {'kind': 'single_input_run', 'run_id': 'parent', 'sample': 'small',
            'revision': 1, 'source': {'path': str(root/'source.png')}, 'reference': None,
            'options': {}, 'auto_draft': True, 'artifacts': {},
            'boundary_light_preparation': boundary}
        manifest = parent_dir/'manifest.json'
        workflow.write_json(manifest, parent)
        before = manifest.read_bytes()
        with patch('design_reconstruction.render_mode_for_run', return_value='native_source_design'):
            for snapshot in (False, True):
                path, child = workflow._source_revision(root, 'parent', snapshot=snapshot)
                locator = Path(child['boundary_light_preparation']['qualification_path'])
                assert locator.is_file() and locator.is_relative_to(path.parent)
                assert locator.read_bytes() == payload
                facts = copy.deepcopy(child['boundary_light_preparation'])
                facts.pop('qualification_path')
                original = copy.deepcopy(boundary); original.pop('qualification_path')
                assert facts == original
                assert manifest.read_bytes() == before and qualification.read_bytes() == payload
            # Missing historical evidence is retained as its historical locator,
            # never rewritten to suggest a new child preparation was performed.
            qualification.unlink()
            _, child = workflow._source_revision(root, 'parent', snapshot=False)
            assert child['boundary_light_preparation'] == boundary
            # Existing relative locators are resolved from the project root.
            qualification.write_bytes(payload)
            parent['boundary_light_preparation']['qualification_path'] = qualification.relative_to(root).as_posix()
            workflow.write_json(manifest, parent)
            path, child = workflow._source_revision(root, 'parent', snapshot=False)
            locator = Path(child['boundary_light_preparation']['qualification_path'])
            assert locator.is_relative_to(path.parent) and locator.read_bytes() == payload
    print('OK: ordinary/snapshot revisions own existing absolute/relative qualification receipts; bytes and all historical facts unchanged; missing history keeps parent locator')


if __name__ == '__main__':
    main()
