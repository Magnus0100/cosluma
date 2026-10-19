"""A model registry copy must not invalidate a saved RAW/Beauty appearance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend_server import Backend
from creative_task_smoke import fixture, rejected
from generation_design import DESIGN_VERSION
from single_source_workflow import sha256, write_json


def main():
    with tempfile.TemporaryDirectory(prefix='saved-completion-') as temporary:
        root = Path(temporary)
        registry, sources, reference = fixture(root)
        copied = root/'user-config/models.json'
        copied.parent.mkdir()
        copied.write_bytes(registry.read_bytes())
        source_hash, reference_hash = sha256(Path(sources[0])), sha256(Path(reference))
        stack = {'source_sha256': source_hash, 'candidate_sha256': reference_hash,
            'source_baseline': {'source_precision': 'raw16_linear',
                'source_display': {'method': 'luminance_soft_shoulder'}},
            'layers': [{'kind': 'reference_appearance_stack'}]}
        parent = {'kind': 'single_input_run', 'run_id': 'saved', 'sample': 'portrait',
            'source': {'path': sources[0], 'sha256': source_hash},
            'reference': {'path': reference, 'sha256': reference_hash}, 'auto_draft': {'saved': True},
            'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready'},
            'options': {'render_mode': 'native_source_design', 'algorithm_profile': 'dinov3_vitl16',
                'model_registry': str(registry), 'appearance_revision': {
                    'portrait_beauty': {'result': 'owned-frozen-layer.json', 'strength': 1., 'disabled_faces': []},
                    'design_reconstruction_plan': stack}}}
        manifest = root/'outputs/runs/saved/manifest.json'
        write_json(manifest, parent)
        before = manifest.read_bytes()
        backend = Backend(root)
        with patch.object(backend, '_continue_checked_task', side_effect=lambda _parent, request, **kw: request) as submit:
            request = backend.complete_source_run('saved', {'model_registry': str(copied)})
            assert request['appearance_revision']['design_reconstruction_plan'] == stack
            assert request['appearance_revision']['portrait_beauty']['strength'] == 1.
            assert submit.call_count == 1
            changed = copy.deepcopy(json.loads(copied.read_text('utf8')))
            changed['dinov3_model_root'] = str(root/'other-model')
            copied.write_text(json.dumps(changed), 'utf8')
            rejected(lambda: backend.complete_source_run('saved', {'model_registry': str(copied)}))
            rejected(lambda: backend.complete_source_run('saved', {'algorithm_profile': 'different-profile'}))
            assert submit.call_count == 1, 'Changed analysis must not queue a broken RAW child'
        assert manifest.read_bytes() == before
        error = Backend._safe_workflow_error({'exit_code': 2, 'stderr': '',
            'run': {'error': {'message': 'Source preparation baseline failed; token=private-value'}}})
        assert error['source'] == 'workflow_error' and 'Source preparation baseline failed' in error['message']
        assert 'private-value' not in error['message'] and '[redacted]' in error['message']
    print('saved completion contracts passed')


if __name__ == '__main__':
    main()
