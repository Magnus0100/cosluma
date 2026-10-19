"""Real tiny saved pixels stay private until the production receipt commits."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from PIL import Image

from backend_server import Backend
from job_source_progress_smoke import checked_start
import portable_project
import single_source_workflow as workflow
import workbench_store as store


def terminal(backend, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = backend.jobs.get(identifier)
        if job['status'] in {'succeeded', 'failed', 'blocked', 'cancelled'}:
            return job
        time.sleep(.02)
    raise AssertionError('CPU fixture did not finish')


def main():
    # Preserve the fixture for review; never touch installed/workspace data.
    root = Path(tempfile.mkdtemp(prefix='result-publication-'))
    source = root/'source.png'; Image.new('RGB', (32, 24), '#456789').save(source)
    real_write = workflow.write_json
    blocked = []
    def fail_final_contract(path, value):
        if Path(path).name == 'manifest.json' and value.get('production_contract') and value.get('run_id') != 'old':
            blocked.append(str(path))
            raise PermissionError('CPU fixture: persistent production contract write failure')
        return real_write(path, value)
    with patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}), \
            patch('source_alpha.source_silhouette', side_effect=lambda _root, _source, image, **kw:
                (Image.new('L', image.size, 255), {'backend': 'CPU fixture matte'})):
        old = workflow.create_run(workflow.parse_args(['--root', str(root), '--source', str(source),
            '--run-id', 'old', '--local-only']))
        assert old['result_publication']['scope'] == 'local_save'
        assert old['result_publication']['state'] == 'committed'
        store.select_saved_result(root, 'old', explicit=False)
        index = root/'outputs/settings/recent_projects.json'
        initial_index = index.read_bytes()
        initial_state = store.saved_result_state(root)
        old_manifest = root/'outputs/runs/old/manifest.json'
        old_bytes = old_manifest.read_bytes()
        backend = Backend(root, embedded=True)
        with patch.object(workflow, 'write_json', side_effect=fail_final_contract):
            started = checked_start(backend, {'source': str(source), 'local_only': True,
                'generate': False, 'parent_run_id': 'old', 'preserve_project_selection': True})
            failed = terminal(backend, started['job_id'])
        assert failed['status'] == failed['execution_status'] == 'failed'
        failed_path, draft = workflow.load_run(root, failed['run_id'])
        assert blocked and draft['result_publication']['state'] == 'pending'
        assert not draft.get('production_contract') and draft['status'] == 'needs_visual_review'
        for key in ('auto_draft', 'render_master', 'render_scene_master'):
            assert Path(draft['artifacts'][key]).is_file()
        assert store.saved_result_state(root) == initial_state
        assert index.read_bytes() == initial_index
        assert failed['run_id'] not in {row['run_id'] for row in store.list_recent(root)}
        assert not store.record_recent(root, sample=draft['sample'], run_id=draft['run_id'])['recorded']
        try:
            store.select_saved_result(root, draft['run_id'])
        except ValueError:
            pass
        else:
            raise AssertionError('An uncommitted checked result became the selected result')
        assert index.read_bytes() == initial_index and old_manifest.read_bytes() == old_bytes
        assert workflow.load_run(root, failed['run_id'])[1]['artifacts'] == draft['artifacts']

        succeeded = checked_start(backend, {'source': str(source), 'local_only': True,
            'generate': False, 'parent_run_id': 'old', 'preserve_project_selection': True})
        successful_job = terminal(backend, succeeded['job_id'])
        assert successful_job['status'] == 'succeeded', successful_job
        _, saved = workflow.load_run(root, successful_job['run_id'])
        assert saved['result_publication']['state'] == 'committed'
        assert saved['result_publication']['scope'] == 'checked_production'
        assert saved['production_contract']['plan_id'] == saved['task_plan']['plan_id']
        assert store.saved_result_state(root)['latest_saved_run_id'] == saved['run_id']
        assert store.saved_result_state(root)['selected_saved_run_id'] == 'old'
        assert old_manifest.read_bytes() == old_bytes

        # Independent explicit saves/copies own their completion; they neither
        # inherit a pending publication nor invent a new production receipt.
        adjusted = workflow.adjust_source_run(root, saved['run_id'], denoise_strength=0)
        assert adjusted['result_publication']['scope'] == 'local_save'
        assert adjusted['result_publication']['state'] == 'committed'
        assert store.saved_result_state(root)['latest_saved_run_id'] == adjusted['run_id']
        copied = workflow.copy_source_run(root, adjusted['run_id'])
        assert copied['result_publication']['run_id'] == copied['run_id']
        assert copied['result_publication']['state'] == 'committed'
        assert store.saved_result_state(root)['latest_saved_run_id'] == copied['run_id']

        # A verified portable import can retain an immutable committed owner.
        # The old identity is provenance, never rewritten as a new commit.
        import_roots = []
        for exported in (saved, copied):
            archive = root/(exported['run_id']+'.cosplay')
            portable_project.export_project(root, exported['run_id'], archive)
            imported_root = Path(tempfile.mkdtemp(prefix='result-publication-import-'))
            import_roots.append(str(imported_root))
            imported = portable_project.import_project(imported_root, archive, run_id='relocated-copy')
            imported_run = workflow.load_run(imported_root, imported['run_id'])[1]
            assert imported_run['result_publication'] == exported['result_publication']
            assert store.saved_result_state(imported_root)['latest_saved_run_id'] == 'relocated-copy'
        # Unpublished records cannot enter the current saved-result index.
        unpublished = json.loads(old_bytes); unpublished.pop('result_publication')
        workflow.write_json(old_manifest, unpublished)
        assert not store.record_recent(root, sample=unpublished['sample'], run_id='old')['recorded']
        old_manifest.write_bytes(old_bytes)
        assert failed_path.is_file() and Path(draft['artifacts']['render_master']).is_file()

        # Completion can save an intermediate child before returning a final
        # one. All such writes remain pending in the same transaction.
        nested_root = Path(tempfile.mkdtemp(prefix='result-publication-nested-'))
        nested_backend = Backend(nested_root, embedded=True)
        actual_create = workflow.create_run
        intermediates = []
        def create_with_child(args):
            intermediate = actual_create(args)
            intermediates.append(intermediate['run_id'])
            assert store.saved_result_state(nested_root)['latest_saved_run_id'] == ''
            return workflow.adjust_source_run(nested_root, intermediate['run_id'], denoise_strength=0)
        with patch.object(workflow, 'create_run', side_effect=create_with_child):
            nested_started = checked_start(nested_backend, {'source': str(source),
                'local_only': True, 'generate': False, 'preserve_project_selection': True})
            nested_job = terminal(nested_backend, nested_started['job_id'])
        assert nested_job['status'] == 'succeeded', nested_job
        assert nested_job['run_id'] != intermediates[0]
        assert workflow.load_run(nested_root, intermediates[0])[1]['result_publication']['state'] == 'pending'
        assert [row['run_id'] for row in store.list_recent(nested_root)] == [nested_job['run_id']]
    print(json.dumps({'ok': True, 'audit_root': str(root), 'import_roots': import_roots,
        'nested_root': str(nested_root),
        'contract_failure_retains_draft_without_promotion': True, 'contract_success_publishes': True,
        'independent_adjust_copy_import_supported_and_unpublished_rejected': True,
        'model_inferences': 0, 'formal_data_modified': False, 'fixtures_retained': True}))


if __name__ == '__main__':
    main()
