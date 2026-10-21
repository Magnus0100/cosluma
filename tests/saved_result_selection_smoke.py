"""Saved-result order and explicit selection cannot be hijacked by review mtime."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image

import workbench_store as store


def write_saved(root, identifier, created, **changes):
    directory = root / 'outputs/runs' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    final, master = directory / 'draft.png', directory / 'master.tif'
    Image.new('RGB', (8, 8), '#345678').save(final)
    Image.new('RGB', (8, 8), '#345678').save(master)
    run = {'run_id': identifier, 'kind': 'single_input_run', 'sample': 'portrait',
        'created_at': created, 'revision': 1, 'status': 'needs_visual_review',
        'source': {'path': str(final)}, 'auto_draft': {'saved': True},
        'artifacts': {'auto_draft': str(final), 'render_master': str(master)}}
    run['result_publication'] = {'schema':'cosplay_saved_result_publication.v1',
        'scope':'local_save','state':'committed','run_id':identifier}
    run.update(changes)
    (directory / 'manifest.json').write_text(json.dumps(run), encoding='utf8')
    return run


def main():
    with tempfile.TemporaryDirectory(prefix='saved-result-contract-') as temporary:
        root = Path(temporary)
        write_saved(root, 'old', '2026-10-14T01:00:00+00:00')
        with patch.object(store, '_now', return_value='2026-10-14T01:01:00+00:00'):
            assert store.record_recent(root, sample='portrait', run_id='old')['recorded']
        write_saved(root, 'new', '2026-10-14T02:00:00+00:00')
        with patch.object(store, '_now', return_value='2026-10-14T02:01:00+00:00'):
            store.record_recent(root, sample='portrait', run_id='new')
        pointer = root / 'outputs/deliveries/active.json'
        pointer.parent.mkdir(parents=True); pointer.write_text('{"run_id":"approved-version"}')
        approved_bytes = pointer.read_bytes()
        store.select_saved_result(root, 'old', explicit=True)
        os.utime(root / 'outputs/runs/old/manifest.json', (2000000000, 2000000000))
        with patch.object(store, '_now', return_value='2026-10-14T09:00:00+00:00'):
            store.record_recent(root, sample='portrait', run_id='old')
        assert [row['run_id'] for row in store.list_recent(root)] == ['new', 'old']
        assert store.saved_result_state(root) == {'latest_saved_run_id': 'new',
            'selected_saved_run_id': 'old', 'selection_explicit': True}
        for identifier, changes in [('candidate', {'status': 'candidate_ready'}),
                ('failed', {'status': 'failed'}),
                ('pending', {'continuation_edit': {'render_state': 'pending'}})]:
            write_saved(root, identifier, '2026-10-14T10:00:00+00:00', **changes)
            assert not store.record_recent(root, sample='portrait', run_id=identifier)['recorded']
        assert store.saved_result_state(root)['latest_saved_run_id'] == 'new'
        assert pointer.read_bytes() == approved_bytes
        # Automatic candidate loading cannot hijack an explicit main version,
        # including loading that same version without a new user selection.
        index = root/'outputs/settings/recent_projects.json'
        chosen_bytes = index.read_bytes()
        store.select_saved_result(root, 'new')
        store.select_saved_result(root, 'old')
        assert index.read_bytes() == chosen_bytes
        assert store.saved_result_state(root)['selected_saved_run_id'] == 'old'
        real_replace = store.os.replace
        attempts = []
        def transient_replace(source, target):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError('transient Windows reader')
            return real_replace(source, target)
        with patch.object(store.os, 'replace', side_effect=transient_replace):
            store.select_saved_result(root, 'new', explicit=True)
        assert len(attempts) == 3
        index = root/'outputs/settings/recent_projects.json'
        stable = index.read_bytes()
        with patch.object(store.os, 'replace', side_effect=PermissionError('persistent file lock')):
            try:
                store.select_saved_result(root, 'old', explicit=True)
            except PermissionError:
                pass
            else:
                raise AssertionError('Persistent publication failure was hidden')
        assert index.read_bytes() == stable and not list(index.parent.glob('*.tmp'))
        # Fresh asynchronous validation may commit without ever waiting on the
        # GUI thread; contention, cancellation and stale manifests preserve it.
        from cache_store import exclusive_file_lock
        digest=hashlib.sha256((root/'outputs/runs/old/manifest.json').read_bytes()).hexdigest()
        with exclusive_file_lock(index.with_suffix('.lock')):
            started=time.perf_counter()
            try:store.select_saved_result(root,'old',explicit=True,lock_timeout=0,expected_manifest_sha256=digest)
            except TimeoutError:pass
            else:raise AssertionError('Nonwaiting selection ignored the held lock')
            assert time.perf_counter()-started<.25
        assert index.read_bytes()==stable
        for options in ({'guard':lambda:False},{'expected_manifest_sha256':'0'*64}):
            try:store.select_saved_result(root,'old',explicit=True,lock_timeout=0,**options)
            except ValueError:pass
            else:raise AssertionError('Stale guarded selection was published')
            assert index.read_bytes()==stable
        calls=[]
        def late_cancel():
            calls.append(1);return len(calls)==1
        try:store.select_saved_result(root,'old',explicit=True,lock_timeout=0,guard=late_cancel)
        except ValueError:pass
        else:raise AssertionError('Late cancelled selection was committed')
        assert len(calls)==2 and index.read_bytes()==stable and not list(index.parent.glob('*.tmp'))
        with patch.object(store.os,'replace',side_effect=PermissionError('nonwaiting replace')) as replace:
            try:store.select_saved_result(root,'old',explicit=True,lock_timeout=0)
            except PermissionError:pass
            else:raise AssertionError('Nonwaiting replace error hidden')
        assert replace.call_count==1 and index.read_bytes()==stable
        store.select_saved_result(root,'old',explicit=True,lock_timeout=0,
            expected_manifest_sha256=digest,guard=lambda:True)
        assert store.saved_result_state(root)['selected_saved_run_id']=='old'
        with patch.object(store,'saved_result_state',side_effect=AssertionError('GUI must not scan all saved runs')):
            assert store.select_saved_result(root,'new',explicit=True,lock_timeout=0,
                guard=lambda:True,return_state=False) is None
        assert store.saved_result_state(root)['selected_saved_run_id']=='new'

    import qt_workbench as qt
    from design_reconstruction_ui_smoke import Window, Pool, last, write_run
    from portrait_beauty_workbench_smoke import run_fixture
    app = qt.QApplication.instance() or qt.QApplication([])
    windows = []
    with tempfile.TemporaryDirectory(prefix='saved-result-startup-ui-') as temporary:
        root = Path(temporary)
        old, _, _, final, _ = run_fixture(root)
        old.update(created_at='2026-10-14T01:00:00+00:00', auto_draft={'saved': True})
        old['options'].update(provider='qwen_image_2_1_uncensored_fast', model='saved-old-workflow.json')
        old['generation_design'] = {'input_prompt':'retired managed prose', 'prompt_template_version':'retired'}
        master = root / 'master.tif'; Image.open(final).save(master)
        old['artifacts']['render_master'] = str(master)
        new = copy.deepcopy(old); new.update(run_id='new-saved-child', revision=2,
            created_at='2026-10-14T02:00:00+00:00', parent_run_id=old['run_id'])
        for run in (old, new):
            run['result_publication'] = {'schema':'cosplay_saved_result_publication.v1',
                'scope':'local_save','state':'committed','run_id':run['run_id']}
        write_run(root, old); write_run(root, new)
        with patch.object(store, '_now', return_value='2026-10-14T02:01:00+00:00'):
            store.record_recent(root, sample=new['sample'], run_id=new['run_id'])
        window = Window(root, 'http://127.0.0.1:1'); window.pool = Pool(); windows.append(window)
        window.provider.setCurrentText('qwen_image_2_1'); window.model.setText('current-quality-workflow.json')
        response = {'recent': store.list_recent(root), 'selection': store.saved_result_state(root)}
        window.result('recent', response)
        assert last(window.pool, 'startup-result-loaded').path.endswith('/' + new['run_id'])
        window.result('startup-result-loaded', {'run': new})
        assert window.source_run_id == new['run_id']
        assert window.provider.currentText() == 'qwen_image_2_1'
        assert window.model.text() == 'current-quality-workflow.json'
        from generation_prompt import DEFAULT_PROMPT
        assert window.prompt.toPlainText() == DEFAULT_PROMPT
        window.result('history-run-selected', {'run': old})
        assert store.saved_result_state(root)['selection_explicit']
        assert store.saved_result_state(root)['latest_saved_run_id'] == new['run_id']
        second = Window(root, 'http://127.0.0.1:1'); second.pool = Pool(); windows.append(second)
        second.result('recent', {'recent': store.list_recent(root), 'selection': store.saved_result_state(root)})
        assert last(second.pool, 'startup-result-loaded').path.endswith('/' + old['run_id'])
        second.result('startup-result-loaded', {'run': old})
        assert second.source_run_id == old['run_id']
        assert store.saved_result_state(root)['selection_explicit']
        second.load_source_run(new)
        assert second.source_run_id == new['run_id'], 'Candidate should remain viewable'
        assert store.saved_result_state(root)['selected_saved_run_id'] == old['run_id']
        assert store.saved_result_state(root)['selection_explicit']
        with patch('preparation_currentness.preparation_currentness', return_value={'state': 'stale'}):
            second.show_preparation_version(old)
        assert '旧算法' in second.preparation_version_review.text()
        assert '重新制作' in second.preparation_version_review.text()
        third = Window(root, 'http://127.0.0.1:1'); third.pool = Pool(); windows.append(third)
        third.result('recent', response)
        third.source_edit.setText(str(root/'a-new-user-input.png'))
        third.result('startup-result-loaded', {'run': new})
        assert not third.source_run_id, 'Late startup restore replaced newer user input'
        for window in windows:
            window.session_timer.stop(); window.native_matte_timer.stop(); window.source_light_timer.stop()
            window._session_checkpoint_failed = False; window.close()
        app.processEvents()
    print(json.dumps({'ok': True, 'save_order_not_review_mtime': True,
        'latest_selected_delivery_separate': True, 'incomplete_results_not_promoted': True,
        'restart_latest_or_explicit_selection': True, 'late_restore_preserves_user_input': True,
        'old_algorithm_action_visible': True, 'model_inferences': 0}))


if __name__ == '__main__':
    main()
