"""Smoke the single checked-task boundary used by CLI and HTTP entrypoints."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cos_node_registry import compile_workflow
from production_coordinator import execute_workflow, route_contract
import creative_task
import single_source_workflow as workflow
import workflow_timing


def persistence_checks(request):
    """Exercise the real OS run lock without model, renderer or HTTP work."""
    with tempfile.TemporaryDirectory(prefix='production-coordinator-',
                                     dir=os.environ.get('COSPLAY_TEST_TEMP')) as temporary:
        root = Path(temporary)
        manifest = root/'outputs/runs/fixture/manifest.json'
        initial = {'kind': 'single_input_run', 'run_id': 'fixture',
            'source': {'sha256': request['task_plan']['inputs']['source']['sha256']},
            'task_intent': copy.deepcopy(request['task_intent']),
            'task_plan': copy.deepcopy(request['task_plan']),
            'status': 'needs_visual_review', 'delivery_status': 'auto_draft'}
        args = SimpleNamespace(root=root, **request)

        @workflow_timing.entry('create_run', lambda args: args.root)
        def create_fixture(_args):
            run = copy.deepcopy(initial)
            with workflow._run_id_lock(root, 'fixture'):
                workflow.write_json(manifest, run)
            return run

        # Pause after the coordinator reads the draft. The concurrent approval
        # must fail to acquire its lock until the contract save has finished;
        # otherwise its accepted state would be overwritten by that stale read.
        snapshot_read, acceptance_attempted, receipt_done = (threading.Event() for _ in range(3))
        acceptance_errors, acceptance_refused = [], []
        coordinator_thread = threading.get_ident()
        original_read = Path.read_text

        def read_with_interleave(path, *read_args, **read_kwargs):
            content = original_read(path, *read_args, **read_kwargs)
            if path == manifest and threading.get_ident() == coordinator_thread and not snapshot_read.is_set():
                snapshot_read.set()
                assert acceptance_attempted.wait(5), 'Concurrent approval did not attempt the run lock'
            return content

        def save_acceptance():
            with workflow._run_id_lock(root, 'fixture'):
                saved = json.loads(original_read(manifest, encoding='utf-8'))
                saved.update(status='delivered', delivery_status='delivered',
                             review={'status': 'accepted', 'reviewer': 'CPU fixture'}, next_action='done')
                workflow.write_json(manifest, saved)

        def acceptance_worker():
            try:
                assert snapshot_read.wait(5), 'Coordinator did not read the fixture'
                try:
                    save_acceptance()
                except FileExistsError:
                    acceptance_refused.append(True)
                    acceptance_attempted.set()
                    assert receipt_done.wait(5), 'Contract save did not finish'
                    save_acceptance()
                else:
                    acceptance_attempted.set()
            except BaseException as error:
                acceptance_errors.append(error)
                acceptance_attempted.set()

        worker = threading.Thread(target=acceptance_worker, name='approval-fixture')
        worker.start()
        try:
            with patch.object(workflow, 'create_run', create_fixture), patch.object(Path, 'read_text', read_with_interleave):
                result = execute_workflow(root, args)
        finally:
            receipt_done.set()
            worker.join(timeout=5)
        assert not worker.is_alive() and not acceptance_errors, acceptance_errors
        assert acceptance_refused == [True], 'Contract save did not own the existing run lock'
        saved = json.loads(original_read(manifest, encoding='utf-8'))
        assert saved['status'] == saved['delivery_status'] == 'delivered'
        assert saved['review']['status'] == 'accepted' and saved['next_action'] == 'done'
        assert saved['production_contract'] == route_contract(request)
        report = json.loads(original_read(Path(result['workflow_timing']['path']), encoding='utf-8'))
        names = [row['name'] for row in report['spans']]
        assert names.index('create_run') < names.index('production_contract_save')
        assert report['status'] == 'complete' and report['operation'] == 'production_workflow'
        assert abs(report['reconciliation']['difference_seconds']) < 1e-8
        assert str(manifest) in report['manifest_paths']
        assert saved['workflow_timing'] == result['workflow_timing']

        # If approval finished before the coordinator reacquires the lock,
        # preserve and return those latest fields instead of the stale draft.
        def already_accepted(_args):
            stale = create_fixture(_args)
            save_acceptance()
            return stale
        with patch.object(workflow, 'create_run', already_accepted):
            accepted = execute_workflow(root, args)
        assert accepted['status'] == 'delivered' and accepted['review']['status'] == 'accepted'

        # A concurrent --force replacement must retain its own source and plan;
        # the old request cannot stamp its contract onto the replacement.
        for changed_field in ('source', 'plan'):
            replacement = None
            def replaced(_args):
                nonlocal replacement
                stale = create_fixture(_args)
                replacement = copy.deepcopy(stale)
                if changed_field == 'source':
                    replacement['source']['sha256'] = 'c'*64
                else:
                    replacement['task_plan']['plan_id'] = 'd'*64
                with workflow._run_id_lock(root, 'fixture'):
                    workflow.write_json(manifest, replacement)
                return stale
            with patch.object(workflow, 'create_run', replaced):
                try:
                    execute_workflow(root, args)
                except RuntimeError as error:
                    assert 'source or checked task plan changed' in str(error)
                else:
                    raise AssertionError('Contract was attached to a different '+changed_field)
            preserved = json.loads(original_read(manifest, encoding='utf-8'))
            assert preserved == replacement and 'production_contract' not in preserved
        assert workflow_timing.current() is None


def main() -> int:
    intent = {"schema_version": creative_task.INTENT_SCHEMA, "mode": "local_portrait",
              "brief": "", "constraints": {"preserve_background": True, "no_effects": True},
              "services": {"text_vision": False}}
    intent = creative_task.normalize_task_intent(intent)
    workflow = compile_workflow("local_portrait")
    plan = {"schema_version": creative_task.PLAN_SCHEMA, "intent": intent, "mode": "local_portrait",
            "ready": True, "image_budget": creative_task.image_budget(intent),
            "plan_id": "a" * 64, "execution_strategy": "layered_reference_transfer",
            "layer_workflow": workflow,
            "inputs": {"source": {"sha256": "b" * 64}}}
    request = {"task_intent": intent, "task_plan": plan}
    receipt = route_contract(request)
    assert receipt["schema"] == "cosplay_production_contract.v1"
    assert receipt["graph_id"] == workflow["graph_id"]
    assert receipt["order"] == workflow["order"]
    broken = copy.deepcopy(request)
    broken["task_plan"]["execution_strategy"] = "unsupported_strategy"
    try:
        route_contract(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("route mismatch must be rejected before create_run")
    persistence_checks(request)
    print(json.dumps({"ok": True, "checked_route_receipt": True,
                      "graph_identity_recorded": True, "route_mismatch_rejected": True,
                      "concurrent_approval_preserved": True, "latest_acceptance_returned": True,
                      "replacement_identity_protected": True, "final_contract_save_timed": True,
                      "model_inferences": 0, "network_requests": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
